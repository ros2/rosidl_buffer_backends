// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Scoped Rust access to native CUDA buffers.
//!
//! [`rosidl_runtime_rs::Buffer`] owns message storage; scoped handles order GPU
//! access through native CUDA events. The `cuda-core` feature adds typed access
//! and retains the acquisition stream until handle cleanup. CUDA access handles
//! are neither `Send` nor `Sync`.

#![warn(unsafe_op_in_unsafe_fn)]

pub mod ffi;

#[cfg(feature = "cuda-core")]
mod cuda_core_adapter;
#[cfg(feature = "cuda-core")]
pub use cuda_core_adapter::{
    from_input_buffer, from_output_buffer, get_primitive_sequence_read_handle, to_buffer, CopyKind,
    CudaReadHandle, CudaWriteHandle,
};

use std::ffi::CStr;
use std::fmt;
use std::marker::PhantomData;
use std::os::raw::c_void;
use std::ptr::{self, NonNull};

use rosidl_runtime_rs::{Buffer, PrimitiveSequence};

/// A borrowed CUDA stream for the low-level buffer API.
///
/// Null selects the backend's internal stream, not CUDA default stream 0.
/// Use the `cuda-core` API to access buffers on the default stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaStream(*mut c_void);

impl CudaStream {
    /// Defer to the backend's process-wide internal stream.
    pub const INTERNAL: Self = Self(ptr::null_mut());

    /// Wrap an existing `cudaStream_t` without taking ownership.
    ///
    /// Null is equivalent to [`Self::INTERNAL`].
    ///
    /// # Safety
    ///
    /// `raw` must be null or a live `cudaStream_t` that outlives every guard
    /// acquired with it.
    pub const unsafe fn from_raw(raw: *mut c_void) -> Self {
        Self(raw)
    }

    /// Return the stored pointer, or null for the unresolved internal-stream sentinel.
    pub fn as_raw(self) -> *mut c_void {
        self.0
    }

    pub fn is_internal(self) -> bool {
        self.0.is_null()
    }

    fn resolve(self) -> Result<*mut c_void> {
        if self.is_internal() {
            internal_stream().map(CudaStream::as_raw)
        } else {
            Ok(self.as_raw())
        }
    }
}

/// Classification of a `cuda_buffer` C ABI failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidArgument,
    BadAlloc,
    Cuda,
    Other,
}

/// Error returned by the CUDA buffer bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaBufferError {
    pub kind: ErrorKind,
    pub message: String,
}

impl fmt::Display for CudaBufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for CudaBufferError {}

/// Result of a CUDA buffer operation.
pub type Result<T> = std::result::Result<T, CudaBufferError>;

fn check(ret: ffi::cuda_buffer_ret_t) -> Result<()> {
    let kind = match ret {
        ffi::CUDA_BUFFER_RET_OK => return Ok(()),
        ffi::CUDA_BUFFER_RET_INVALID_ARGUMENT => ErrorKind::InvalidArgument,
        ffi::CUDA_BUFFER_RET_BAD_ALLOC => ErrorKind::BadAlloc,
        ffi::CUDA_BUFFER_RET_CUDA_ERROR => ErrorKind::Cuda,
        _ => ErrorKind::Other,
    };
    Err(CudaBufferError {
        kind,
        message: last_error_message(),
    })
}

fn last_error_message() -> String {
    // SAFETY: the ABI returns a non-null, thread-local C string.
    unsafe { CStr::from_ptr(ffi::cuda_buffer_error_message()) }
        .to_string_lossy()
        .into_owned()
}

fn corrupt_abi(message: &str) -> CudaBufferError {
    CudaBufferError {
        kind: ErrorKind::Other,
        message: message.to_string(),
    }
}

/// Get the backend's process-wide internal CUDA stream.
pub fn internal_stream() -> Result<CudaStream> {
    let mut raw = ptr::null_mut();
    check(unsafe { ffi::cuda_buffer_internal_stream(&mut raw) })?;
    Ok(unsafe { CudaStream::from_raw(raw) })
}

/// Report whether an opaque `rosidl::Buffer<uint8_t> *` uses the CUDA backend.
///
/// # Safety
///
/// `buffer` must be null or point to a live `rosidl::Buffer<uint8_t>`.
pub unsafe fn is_cuda_backed(buffer: *const c_void) -> bool {
    unsafe { ffi::cuda_buffer_is_cuda_backed(buffer) }
}

/// Allocate uninitialized CUDA bytes for a generated message field.
///
/// Uses the native CUDA pool without acquiring a handle or copying data.
/// Initialize the output through `from_output_buffer` before publishing it.
pub fn allocate_buffer(byte_count: usize) -> Result<Buffer<u8>> {
    let mut raw = ptr::null_mut();
    check(unsafe { ffi::cuda_buffer_allocate(byte_count, &mut raw) })?;
    // SAFETY: successful allocation transfers a native byte buffer of this size.
    // The sequence takes sole ownership and releases it through rosidl_buffer.
    unsafe { PrimitiveSequence::from_owned_rosidl_buffer(raw, byte_count) }
        .map(Buffer::from)
        .ok_or_else(|| corrupt_abi("allocation returned a null buffer"))
}

/// Scoped native read access, released on drop.
///
/// Prefer [`read_buffer`], which ties the handle to a borrow of the buffer.
#[must_use]
pub struct ReadHandle<'a> {
    raw: NonNull<ffi::cuda_buffer_read_handle_t>,
    _owner: PhantomData<&'a ()>,
}

impl ReadHandle<'_> {
    /// Acquire read access to an opaque `rosidl::Buffer<uint8_t> *`.
    ///
    /// A non-CUDA buffer is promoted with a host-to-device copy; the promotion is
    /// retained by the handle and never becomes the caller's to release.
    ///
    /// # Safety
    ///
    /// `buffer` must point to a live `rosidl::Buffer<uint8_t>` that outlives the
    /// returned handle. Enqueue work on `stream` before dropping the handle;
    /// the stream must remain live until then.
    pub unsafe fn acquire(buffer: *const c_void, stream: CudaStream) -> Result<Self> {
        let mut raw = ptr::null_mut();
        let stream = stream.resolve()?;
        check(unsafe { ffi::cuda_buffer_acquire_read(buffer, stream, &mut raw) })?;
        NonNull::new(raw)
            .map(|raw| Self {
                raw,
                _owner: PhantomData,
            })
            .ok_or_else(|| corrupt_abi("read acquisition returned a null handle"))
    }

    /// Device pointer to the readable bytes.
    pub fn device_ptr(&self) -> *const u8 {
        unsafe { ffi::cuda_buffer_read_handle_data(self.raw.as_ptr()) }
    }

    /// Number of readable bytes.
    pub fn len(&self) -> usize {
        unsafe { ffi::cuda_buffer_read_handle_size(self.raw.as_ptr()) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for ReadHandle<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadHandle")
            .field("device_ptr", &self.device_ptr())
            .field("len", &self.len())
            .finish()
    }
}

impl Drop for ReadHandle<'_> {
    fn drop(&mut self) {
        unsafe { ffi::cuda_buffer_read_handle_destroy(self.raw.as_ptr()) };
    }
}

/// Native write access, released on drop.
///
/// Prefer [`write_buffer`], which ties the handle to a mutable borrow of the
/// buffer.
#[must_use]
pub struct WriteHandle {
    raw: NonNull<ffi::cuda_buffer_write_handle_t>,
}

impl WriteHandle {
    /// Acquire write access to the opaque `rosidl::Buffer<uint8_t> *` in `buffer`.
    ///
    /// When the buffer is not CUDA-backed it is promoted, and on success
    /// `*buffer` is replaced with a newly allocated CUDA-backed buffer that the
    /// caller now owns in addition to the pointer it passed in. The promoted
    /// contents are uninitialized. On failure `*buffer` is unchanged.
    ///
    /// # Safety
    ///
    /// `*buffer` must point to a live `rosidl::Buffer<uint8_t>` through all uses
    /// of the returned handle. Submit work before the buffer is published, read,
    /// or destroyed. The stream must remain live through handle cleanup.
    pub unsafe fn acquire(buffer: &mut *mut c_void, stream: CudaStream) -> Result<Self> {
        let mut raw = ptr::null_mut();
        let stream = stream.resolve()?;
        check(unsafe { ffi::cuda_buffer_acquire_write(buffer, stream, &mut raw) })?;
        NonNull::new(raw)
            .map(|raw| Self { raw })
            .ok_or_else(|| corrupt_abi("write acquisition returned a null handle"))
    }

    /// Device pointer to the writable bytes.
    pub fn device_ptr(&self) -> *mut u8 {
        unsafe { ffi::cuda_buffer_write_handle_data(self.raw.as_ptr()) }
    }

    /// Number of writable bytes.
    pub fn len(&self) -> usize {
        unsafe { ffi::cuda_buffer_write_handle_size(self.raw.as_ptr()) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for WriteHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WriteHandle")
            .field("device_ptr", &self.device_ptr())
            .field("len", &self.len())
            .finish()
    }
}

impl Drop for WriteHandle {
    fn drop(&mut self) {
        unsafe { ffi::cuda_buffer_write_handle_destroy(self.raw.as_ptr()) };
    }
}

/// Acquire raw CUDA read access from native-backed message storage.
///
/// The handle borrows `buffer` through cleanup. For ordinary CPU storage, use
/// the typed `from_input_buffer` API, which uploads it before returning.
pub fn read_buffer(buffer: &Buffer<u8>, stream: CudaStream) -> Result<CudaReadGuard<'_>> {
    read_primitive_sequence(buffer.as_sequence(), stream)
}

/// Acquire raw CUDA write access to a message buffer.
///
/// Non-CUDA storage is replaced with uninitialized CUDA storage of the same
/// byte length. Initialize it before publishing. Failure leaves the owner unchanged.
pub fn write_buffer(buffer: &mut Buffer<u8>, stream: CudaStream) -> Result<CudaWriteGuard<'_>> {
    let mut promoted = None;
    let mut raw = match buffer.as_sequence().rosidl_buffer_ptr() {
        Some(raw) if unsafe { is_cuda_backed(raw) } => raw,
        _ => {
            let allocation = allocate_buffer(buffer.len())?;
            let raw = allocation
                .as_sequence()
                .rosidl_buffer_ptr()
                .expect("CUDA allocation is native-backed");
            promoted = Some(allocation);
            raw
        }
    };
    // SAFETY: raw refers to CUDA storage owned by buffer or promoted. The
    // returned guard borrows buffer, which receives any promoted allocation.
    let handle = unsafe { WriteHandle::acquire(&mut raw, stream) }?;
    if let Some(allocation) = promoted {
        *buffer = allocation;
    }
    Ok(CudaWriteGuard {
        handle,
        _owner: PhantomData,
    })
}

/// Scoped read access tied to a borrow of its [`Buffer`].
pub type CudaReadGuard<'a> = ReadHandle<'a>;

/// Scoped write access tied to a mutable borrow of its [`Buffer`].
#[derive(Debug)]
pub struct CudaWriteGuard<'a> {
    handle: WriteHandle,
    _owner: PhantomData<&'a mut Buffer<u8>>,
}

impl CudaWriteGuard<'_> {
    pub fn device_ptr(&self) -> *mut u8 {
        self.handle.device_ptr()
    }

    pub fn len(&self) -> usize {
        self.handle.len()
    }

    pub fn is_empty(&self) -> bool {
        self.handle.is_empty()
    }
}

/// Acquires CUDA read access directly from an RMW-native `uint8[]` field.
pub fn read_primitive_sequence(
    sequence: &PrimitiveSequence<u8>,
    stream: CudaStream,
) -> Result<CudaReadGuard<'_>> {
    let raw = sequence
        .rosidl_buffer_ptr()
        .ok_or_else(|| CudaBufferError {
            kind: ErrorKind::InvalidArgument,
            message: "primitive sequence is not Buffer-backed".into(),
        })?;
    // SAFETY: the returned handle borrows sequence, which retains the owner.
    unsafe { ReadHandle::acquire(raw, stream) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_internal_stream_is_resolved_before_acquisition() {
        let internal = internal_stream().unwrap().as_raw();
        assert!(!internal.is_null());
        assert_eq!(CudaStream::INTERNAL.resolve().unwrap(), internal);
        assert_eq!(internal_stream().unwrap().resolve().unwrap(), internal);

        let mut buffer = allocate_buffer(1).unwrap();
        let write = write_buffer(&mut buffer, CudaStream::INTERNAL).unwrap();
        static SOURCE: [u8; 1] = [42];
        // SAFETY: SOURCE is static, the destination is live, and the backend's
        // internal stream outlives the copy and the write handle.
        check(unsafe {
            ffi::cuda_buffer_to_buffer_on_stream(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                write.handle.raw.as_ptr(),
                internal,
                ffi::CUDA_BUFFER_COPY_HOST_TO_DEVICE,
            )
        })
        .unwrap();
    }
}
