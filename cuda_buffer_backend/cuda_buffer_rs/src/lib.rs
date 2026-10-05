// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Scoped Rust access to native CUDA buffers.
//!
//! [`rosidl_buffer_rs::Buffer`] owns message storage; scoped handles order GPU
//! access through native CUDA events. The `cuda-core` feature adds typed access
//! and retains the acquisition stream until handle cleanup. CUDA access handles
//! are neither `Send` nor `Sync`.

#![warn(unsafe_op_in_unsafe_fn)]

#[cfg_attr(not(feature = "cuda-core"), allow(dead_code))]
mod bridge;
use bridge::ffi;
use rosidl_buffer_rs::CxxBuffer;

#[cfg(feature = "cuda-core")]
mod cuda_core_adapter;
#[cfg(feature = "cuda-core")]
pub use cuda_core_adapter::{
    from_input_buffer, from_output_buffer, get_primitive_sequence_read_handle, to_buffer, CopyKind,
    CudaReadHandle, CudaWriteHandle,
};

use std::fmt;
use std::marker::PhantomData;
use std::os::raw::c_void;
use std::pin::Pin;
use std::ptr;

use rosidl_buffer_rs::{Buffer, PrimitiveSequence};

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
    /// `raw` must be null or a live `cudaStream_t` that outlives every handle
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

/// Classification of a native CUDA buffer failure.
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

fn native_call<T>(
    call: impl FnOnce(&mut ffi::NativeErrorKind) -> std::result::Result<T, cxx::Exception>,
) -> Result<T> {
    let mut kind = ffi::NativeErrorKind::Other;
    call(&mut kind).map_err(|error| CudaBufferError {
        kind: match kind {
            ffi::NativeErrorKind::InvalidArgument => ErrorKind::InvalidArgument,
            ffi::NativeErrorKind::BadAlloc => ErrorKind::BadAlloc,
            ffi::NativeErrorKind::Cuda => ErrorKind::Cuda,
            _ => ErrorKind::Other,
        },
        message: error.to_string(),
    })
}

// SAFETY: raw must refer to a live native buffer for the returned lifetime.
unsafe fn native_buffer<'a>(raw: *const c_void) -> Result<&'a CxxBuffer> {
    unsafe { raw.cast::<CxxBuffer>().as_ref() }.ok_or_else(|| CudaBufferError {
        kind: ErrorKind::InvalidArgument,
        message: "native buffer must not be null".into(),
    })
}

/// Get the backend's process-wide internal CUDA stream.
pub fn internal_stream() -> Result<CudaStream> {
    let raw = native_call(ffi::internal_stream)?;
    Ok(unsafe { CudaStream::from_raw(raw as *mut c_void) })
}

/// Report whether an opaque `rosidl::Buffer<uint8_t> *` uses the CUDA backend.
///
/// # Safety
///
/// `buffer` must be null or point to a live `rosidl::Buffer<uint8_t>`.
pub unsafe fn is_cuda_backed(buffer: *const c_void) -> bool {
    unsafe { native_buffer(buffer) }.is_ok_and(ffi::is_cuda_backed)
}

/// Allocate uninitialized CUDA bytes for a generated message field.
///
/// Uses the native CUDA pool without acquiring a handle or copying data.
/// Initialize the output through `from_output_buffer` before publishing it.
pub fn allocate_buffer(byte_count: usize) -> Result<Buffer<u8>> {
    let buffer = native_call(|error| ffi::allocate(byte_count, error))?;
    rosidl_buffer_rs::native::into_buffer(buffer).map_err(|error| CudaBufferError {
        kind: ErrorKind::Other,
        message: error.to_string(),
    })
}

/// Scoped native read access, released on drop.
///
/// Prefer [`read_buffer`], which ties the handle to a borrow of the buffer.
///
/// The owner must remain alive through native handle cleanup:
/// ```compile_fail,E0505
/// use cuda_buffer_rs::{allocate_buffer, read_buffer, CudaStream};
/// let buffer = allocate_buffer(4).unwrap();
/// let _read = read_buffer(&buffer, CudaStream::INTERNAL).unwrap();
/// drop(buffer);
/// ```
#[must_use]
pub struct ReadHandle<'a> {
    raw: cxx::UniquePtr<ffi::CxxReadHandle>,
    _owner: PhantomData<&'a ()>,
}

impl Drop for ReadHandle<'_> {
    fn drop(&mut self) {
        // Keep the owner borrowed through the native destructor's event cleanup.
    }
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
        unsafe { Self::acquire_exact(buffer, stream.resolve()?) }
    }

    unsafe fn acquire_exact(buffer: *const c_void, stream: *mut c_void) -> Result<Self> {
        let buffer = unsafe { native_buffer(buffer) }?;
        let raw =
            native_call(|error| unsafe { ffi::acquire_read(buffer, stream as usize, error) })?;
        Ok(Self {
            raw,
            _owner: PhantomData,
        })
    }

    /// Device pointer to the readable bytes.
    pub fn device_ptr(&self) -> *const u8 {
        self.raw.data()
    }

    /// Number of readable bytes.
    pub fn len(&self) -> usize {
        self.raw.size()
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

/// Native write access, released on drop.
///
/// Prefer [`write_buffer`], which ties the handle to a mutable borrow of the
/// buffer.
#[must_use]
pub struct WriteHandle {
    raw: cxx::UniquePtr<ffi::CxxWriteHandle>,
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
        unsafe { Self::acquire_exact(buffer, stream.resolve()?) }
    }

    unsafe fn acquire_exact(buffer: &mut *mut c_void, stream: *mut c_void) -> Result<Self> {
        let source = unsafe { native_buffer(*buffer) }?;
        let promoted = if ffi::is_cuda_backed(source) {
            cxx::UniquePtr::null()
        } else {
            native_call(|error| ffi::allocate(source.size(), error))?
        };
        let target = if promoted.is_null() {
            *buffer as *mut CxxBuffer
        } else {
            promoted.as_mut_ptr()
        };
        // SAFETY: the caller exclusively borrows the original buffer. A
        // promoted buffer is owned here until successful acquisition.
        let target = unsafe { Pin::new_unchecked(&mut *target) };
        let raw =
            native_call(|error| unsafe { ffi::acquire_write(target, stream as usize, error) })?;
        if !promoted.is_null() {
            *buffer = promoted.into_raw().cast();
        }
        Ok(Self { raw })
    }

    /// Device pointer to the writable bytes.
    pub fn device_ptr(&self) -> *mut u8 {
        self.raw.data()
    }

    /// Number of writable bytes.
    pub fn len(&self) -> usize {
        self.raw.size()
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
    // returned handle borrows buffer, which receives any promoted allocation.
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
