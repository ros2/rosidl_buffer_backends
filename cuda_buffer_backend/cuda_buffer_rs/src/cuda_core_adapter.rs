// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Typed cuda-core access to backend-owned VMM storage.

use std::borrow::{Borrow, Cow};
use std::fmt;
use std::marker::PhantomData;
use std::mem::{align_of, size_of, ManuallyDrop};
use std::sync::Arc;

use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
use rosidl_buffer_rs::{Buffer, PrimitiveSequence};

use crate::{
    allocate_buffer, ffi, native_buffer, native_call, CudaBufferError, ErrorKind, ReadHandle,
    Result, WriteHandle,
};

fn invalid(message: &str) -> CudaBufferError {
    CudaBufferError {
        kind: ErrorKind::InvalidArgument,
        message: message.into(),
    }
}

fn driver_error(error: impl std::fmt::Display) -> CudaBufferError {
    CudaBufferError {
        kind: ErrorKind::Cuda,
        message: error.to_string(),
    }
}

fn element_count<T>(bytes: usize) -> Result<usize> {
    let size = size_of::<T>();
    if size == 0 {
        return Err(invalid("zero-sized CUDA elements are not supported"));
    }
    if bytes == 0 || !bytes.is_multiple_of(size) {
        return Err(invalid(
            "buffer must contain a nonzero whole number of elements",
        ));
    }
    Ok(bytes / size)
}

fn prepare<T>(buffer: *const std::ffi::c_void, bytes: usize, stream: &CudaStream) -> Result<usize> {
    let len = element_count::<T>(bytes)?;
    let buffer = unsafe { native_buffer(buffer) }?;
    let device = native_call(|error| ffi::device_id(buffer, error))?;
    // Both cuda-core and the backend use the device's primary context.
    if device < 0 || device as usize != stream.context().ordinal() {
        return Err(invalid(
            "CUDA stream and buffer belong to different devices",
        ));
    }
    stream.context().bind_to_thread().map_err(driver_error)?;
    Ok(len)
}

struct Access<T, H> {
    facade: ManuallyDrop<DeviceBuffer<T>>,
    native: Option<H>,
    stream: Arc<CudaStream>,
    promoted: Option<Buffer<u8>>,
}

impl<T: DeviceCopy, H> Access<T, H> {
    fn new(native: H, address: usize, len: usize, stream: &Arc<CudaStream>) -> Result<Self> {
        if address == 0 || !address.is_multiple_of(align_of::<T>()) {
            return Err(invalid(
                "CUDA device address is null or misaligned for the element type",
            ));
        }
        // SAFETY: callers retain the backend owner during device access and
        // validate the element layout and context before constructing this view.
        // This relies on cuda-core 0.3.1's raw-parts implementation: construction
        // stores the address without touching the allocation. While the facade
        // stays in Access, ManuallyDrop and into_raw_parts suppress cuMemFree.
        // The mutable accessor cannot enforce that callers leave it in place.
        let facade = ManuallyDrop::new(unsafe {
            DeviceBuffer::from_raw_parts(address as u64, len, Arc::clone(stream.context()))
        });
        Ok(Self {
            facade,
            native: Some(native),
            stream: Arc::clone(stream),
            promoted: None,
        })
    }
}

impl<T, H> Access<T, H> {
    fn buffer(&self) -> &DeviceBuffer<T> {
        &self.facade
    }

    fn buffer_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.facade
    }
}

impl<T, H> Drop for Access<T, H> {
    fn drop(&mut self) {
        // Detach the facade before recording the native access event.
        // The retained stream/context remain live through native cleanup.
        self.stream
            .context()
            .record_err(self.stream.context().bind_to_thread());
        // SAFETY: drop takes the facade once. into_raw_parts returns its context
        // reference without freeing the backend-owned allocation.
        let facade = unsafe { ManuallyDrop::take(&mut self.facade) };
        let (_, _, context) = facade.into_raw_parts();
        drop(context);
        drop(self.native.take());
        drop(self.promoted.take());
    }
}

/// Typed CUDA read access borrowing its backend owner.
///
/// Enable the `cuda-core` feature. CUDA input retains its device storage; CPU
/// input is promoted. The native handle orders access on [`Self::stream`].
#[must_use]
pub struct CudaReadHandle<'a, T: DeviceCopy> {
    access: Access<T, ReadHandle<'a>>,
    _owner: PhantomData<&'a Buffer<u8>>,
}

/// Typed exclusive CUDA write access borrowing its backend owner.
///
/// The backend permits one write phase, followed by read phases. Read access,
/// serialization, or owner destruction finalizes a floating write. Handle drop
/// finalizes any remaining write; acquire a new buffer for subsequent writes.
#[must_use]
pub struct CudaWriteHandle<'a, T: DeviceCopy> {
    access: Access<T, WriteHandle>,
    _owner: PhantomData<&'a mut Buffer<u8>>,
}

// SAFETY: buffer must remain live and exclusively borrowed for the returned lifetime.
unsafe fn acquire_write<'a, T: DeviceCopy>(
    buffer: *mut std::ffi::c_void,
    bytes: usize,
    stream: &Arc<CudaStream>,
) -> Result<CudaWriteHandle<'a, T>> {
    let len = prepare::<T>(buffer, bytes, stream)?;
    let mut slot = buffer;
    let native = unsafe { WriteHandle::acquire_exact(&mut slot, stream.cu_stream().cast()) }?;
    let address = native.device_ptr() as usize;
    Ok(CudaWriteHandle {
        access: Access::new(native, address, len, stream)?,
        _owner: PhantomData,
    })
}

// SAFETY: buffer must remain live and immutable for the returned lifetime.
unsafe fn acquire_read<'a, T: DeviceCopy>(
    buffer: *const std::ffi::c_void,
    bytes: usize,
    stream: &Arc<CudaStream>,
) -> Result<CudaReadHandle<'a, T>> {
    let len = prepare::<T>(buffer, bytes, stream)?;
    let native = unsafe { ReadHandle::acquire_exact(buffer, stream.cu_stream().cast()) }?;
    let address = native.device_ptr() as usize;
    Ok(CudaReadHandle {
        access: Access::new(native, address, len, stream)?,
        _owner: PhantomData,
    })
}

/// Borrow typed CUDA data from an RMW-native message without extracting its owner.
/// Accepts runtime sequences and backend primitive sequences through a shared borrow.
pub fn get_primitive_sequence_read_handle<'a, T: DeviceCopy>(
    sequence: &'a (impl Borrow<PrimitiveSequence<u8>> + ?Sized),
    stream: &Arc<CudaStream>,
) -> Result<CudaReadHandle<'a, T>> {
    let sequence = sequence.borrow();
    let raw = sequence
        .rosidl_buffer_ptr()
        .ok_or_else(|| invalid("primitive sequence is not Buffer-backed"))?;
    // SAFETY: sequence retains the owner for the returned lifetime.
    unsafe { acquire_read(raw, sequence.len(), stream) }
}

macro_rules! common_accessors {
    () => {
        /// Number of typed elements (not bytes).
        pub fn len(&self) -> usize {
            self.access.buffer().len()
        }
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }
        /// Number of bytes covered by this handle.
        pub fn byte_len(&self) -> usize {
            self.access.buffer().num_bytes()
        }
        /// Stream retained until after the native access event is recorded.
        pub fn stream(&self) -> &Arc<CudaStream> {
            &self.access.stream
        }
        /// Borrowed device pointer. Never free it or use it beyond this handle.
        pub fn device_ptr(&self) -> *const T {
            self.access.buffer().cu_deviceptr() as usize as *const T
        }
        /// Copy to host, waiting on this handle's acquisition stream.
        pub fn to_host_vec(&self) -> Result<Vec<T>> {
            self.stream()
                .context()
                .bind_to_thread()
                .map_err(driver_error)?;
            self.access
                .buffer()
                .to_host_vec(self.stream())
                .map_err(driver_error)
        }
    };
}

macro_rules! debug_handle {
    ($handle:ident) => {
        impl<T: DeviceCopy> fmt::Debug for $handle<'_, T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($handle))
                    .field("device_ptr", &self.device_ptr())
                    .field("len", &self.len())
                    .field("stream", &self.stream().cu_stream())
                    .finish()
            }
        }
    };
}

debug_handle!(CudaReadHandle);
debug_handle!(CudaWriteHandle);

impl<T: DeviceCopy> CudaReadHandle<'_, T> {
    common_accessors!();

    /// Borrow a cuda-core buffer view without copying.
    ///
    /// # Caller requirements
    /// Submit all work on this handle's stream while its owner is borrowed.
    /// Complete submission before publishing the buffer or releasing the handle.
    /// Keep the pointer, length, context, and allocation owner unchanged.
    /// Do not modify the GPU contents, including through kernels or extracted
    /// raw pointers.
    /// These requirements are not enforced by the returned reference.
    pub fn as_device_buffer(&self) -> &DeviceBuffer<T> {
        self.access.buffer()
    }

    /// Read-only device pointer borrowed for this handle's lifetime.
    pub fn get_ptr(&self) -> *const T {
        self.device_ptr()
    }
}

impl<T: DeviceCopy> CudaWriteHandle<'_, T> {
    common_accessors!();

    /// Writable device pointer borrowed for this handle's lifetime.
    pub fn get_ptr(&mut self) -> *mut T {
        self.device_ptr_mut()
    }

    /// Copy initialized host values to the acquisition stream and wait for completion.
    pub fn copy_from_host(&mut self, values: &[T]) -> Result<()> {
        if values.len() != self.len() {
            return Err(invalid("host slice length must match the buffer"));
        }
        let stream = Arc::clone(self.stream());
        stream.context().bind_to_thread().map_err(driver_error)?;
        self.access
            .buffer_mut()
            .copy_from_host(&stream, values)
            .map_err(driver_error)
    }

    /// Mutable raw device pointer, valid only under this handle's stream contract.
    pub fn device_ptr_mut(&mut self) -> *mut T {
        self.device_ptr() as *mut T
    }

    /// Borrow a mutable cuda-core buffer view without copying.
    ///
    /// # Caller requirements
    /// Submit all work on this handle's stream while its owner is borrowed.
    /// Complete submission before publishing the buffer or releasing the handle.
    /// Modify the device contents, not the `DeviceBuffer` object:
    /// - Do not assign through the reference (`*view = other`), use
    ///   `std::mem::replace` / `std::mem::swap`, or otherwise extract the object.
    /// - Do not resize, reallocate, or free its storage, or change its pointer,
    ///   length, or context. Do not use extracted pointers after handle release.
    ///
    /// Initialize the entire output before publishing it.
    ///
    /// Violations can cause use-after-free, data races, or an invalid `cuMemFree`
    /// of backend-owned VMM storage, even when the caller uses only safe Rust.
    pub fn as_device_buffer(&mut self) -> &mut DeviceBuffer<T> {
        self.access.buffer_mut()
    }
}

/// Source memory for a copy into a CUDA output handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum CopyKind {
    /// Copy from host memory to the output's CUDA allocation.
    HostToDevice = 1,
    /// Copy from CUDA memory to the output's CUDA allocation.
    DeviceToDevice = 3,
}

/// Enqueue a byte copy into an existing output handle without allocating.
///
/// `stream` must be the handle's acquisition stream. `byte_count` may not exceed
/// the output's byte length. A zero-byte copy is a no-op. This function does not
/// synchronize. Read acquisition, publication, owner destruction, or handle
/// cleanup records the producer event after the queued work.
///
/// # Caller requirements
/// `source` must identify at least `byte_count` readable bytes of the selected
/// memory kind, accessible from this CUDA context and not overlapping the
/// destination. Keep the source allocation alive and unchanged until the copy
/// completes, and order any producer of the source before this copy on `stream`.
/// The source's validity and lifetime cannot be checked by this function.
#[allow(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "Source validity and asynchronous lifetime are documented caller requirements"
)]
pub fn to_buffer<T: DeviceCopy>(
    source: *const std::ffi::c_void,
    byte_count: usize,
    output: &mut CudaWriteHandle<'_, T>,
    stream: &Arc<CudaStream>,
    kind: CopyKind,
) -> Result<()> {
    if byte_count == 0 {
        return Ok(());
    }
    if source.is_null() || byte_count > output.byte_len() {
        return Err(invalid(
            "copy source is null or byte count exceeds the output",
        ));
    }
    if stream.cu_stream() != output.stream().cu_stream()
        || stream.context().ordinal() != output.stream().context().ordinal()
    {
        return Err(invalid("copy stream must match the output handle's stream"));
    }
    stream.context().bind_to_thread().map_err(driver_error)?;
    native_call(|error| unsafe {
        ffi::copy_to_buffer(
            output
                .access
                .native
                .as_mut()
                .expect("live write handle")
                .raw
                .pin_mut(),
            source.cast(),
            byte_count,
            stream.cu_stream() as usize,
            kind as i32,
            error,
        )
    })
}

/// Acquire CUDA read access to a message field on the consumer's stream.
///
/// CUDA storage is borrowed without copying. CPU input is copied to a temporary
/// CUDA allocation retained by the handle; the host transfer completes before
/// returning. The source field remains unchanged.
pub fn from_input_buffer<'a, T: DeviceCopy>(
    buffer: &'a Buffer<u8>,
    stream: &Arc<CudaStream>,
) -> Result<CudaReadHandle<'a, T>> {
    element_count::<T>(buffer.len())?;
    if let Some(raw) = buffer.as_sequence().rosidl_buffer_ptr() {
        // SAFETY: buffer retains the native owner through the returned handle.
        if unsafe { crate::is_cuda_backed(raw) } {
            return unsafe { acquire_read(raw, buffer.len(), stream) };
        }
    }
    let host = match buffer.as_slice() {
        Some(values) => Cow::Borrowed(values),
        None => Cow::Owned(buffer.to_vec().map_err(|error| CudaBufferError {
            kind: ErrorKind::Other,
            message: error.to_string(),
        })?),
    };
    stream.context().bind_to_thread().map_err(driver_error)?;
    let mut promoted = allocate_buffer(buffer.len())?;
    from_output_buffer::<u8>(&mut promoted, stream)?.copy_from_host(&host)?;
    let raw = promoted
        .as_sequence()
        .rosidl_buffer_ptr()
        .expect("CUDA allocation is native-backed");
    // SAFETY: the returned handle retains promoted until after native cleanup.
    let mut read = unsafe { acquire_read(raw, promoted.len(), stream) }?;
    read.access.promoted = Some(promoted);
    Ok(read)
}

/// Acquire exclusive CUDA write access to an output message field.
///
/// CUDA storage is reused. Other storage is replaced with a CUDA allocation of
/// the same byte length; previous contents are not copied. Initialize the whole
/// output on this stream before publishing the message. Publication finalizes
/// the write; the unused handle can drop automatically at function exit.
/// Empty buffers and invalid typed lengths are rejected before replacement.
pub fn from_output_buffer<'a, T: DeviceCopy>(
    buffer: &'a mut Buffer<u8>,
    stream: &Arc<CudaStream>,
) -> Result<CudaWriteHandle<'a, T>> {
    element_count::<T>(buffer.len())?;
    if let Some(raw) = buffer.as_sequence().rosidl_buffer_ptr() {
        // SAFETY: buffer retains the owner and is exclusively borrowed.
        if unsafe { crate::is_cuda_backed(raw) } {
            return unsafe { acquire_write(raw, buffer.len(), stream) };
        }
    }
    stream.context().bind_to_thread().map_err(driver_error)?;
    let promoted = allocate_buffer(buffer.len())?;
    let raw = promoted
        .as_sequence()
        .rosidl_buffer_ptr()
        .expect("CUDA allocation is native-backed");
    // SAFETY: ownership is transferred into the exclusively borrowed field.
    let write = unsafe { acquire_write(raw, promoted.len(), stream) }?;
    *buffer = promoted;
    Ok(write)
}
