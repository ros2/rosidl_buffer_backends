// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cuda_buffer_rs")]
pub(crate) mod ffi {
    #[repr(u8)]
    enum NativeErrorKind {
        InvalidArgument,
        BadAlloc,
        Cuda,
        Other,
    }

    // SAFETY: owning results use UniquePtr; CUDA borrowing and asynchronous
    // access remain unsafe and are scoped by the public Rust handles.
    unsafe extern "C++" {
        include!("cuda_buffer_rs/src/bridge.hpp");

        #[namespace = "rosidl_buffer_rs"]
        type CxxBuffer = rosidl_buffer_rs::CxxBuffer;
        type CxxReadHandle;
        type CxxWriteHandle;

        fn internal_stream(error: &mut NativeErrorKind) -> Result<usize>;
        fn allocate(size: usize, error: &mut NativeErrorKind) -> Result<UniquePtr<CxxBuffer>>;
        fn is_cuda_backed(buffer: &CxxBuffer) -> bool;
        fn device_id(buffer: &CxxBuffer, error: &mut NativeErrorKind) -> Result<i32>;

        // Callers retain the buffer and stream until handle destruction.
        unsafe fn acquire_read(
            buffer: &CxxBuffer,
            stream: usize,
            error: &mut NativeErrorKind,
        ) -> Result<UniquePtr<CxxReadHandle>>;
        unsafe fn acquire_write(
            buffer: Pin<&mut CxxBuffer>,
            stream: usize,
            error: &mut NativeErrorKind,
        ) -> Result<UniquePtr<CxxWriteHandle>>;
        fn data(self: &CxxReadHandle) -> *const u8;
        fn size(self: &CxxReadHandle) -> usize;
        fn data(self: &CxxWriteHandle) -> *mut u8;
        fn size(self: &CxxWriteHandle) -> usize;
        unsafe fn copy_to_buffer(
            handle: Pin<&mut CxxWriteHandle>,
            source: *const u8,
            size: usize,
            stream: usize,
            kind: i32,
            error: &mut NativeErrorKind,
        ) -> Result<()>;
    }
}
