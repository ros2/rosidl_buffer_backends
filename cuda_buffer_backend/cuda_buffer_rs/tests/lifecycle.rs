use cuda_buffer_rs::{CudaBuffer, CudaStream, ErrorKind};

#[test]
fn internal_stream_is_available() {
    let stream = cuda_buffer_rs::internal_stream().expect("internal stream");
    assert!(!stream.as_raw().is_null());
    assert!(!stream.is_internal());
    assert!(CudaStream::INTERNAL.is_internal());
}

#[test]
fn allocated_buffer_is_cuda_backed() {
    let buffer = CudaBuffer::allocate(1024).expect("allocate");
    assert_eq!(buffer.len(), 1024);
    assert!(!buffer.is_empty());
    assert!(buffer.is_cuda_backed());
    assert!(!buffer.as_ptr().is_null());
}

#[test]
fn write_then_read_guards_expose_the_same_device_memory() {
    let stream = cuda_buffer_rs::internal_stream().expect("internal stream");
    let mut buffer = CudaBuffer::allocate(2048).expect("allocate");

    let written = {
        let guard = buffer.write(stream).expect("write guard");
        assert_eq!(guard.len(), 2048);
        assert!(!guard.device_ptr().is_null());
        guard.device_ptr()
    };

    let guard = buffer.read(stream).expect("read guard");
    assert_eq!(guard.len(), 2048);
    assert_eq!(guard.device_ptr(), written as *const u8);
}

#[test]
fn internal_stream_constant_is_accepted() {
    let mut buffer = CudaBuffer::allocate(64).expect("allocate");
    assert!(!buffer
        .write(CudaStream::INTERNAL)
        .expect("write guard")
        .device_ptr()
        .is_null());
    assert!(!buffer
        .read(CudaStream::INTERNAL)
        .expect("read guard")
        .device_ptr()
        .is_null());
}

#[test]
fn empty_buffer_rejects_guards() {
    let mut buffer = CudaBuffer::allocate(0).expect("allocate");
    assert!(buffer.is_empty());
    assert_eq!(
        buffer.read(CudaStream::INTERNAL).unwrap_err().kind,
        ErrorKind::InvalidArgument
    );
    assert_eq!(
        buffer.write(CudaStream::INTERNAL).unwrap_err().kind,
        ErrorKind::InvalidArgument
    );
}

#[test]
fn into_raw_and_from_raw_round_trip_ownership() {
    let buffer = CudaBuffer::allocate(256).expect("allocate");
    let raw = buffer.into_raw();
    assert!(!raw.is_null());

    let reclaimed = unsafe { CudaBuffer::from_raw(raw, 256) }.expect("reclaim");
    assert_eq!(reclaimed.len(), 256);
    assert!(reclaimed.is_cuda_backed());
}

#[test]
fn from_raw_rejects_null() {
    assert!(unsafe { CudaBuffer::from_raw(std::ptr::null_mut(), 0) }.is_none());
}

#[test]
fn primitive_sequence_transfers_buffer_ownership() {
    let buffer = CudaBuffer::allocate(512).expect("allocate");
    let sequence = buffer.into_primitive_sequence();
    assert!(sequence.is_rosidl_buffer());
    assert_eq!(sequence.len(), 512);

    let guard =
        cuda_buffer_rs::read_primitive_sequence(&sequence, CudaStream::INTERNAL).expect("read");
    assert_eq!(guard.len(), 512);
    assert!(!guard.device_ptr().is_null());
    drop(guard);

    let reclaimed = CudaBuffer::from_primitive_sequence(sequence).expect("reclaim");
    assert_eq!(reclaimed.len(), 512);
    assert!(reclaimed.is_cuda_backed());
}

#[test]
fn repeated_allocate_and_guard_cycles_are_stable() {
    for _ in 0..64 {
        let mut buffer = CudaBuffer::allocate(4096).expect("allocate");
        assert!(!buffer
            .write(CudaStream::INTERNAL)
            .expect("write")
            .device_ptr()
            .is_null());
        assert!(!buffer
            .read(CudaStream::INTERNAL)
            .expect("read")
            .device_ptr()
            .is_null());
    }
}
