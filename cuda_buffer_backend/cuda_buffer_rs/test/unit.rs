// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

use cuda_buffer_rs::{CudaBuffer, CudaStream, ErrorKind};

#[test]
fn raw_guards_share_device_memory_and_finalize_one_write() {
    let internal = cuda_buffer_rs::internal_stream().unwrap();
    assert!(!internal.as_raw().is_null());
    assert!(!internal.is_internal());
    assert!(CudaStream::INTERNAL.is_internal());
    for stream in [internal, CudaStream::INTERNAL] {
        let mut buffer = CudaBuffer::allocate(2048).unwrap();
        assert_eq!(buffer.len(), 2048);
        assert!(!buffer.is_empty());
        assert!(buffer.is_cuda_backed());
        let written = {
            let guard = buffer.write(stream).unwrap();
            assert_eq!(guard.len(), 2048);
            assert!(!guard.device_ptr().is_null());
            guard.device_ptr()
        };
        let error = buffer.write(stream).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Cuda);
        assert!(error.message.contains("finalized"));
        let guard = buffer.read(stream).unwrap();
        assert_eq!(guard.len(), 2048);
        assert_eq!(guard.device_ptr(), written.cast_const());
    }
}

#[test]
fn raw_and_sequence_conversions_transfer_ownership() {
    assert!(unsafe { CudaBuffer::from_raw(std::ptr::null_mut(), 0) }.is_none());
    let raw = CudaBuffer::allocate(512).unwrap().into_raw();
    assert!(!raw.is_null());
    // SAFETY: into_raw transferred the sole native owner with this byte length.
    let buffer = unsafe { CudaBuffer::from_raw(raw, 512) }.unwrap();
    assert_eq!(buffer.len(), 512);
    assert!(buffer.is_cuda_backed());
    let sequence = buffer.into_primitive_sequence();
    assert_eq!(sequence.rosidl_buffer_ptr(), Some(raw));
    assert_eq!(sequence.len(), 512);
    let guard = cuda_buffer_rs::read_primitive_sequence(&sequence, CudaStream::INTERNAL).unwrap();
    assert_eq!(guard.len(), 512);
    assert!(!guard.device_ptr().is_null());
    drop(guard);
    let reclaimed = CudaBuffer::from_primitive_sequence(sequence).unwrap();
    assert_eq!(reclaimed.len(), 512);
    assert!(reclaimed.is_cuda_backed());
    assert_eq!(
        reclaimed.into_buffer().as_sequence().rosidl_buffer_ptr(),
        Some(raw)
    );
}

#[test]
fn empty_buffers_reject_guards_and_errors_survive_later_ffi_calls() {
    let mut buffer = CudaBuffer::allocate(0).unwrap();
    assert!(buffer.is_empty());
    let error = buffer.read(CudaStream::INTERNAL).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidArgument);
    let message = error.message.clone();
    assert!(!message.is_empty());
    assert!(std::error::Error::source(&error).is_none());
    assert_eq!(
        buffer.write(CudaStream::INTERNAL).unwrap_err().kind,
        ErrorKind::InvalidArgument
    );
    let _stream = cuda_buffer_rs::internal_stream().unwrap();
    assert_eq!(error.message, message);
    assert!(error.to_string().contains(&message));
    assert!(!unsafe { cuda_buffer_rs::is_cuda_backed(std::ptr::null()) });
}

#[test]
fn normal_sequences_are_returned_on_rejected_adoption() {
    let sequence = rosidl_runtime_rs::PrimitiveSequence::from(&[1u8, 2, 3][..]);
    let error =
        cuda_buffer_rs::read_primitive_sequence(&sequence, CudaStream::INTERNAL).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidArgument);
    let sequence = CudaBuffer::from_primitive_sequence(sequence).unwrap_err();
    assert_eq!(sequence.as_slice(), &[1, 2, 3]);
}

#[cfg(feature = "cuda-core")]
mod typed {
    use super::*;
    use cuda_buffer_rs::{
        allocate_buffer, from_input_buffer, from_output_buffer, to_buffer, CopyKind,
    };
    use cuda_core::{CudaContext, DeviceBuffer};
    use std::ffi::c_void;
    use std::mem::size_of_val;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    use std::time::{Duration, Instant};

    #[test]
    fn invalid_layout_is_rejected_before_write_acquisition() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.default_stream();
        let mut buffer = CudaBuffer::allocate(7).unwrap();
        assert_eq!(
            buffer.get_write_handle::<u32>(&stream).err().unwrap().kind,
            ErrorKind::InvalidArgument
        );
        assert_eq!(
            buffer.get_write_handle::<()>(&stream).err().unwrap().kind,
            ErrorKind::InvalidArgument
        );
        let mut writer = buffer.get_write_handle::<u8>(&stream).unwrap();
        assert_eq!(
            writer.copy_from_host(&[1, 2]).unwrap_err().kind,
            ErrorKind::InvalidArgument
        );
        writer.copy_from_host(&[3; 7]).unwrap();
        drop(writer);
        assert!(buffer.get_read_handle::<u32>(&stream).is_err());
        assert!(buffer.get_read_handle::<()>(&stream).is_err());
        assert_eq!(
            buffer
                .get_read_handle::<u8>(&stream)
                .unwrap()
                .to_host_vec()
                .unwrap(),
            vec![3; 7]
        );
        assert!(CudaBuffer::allocate(0)
            .unwrap()
            .get_read_handle::<u8>(&stream)
            .is_err());
    }

    #[test]
    fn stream_and_context_references_are_released_on_drop_and_unwind() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let context_refs = Arc::strong_count(&context);
        let stream_refs = Arc::strong_count(&stream);
        let mut buffer = CudaBuffer::allocate(128).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut writer = buffer.get_write_handle::<u32>(&stream).unwrap();
            writer.copy_from_host(&[91; 32]).unwrap();
            panic!("exercise unwinding after native acquisition");
        }));
        assert!(result.is_err());
        assert_eq!(Arc::strong_count(&context), context_refs);
        assert_eq!(Arc::strong_count(&stream), stream_refs);
        for _ in 0..2 {
            let handle = buffer.get_read_handle::<u32>(&stream).unwrap();
            assert_eq!(handle.to_host_vec().unwrap(), vec![91; 32]);
        }
        assert_eq!(Arc::strong_count(&context), context_refs);
        assert_eq!(Arc::strong_count(&stream), stream_refs);
        context.check_err().unwrap();
    }

    #[test]
    fn handle_retains_stream_when_caller_drops_its_arc() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let weak = Arc::downgrade(&stream);
        let mut buffer = CudaBuffer::allocate(128).unwrap();
        let mut writer = buffer.get_write_handle::<u32>(&stream).unwrap();
        drop(stream);
        assert!(weak.upgrade().is_some());
        writer.copy_from_host(&[12; 32]).unwrap();
        drop(writer);
        assert!(weak.upgrade().is_none());
        let stream = context.default_stream();
        assert_eq!(
            buffer
                .get_read_handle::<u32>(&stream)
                .unwrap()
                .to_host_vec()
                .unwrap(),
            vec![12; 32]
        );
    }

    #[test]
    fn host_copy_uses_bytes_on_default_and_nonblocking_streams() {
        let context = CudaContext::new(0).unwrap();
        for stream in [context.default_stream(), context.new_stream().unwrap()] {
            let values = [3u32, 7, 11, 19];
            let mut data = allocate_buffer(size_of_val(&values)).unwrap();
            {
                let mut output = from_output_buffer::<u32>(&mut data, &stream).unwrap();
                // SAFETY: values stays live and unchanged through stream synchronization.
                unsafe {
                    to_buffer(
                        values.as_ptr().cast(),
                        size_of_val(&values),
                        &mut output,
                        &stream,
                        CopyKind::HostToDevice,
                    )
                }
                .unwrap();
                stream.synchronize().unwrap();
            }
            let input = from_input_buffer::<u32>(&data, &stream).unwrap();
            assert_eq!(input.to_host_vec().unwrap(), values);
        }
    }

    #[test]
    fn device_copy_is_async_and_preserves_the_output_allocation() {
        let context = CudaContext::new(0).unwrap();
        let producer = context.new_stream().unwrap();
        let consumer = context.default_stream();
        let values = [23u32; 32];
        let source = DeviceBuffer::from_host(&producer, &values).unwrap();
        let mut data = allocate_buffer(size_of_val(&values)).unwrap();
        let owner = data.as_sequence().rosidl_buffer_ptr();
        let mut output = from_output_buffer::<u32>(&mut data, &producer).unwrap();
        let gate = Arc::new(AtomicBool::new(false));
        let callback_gate = Arc::clone(&gate);
        producer
            .launch_host_function(move || {
                let deadline = Instant::now() + Duration::from_secs(2);
                while !callback_gate.load(Ordering::Acquire) && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(1));
                }
                callback_gate.store(true, Ordering::Release);
            })
            .unwrap();
        // SAFETY: source and its completed contents remain live through the consumer read.
        let copied = unsafe {
            to_buffer(
                source.cu_deviceptr() as usize as *const c_void,
                size_of_val(&values),
                &mut output,
                &producer,
                CopyKind::DeviceToDevice,
            )
        };
        let returned_before_gate_opened = !gate.load(Ordering::Acquire);
        copied.unwrap();
        let input = from_input_buffer::<u32>(&data, &consumer).unwrap();
        gate.store(true, Ordering::Release);
        assert_eq!(input.to_host_vec().unwrap(), values);
        assert!(
            returned_before_gate_opened,
            "copy synchronized the producer stream"
        );
        assert_eq!(data.as_sequence().rosidl_buffer_ptr(), owner);
        context.check_err().unwrap();
    }

    #[test]
    fn invalid_copy_requests_leave_the_output_usable() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let other = context.new_stream().unwrap();
        let mut data = allocate_buffer(4).unwrap();
        let values = [29u8; 4];
        {
            let mut output = from_output_buffer::<u8>(&mut data, &stream).unwrap();
            for (source, size, selected) in [
                (std::ptr::null(), 4, &stream),
                (values.as_ptr().cast(), 5, &stream),
                (values.as_ptr().cast(), 4, &other),
            ] {
                // SAFETY: invalid arguments must be rejected before submitting a copy.
                let error = unsafe {
                    to_buffer(source, size, &mut output, selected, CopyKind::HostToDevice)
                }
                .unwrap_err();
                assert_eq!(error.kind, ErrorKind::InvalidArgument);
            }
            // SAFETY: zero bytes access no source memory.
            unsafe {
                to_buffer(
                    std::ptr::null(),
                    0,
                    &mut output,
                    &other,
                    CopyKind::HostToDevice,
                )
            }
            .unwrap();
            output.copy_from_host(&values).unwrap();
        }
        assert_eq!(
            from_input_buffer::<u8>(&data, &stream)
                .unwrap()
                .to_host_vec()
                .unwrap(),
            values
        );
    }

    #[test]
    fn unpublished_owner_can_drop_before_unused_write_handle() {
        static VALUES: [u8; 4] = [3, 5, 7, 11];
        let context = CudaContext::new(0).unwrap();
        for stream in [context.default_stream(), context.new_stream().unwrap()] {
            let mut data = allocate_buffer(VALUES.len()).unwrap();
            let mut output = from_output_buffer::<u8>(&mut data, &stream).unwrap();
            // SAFETY: VALUES remains valid through the queued copy.
            unsafe {
                to_buffer(
                    VALUES.as_ptr().cast(),
                    VALUES.len(),
                    &mut output,
                    &stream,
                    CopyKind::HostToDevice,
                )
            }
            .unwrap();
            drop(data);
            stream.synchronize().unwrap();
        }
        context.check_err().unwrap();
    }

    #[test]
    fn cpu_adapters_preserve_input_and_replace_output_storage() {
        extern "C" {
            fn rosidl_buffer_uint8_create_cpu(
                data: *const u8,
                len: usize,
                output: *mut *mut c_void,
            ) -> i32;
        }
        let context = CudaContext::new(0).unwrap();
        for stream in [context.default_stream(), context.new_stream().unwrap()] {
            let mut pointer = std::ptr::null_mut();
            let values = [1u8, 2, 3, 4];
            // SAFETY: values and pointer are valid for this synchronous construction.
            assert_eq!(
                unsafe { rosidl_buffer_uint8_create_cpu(values.as_ptr(), 4, &mut pointer) },
                0
            );
            // SAFETY: create_cpu transferred the sole native owner of four bytes.
            let mut opaque = unsafe { CudaBuffer::from_raw(pointer, 4) }.unwrap();
            assert!(opaque.get_read_handle::<u8>(&stream).is_err());
            assert!(opaque.get_write_handle::<u8>(&stream).is_err());
            for mut buffer in [
                rosidl_runtime_rs::Buffer::from(&values[..]),
                opaque.into_buffer(),
            ] {
                let owner = buffer.as_sequence().rosidl_buffer_ptr();
                let references = Arc::strong_count(&stream);
                let input = from_input_buffer::<u8>(&buffer, &stream).unwrap();
                assert_eq!(input.to_host_vec().unwrap(), values);
                drop(input);
                assert_eq!(Arc::strong_count(&stream), references);
                assert_eq!(buffer.backend_name().unwrap(), "cpu");
                assert_eq!(buffer.as_sequence().rosidl_buffer_ptr(), owner);
                assert_eq!(buffer.to_vec().unwrap(), values);
                assert!(from_output_buffer::<u64>(&mut buffer, &stream).is_err());
                assert!(from_output_buffer::<()>(&mut buffer, &stream).is_err());
                assert_eq!(buffer.as_sequence().rosidl_buffer_ptr(), owner);
                let mut output = from_output_buffer::<u8>(&mut buffer, &stream).unwrap();
                output.copy_from_host(&[4, 3, 2, 1]).unwrap();
                let input = from_input_buffer::<u8>(&buffer, &stream).unwrap();
                assert_eq!(input.to_host_vec().unwrap(), [4, 3, 2, 1]);
                drop(input);
                assert_eq!(buffer.backend_name().unwrap(), "cuda");
                assert!(from_output_buffer::<u8>(&mut buffer, &stream).is_err());
            }
            context.check_err().unwrap();
        }
    }
}
