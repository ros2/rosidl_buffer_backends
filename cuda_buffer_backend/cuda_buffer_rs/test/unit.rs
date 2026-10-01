// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

use cuda_buffer_rs::{allocate_buffer, read_buffer, write_buffer, CudaStream, ErrorKind};
use rosidl_runtime_rs::{Buffer, PrimitiveSequence};

#[test]
fn raw_guards_share_device_memory_and_finalize_one_write() {
    let internal = cuda_buffer_rs::internal_stream().unwrap();
    assert!(!internal.as_raw().is_null());
    assert!(!internal.is_internal());
    assert!(CudaStream::INTERNAL.is_internal());
    for stream in [internal, CudaStream::INTERNAL] {
        let mut buffer = allocate_buffer(2048).unwrap();
        assert_eq!(buffer.len(), 2048);
        assert!(!buffer.is_empty());
        assert_eq!(buffer.backend_name().unwrap(), "cuda");
        let written = {
            let guard = write_buffer(&mut buffer, stream).unwrap();
            assert_eq!(guard.len(), 2048);
            assert!(!guard.device_ptr().is_null());
            guard.device_ptr()
        };
        let error = write_buffer(&mut buffer, stream).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Cuda);
        assert!(error.message.contains("finalized"));
        let guard = read_buffer(&buffer, stream).unwrap();
        assert_eq!(guard.len(), 2048);
        assert_eq!(guard.device_ptr(), written.cast_const());
    }
}

#[test]
fn buffer_and_sequence_conversions_transfer_ownership() {
    let buffer: Buffer<u8> = allocate_buffer(512).unwrap();
    let raw = buffer.as_sequence().rosidl_buffer_ptr().unwrap();
    let sequence = buffer.into_sequence();
    assert_eq!(sequence.rosidl_buffer_ptr(), Some(raw));
    assert_eq!(sequence.len(), 512);
    let guard = cuda_buffer_rs::read_primitive_sequence(&sequence, CudaStream::INTERNAL).unwrap();
    let device_ptr = guard.device_ptr();
    assert_eq!(guard.len(), 512);
    assert!(!device_ptr.is_null());
    drop(guard);

    let reclaimed = Buffer::from(sequence);
    assert_eq!(reclaimed.len(), 512);
    assert_eq!(reclaimed.backend_name().unwrap(), "cuda");
    assert_eq!(reclaimed.as_sequence().rosidl_buffer_ptr(), Some(raw));
    let read = read_buffer(&reclaimed, CudaStream::INTERNAL).unwrap();
    assert_eq!(read.device_ptr(), device_ptr);
}

#[test]
fn empty_buffers_reject_guards_and_errors_survive_later_ffi_calls() {
    let mut buffer = allocate_buffer(0).unwrap();
    assert!(buffer.is_empty());
    let error = read_buffer(&buffer, CudaStream::INTERNAL).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidArgument);
    let message = error.message.clone();
    assert!(!message.is_empty());
    assert!(std::error::Error::source(&error).is_none());
    assert_eq!(
        write_buffer(&mut buffer, CudaStream::INTERNAL)
            .unwrap_err()
            .kind,
        ErrorKind::InvalidArgument
    );
    let _stream = cuda_buffer_rs::internal_stream().unwrap();
    assert_eq!(error.message, message);
    assert!(error.to_string().contains(&message));
    assert!(!unsafe { cuda_buffer_rs::is_cuda_backed(std::ptr::null()) });
}

#[test]
fn normal_sequences_are_preserved_on_rejected_raw_read() {
    let sequence = PrimitiveSequence::from(&[1u8, 2, 3][..]);
    let error =
        cuda_buffer_rs::read_primitive_sequence(&sequence, CudaStream::INTERNAL).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidArgument);
    assert_eq!(sequence.as_slice(), &[1, 2, 3]);
    let buffer = Buffer::from(sequence);
    assert_eq!(
        read_buffer(&buffer, CudaStream::INTERNAL).unwrap_err().kind,
        ErrorKind::InvalidArgument
    );
    assert_eq!(buffer.as_slice().unwrap(), &[1, 2, 3]);
}

#[test]
fn raw_write_promotes_cpu_storage_and_preserves_empty_inputs_on_failure() {
    let mut buffer = Buffer::from(vec![1u8, 2, 3]);
    let written = {
        let output = write_buffer(&mut buffer, CudaStream::INTERNAL).unwrap();
        assert_eq!(output.len(), 3);
        output.device_ptr()
    };
    assert_eq!(buffer.backend_name().unwrap(), "cuda");
    let input = read_buffer(&buffer, CudaStream::INTERNAL).unwrap();
    assert_eq!(input.device_ptr(), written.cast_const());

    for mut empty in [Buffer::<u8>::default(), allocate_buffer(0).unwrap()] {
        let owner = empty.as_sequence().rosidl_buffer_ptr();
        let backend = empty.backend_name().unwrap();
        assert_eq!(
            write_buffer(&mut empty, CudaStream::INTERNAL)
                .unwrap_err()
                .kind,
            ErrorKind::InvalidArgument
        );
        assert_eq!(empty.as_sequence().rosidl_buffer_ptr(), owner);
        assert_eq!(empty.backend_name().unwrap(), backend);
    }
}

#[cfg(feature = "cuda-core")]
mod typed {
    use super::*;
    use cuda_buffer_rs::{from_input_buffer, from_output_buffer, to_buffer, CopyKind};
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
        let mut buffer = allocate_buffer(7).unwrap();
        assert_eq!(
            from_output_buffer::<u32>(&mut buffer, &stream)
                .err()
                .unwrap()
                .kind,
            ErrorKind::InvalidArgument
        );
        assert_eq!(
            from_output_buffer::<()>(&mut buffer, &stream)
                .err()
                .unwrap()
                .kind,
            ErrorKind::InvalidArgument
        );
        let mut writer = from_output_buffer::<u8>(&mut buffer, &stream).unwrap();
        assert_eq!(
            writer.copy_from_host(&[1, 2]).unwrap_err().kind,
            ErrorKind::InvalidArgument
        );
        writer.copy_from_host(&[3; 7]).unwrap();
        drop(writer);
        assert!(from_input_buffer::<u32>(&buffer, &stream).is_err());
        assert!(from_input_buffer::<()>(&buffer, &stream).is_err());
        assert_eq!(
            from_input_buffer::<u8>(&buffer, &stream)
                .unwrap()
                .to_host_vec()
                .unwrap(),
            vec![3; 7]
        );
        assert!(from_input_buffer::<u8>(&allocate_buffer(0).unwrap(), &stream).is_err());
    }

    #[test]
    fn stream_and_context_references_are_released_on_drop_and_unwind() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let context_refs = Arc::strong_count(&context);
        let stream_refs = Arc::strong_count(&stream);
        let mut buffer = allocate_buffer(128).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut writer = from_output_buffer::<u32>(&mut buffer, &stream).unwrap();
            writer.copy_from_host(&[91; 32]).unwrap();
            panic!("exercise unwinding after native acquisition");
        }));
        assert!(result.is_err());
        assert_eq!(Arc::strong_count(&context), context_refs);
        assert_eq!(Arc::strong_count(&stream), stream_refs);
        for _ in 0..2 {
            let handle = from_input_buffer::<u32>(&buffer, &stream).unwrap();
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
        let mut buffer = allocate_buffer(128).unwrap();
        let mut writer = from_output_buffer::<u32>(&mut buffer, &stream).unwrap();
        drop(stream);
        assert!(weak.upgrade().is_some());
        writer.copy_from_host(&[12; 32]).unwrap();
        drop(writer);
        assert!(weak.upgrade().is_none());
        let stream = context.default_stream();
        assert_eq!(
            from_input_buffer::<u32>(&buffer, &stream)
                .unwrap()
                .to_host_vec()
                .unwrap(),
            vec![12; 32]
        );
    }

    #[test]
    fn device_views_borrow_the_message_allocation() {
        let context = CudaContext::new(0).unwrap();
        for stream in [context.default_stream(), context.new_stream().unwrap()] {
            let mut data: Buffer<u8> = allocate_buffer(16).unwrap();
            let owner = data.as_sequence().rosidl_buffer_ptr();
            let address = {
                let mut output = from_output_buffer::<u32>(&mut data, &stream).unwrap();
                let address = output.get_ptr() as usize as u64;
                // SAFETY: only inspect the view; ownership and metadata stay unchanged.
                let view: &mut DeviceBuffer<u32> = unsafe { output.as_device_buffer() };
                assert_eq!(view.cu_deviceptr(), address);
                output.copy_from_host(&[11, 13, 17, 19]).unwrap();
                address
            };
            let input = from_input_buffer::<u32>(&data, &stream).unwrap();
            // SAFETY: only inspect the read-only view.
            let view: &DeviceBuffer<u32> = unsafe { input.as_device_buffer() };
            assert_eq!(view.cu_deviceptr(), address);
            assert_eq!(input.len(), 4);
            assert_eq!(input.to_host_vec().unwrap(), [11, 13, 17, 19]);
            drop(input);
            assert_eq!(data.as_sequence().rosidl_buffer_ptr(), owner);
        }
        context.check_err().unwrap();
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
            let opaque = Buffer::from(
                unsafe { PrimitiveSequence::from_owned_rosidl_buffer(pointer, 4) }.unwrap(),
            );
            for mut buffer in [rosidl_runtime_rs::Buffer::from(&values[..]), opaque] {
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
