// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

use cuda_buffer_rs::{allocate_buffer, read_buffer, write_buffer, CudaStream, ErrorKind};
use rosidl_runtime_rs::Buffer;

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

    let sequence: rosidl_runtime_rs::Sequence<u8> = sequence.into();
    {
        let guard =
            cuda_buffer_rs::read_primitive_sequence(&sequence, CudaStream::INTERNAL).unwrap();
        assert_eq!(guard.device_ptr(), device_ptr);
        assert_eq!(guard.len(), 512);
    }

    let reclaimed = Buffer::from(sequence);
    assert_eq!(reclaimed.len(), 512);
    assert_eq!(reclaimed.backend_name().unwrap(), "cuda");
    assert_eq!(reclaimed.as_sequence().rosidl_buffer_ptr(), Some(raw));
    let read = read_buffer(&reclaimed, CudaStream::INTERNAL).unwrap();
    assert_eq!(read.device_ptr(), device_ptr);
}

#[test]
fn raw_write_promotes_cpu_storage() {
    let mut buffer = Buffer::from(vec![1u8, 2, 3]);
    let written = {
        let output = write_buffer(&mut buffer, CudaStream::INTERNAL).unwrap();
        assert_eq!(output.len(), 3);
        output.device_ptr()
    };
    assert_eq!(buffer.backend_name().unwrap(), "cuda");
    let input = read_buffer(&buffer, CudaStream::INTERNAL).unwrap();
    assert_eq!(input.device_ptr(), written.cast_const());
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
    fn native_clone_and_equality_use_backend_storage() {
        let context = CudaContext::new(0).unwrap();
        for stream in [context.default_stream(), context.new_stream().unwrap()] {
            let values = [2u8, 4, 6, 8];
            let mut source = allocate_buffer(values.len()).unwrap();
            from_output_buffer::<u8>(&mut source, &stream)
                .unwrap()
                .copy_from_host(&values)
                .unwrap();
            let clone = source.try_clone().unwrap();
            assert_eq!(clone.backend_name().unwrap(), "cuda");
            assert_ne!(
                clone.as_sequence().rosidl_buffer_ptr(),
                source.as_sequence().rosidl_buffer_ptr()
            );
            {
                let original = from_input_buffer::<u8>(&source, &stream).unwrap();
                let copied = from_input_buffer::<u8>(&clone, &stream).unwrap();
                assert_ne!(original.device_ptr(), copied.device_ptr());
            }
            assert_eq!(clone, source);
            let cpu = Buffer::from(values.to_vec());
            assert_eq!(clone, cpu);
            assert_eq!(cpu, clone);
            assert_ne!(clone, Buffer::from(vec![2u8, 4, 6, 9]));
            drop(source);
            assert_eq!(
                from_input_buffer::<u8>(&clone, &stream)
                    .unwrap()
                    .to_host_vec()
                    .unwrap(),
                values
            );
        }
    }

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
                // Only inspect the view; ownership and metadata stay unchanged.
                let view: &mut DeviceBuffer<u32> = output.as_device_buffer();
                assert_eq!(view.cu_deviceptr(), address);
                output.copy_from_host(&[11, 13, 17, 19]).unwrap();
                address
            };
            let input = from_input_buffer::<u32>(&data, &stream).unwrap();
            // Only inspect the read-only view.
            let view: &DeviceBuffer<u32> = input.as_device_buffer();
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
                // Keep values live and unchanged through stream synchronization.
                to_buffer(
                    values.as_ptr().cast(),
                    size_of_val(&values),
                    &mut output,
                    &stream,
                    CopyKind::HostToDevice,
                )
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
        // Keep source and its completed contents live through the consumer read.
        let copied = to_buffer(
            source.cu_deviceptr() as usize as *const c_void,
            size_of_val(&values),
            &mut output,
            &producer,
            CopyKind::DeviceToDevice,
        );
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
    fn unpublished_owner_can_drop_before_unused_write_handle() {
        static VALUES: [u8; 4] = [3, 5, 7, 11];
        let context = CudaContext::new(0).unwrap();
        for stream in [context.default_stream(), context.new_stream().unwrap()] {
            let mut data = allocate_buffer(VALUES.len()).unwrap();
            let mut output = from_output_buffer::<u8>(&mut data, &stream).unwrap();
            // Static VALUES remains valid through the queued copy.
            to_buffer(
                VALUES.as_ptr().cast(),
                VALUES.len(),
                &mut output,
                &stream,
                CopyKind::HostToDevice,
            )
            .unwrap();
            drop(data);
            stream.synchronize().unwrap();
        }
        context.check_err().unwrap();
    }

    #[test]
    fn cpu_adapters_preserve_input_and_replace_output_storage() {
        let context = CudaContext::new(0).unwrap();
        for stream in [context.default_stream(), context.new_stream().unwrap()] {
            let values = [1u8, 2, 3, 4];
            let native = rosidl_runtime_rs::native::ffi::create_cpu(&values).unwrap();
            let opaque = rosidl_runtime_rs::native::into_buffer(native).unwrap();
            for mut buffer in [rosidl_runtime_rs::Buffer::from(&values[..]), opaque] {
                let owner = buffer.as_sequence().rosidl_buffer_ptr();
                let input = from_input_buffer::<u8>(&buffer, &stream).unwrap();
                assert_eq!(input.to_host_vec().unwrap(), values);
                drop(input);
                assert_eq!(buffer.backend_name().unwrap(), "cpu");
                assert_eq!(buffer.as_sequence().rosidl_buffer_ptr(), owner);
                assert_eq!(buffer.to_vec().unwrap(), values);
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
