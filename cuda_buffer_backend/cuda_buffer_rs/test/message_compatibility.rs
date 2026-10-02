// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rclrs::{Context, CreateBasicExecutor, SpinOptions};
use ros_env::{rcl_interfaces, sensor_msgs, test_msgs};
use rosidl_runtime_rs::{Buffer, Message};

fn make_cuda_buffer(values: &[u8]) -> Buffer<u8> {
    let mut buffer = Buffer::from(vec![0u8; values.len()]);
    let stream = cuda_core::CudaContext::new(0).unwrap().default_stream();
    cuda_buffer_rs::from_output_buffer::<u8>(&mut buffer, &stream)
        .unwrap()
        .copy_from_host(values)
        .unwrap();
    buffer
}

#[test]
fn nested_cuda_fields_convert_to_cpu() {
    let mut nested = test_msgs::msg::buffer::MultiNested::default();
    let unbounded = test_msgs::msg::buffer::UnboundedSequences {
        uint8_values: make_cuda_buffer(&[9, 7, 5]),
        ..Default::default()
    };
    nested.array_of_unbounded_sequences[0] = unbounded.clone();
    nested.bounded_sequence_of_unbounded_sequences = vec![unbounded.clone()].try_into().unwrap();
    nested.unbounded_sequence_of_unbounded_sequences = vec![unbounded];
    nested.array_of_bounded_sequences[0].uint8_values =
        make_cuda_buffer(&[3, 2, 1]).try_into().unwrap();
    let native =
        test_msgs::msg::buffer::MultiNested::into_rmw_message(Cow::Borrowed(&nested)).into_owned();
    assert!(native.array_of_unbounded_sequences[0]
        .uint8_values
        .is_rosidl_buffer());
    let cpu = test_msgs::msg::MultiNested::try_from_rmw_message(native).unwrap();
    assert_eq!(
        cpu.array_of_unbounded_sequences[0].uint8_values,
        vec![9, 7, 5]
    );
    assert_eq!(
        cpu.bounded_sequence_of_unbounded_sequences[0]
            .uint8_values
            .as_slice(),
        &[9, 7, 5]
    );
    assert_eq!(
        cpu.unbounded_sequence_of_unbounded_sequences[0].uint8_values,
        vec![9, 7, 5]
    );
    assert_eq!(
        cpu.array_of_bounded_sequences[0].uint8_values.as_slice(),
        &[3, 2, 1]
    );
    assert_eq!(nested.try_into_cpu().unwrap(), cpu);
}

#[test]
fn owned_message_roundtrip_preserves_buffer_owner() {
    let image = sensor_msgs::msg::buffer::Image {
        data: make_cuda_buffer(&[4, 8, 12]),
        ..Default::default()
    };
    let pointer = image.data.as_sequence().rosidl_buffer_ptr();
    assert!(pointer.is_some());
    assert!(image.data.as_slice().is_none());
    let native = sensor_msgs::msg::buffer::Image::into_rmw_message(Cow::Owned(image)).into_owned();
    assert_eq!(native.data.rosidl_buffer_ptr(), pointer);
    let image = sensor_msgs::msg::buffer::Image::from_rmw_message(native);
    assert_eq!(image.data.as_sequence().rosidl_buffer_ptr(), pointer);
    assert_eq!(image.data.to_vec().unwrap(), vec![4, 8, 12]);
    let copy = image.clone();
    drop(image);
    assert_eq!(copy.try_into_cpu().unwrap().data, vec![4, 8, 12]);
}

#[test]
fn cuda_buffer_json_matches_the_cpu_schema() {
    let image = sensor_msgs::msg::Image {
        data: vec![1, 2, 3],
        ..Default::default()
    };
    let json = serde_json::to_string(&image).unwrap();
    let gpu = sensor_msgs::msg::buffer::Image {
        data: make_cuda_buffer(&[1, 2, 3]),
        ..Default::default()
    };
    assert_eq!(serde_json::to_string(&gpu).unwrap(), json);
}

#[test]
fn cpu_client_receives_nested_cuda_service_response() {
    use rcl_interfaces::{msg, srv};
    let mut executor = Context::default().create_basic_executor();
    let node = executor.create_node("portable_service_test").unwrap();
    let name = format!("portable_service_cuda_{}", std::process::id());
    let _service = node
        .create_service::<srv::buffer::GetParameters, _>(
            &name,
            move |request: srv::buffer::GetParameters_Request| {
                assert_eq!(request.names, vec!["pixels"]);
                srv::buffer::GetParameters_Response {
                    values: vec![msg::buffer::ParameterValue {
                        byte_array_value: make_cuda_buffer(&[13, 17, 23]),
                        ..Default::default()
                    }],
                }
            },
        )
        .unwrap();
    let client = node.create_client::<srv::GetParameters>(&name).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !client.service_is_ready().unwrap() {
        assert!(Instant::now() < deadline, "service discovery timed out");
        std::thread::sleep(Duration::from_millis(10));
    }
    let received = Arc::new(Mutex::new(None));
    let output = Arc::clone(&received);
    let _call = client
        .call_then(
            srv::GetParameters_Request {
                names: vec!["pixels".into()],
            },
            move |response: srv::GetParameters_Response| {
                *output.lock().unwrap() = Some(response.values[0].byte_array_value.clone());
            },
        )
        .unwrap();
    loop {
        let errors = executor.spin(SpinOptions::spin_once().timeout(Duration::from_millis(20)));
        assert!(
            errors.iter().all(rclrs::RclrsError::is_timeout),
            "{errors:?}"
        );
        if let Some(values) = &*received.lock().unwrap() {
            assert_eq!(values, &[13, 17, 23]);
            break;
        }
        assert!(Instant::now() < deadline, "service response timed out");
    }
}

#[test]
fn buffer_client_sends_nested_cuda_request_to_cpu_service() {
    use rcl_interfaces::{msg, srv};
    let mut executor = Context::default().create_basic_executor();
    let node = executor.create_node("portable_request_test").unwrap();
    let name = format!("portable_request_{}", std::process::id());
    let _service = node
        .create_service::<srv::SetParameters, _>(&name, |request: srv::SetParameters_Request| {
            assert_eq!(request.parameters[0].value.byte_array_value, vec![29, 31]);
            srv::SetParameters_Response {
                results: vec![msg::SetParametersResult {
                    successful: true,
                    reason: "received".into(),
                }],
            }
        })
        .unwrap();
    let client = node
        .create_client::<srv::buffer::SetParameters>(&name)
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !client.service_is_ready().unwrap() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(10));
    }
    let request = srv::buffer::SetParameters_Request {
        parameters: vec![msg::buffer::Parameter {
            name: "pixels".into(),
            value: msg::buffer::ParameterValue {
                byte_array_value: make_cuda_buffer(&[29, 31]),
                ..Default::default()
            },
        }],
    };
    let received = Arc::new(Mutex::new(false));
    let output = Arc::clone(&received);
    let _call = client
        .call_then(
            request,
            move |response: srv::buffer::SetParameters_Response| {
                assert!(response.results[0].successful);
                assert_eq!(response.results[0].reason, "received");
                *output.lock().unwrap() = true;
            },
        )
        .unwrap();
    while !*received.lock().unwrap() {
        assert!(Instant::now() < deadline, "service response timed out");
        let errors = executor.spin(SpinOptions::spin_once().timeout(Duration::from_millis(20)));
        assert!(
            errors.iter().all(rclrs::RclrsError::is_timeout),
            "{errors:?}"
        );
    }
}
