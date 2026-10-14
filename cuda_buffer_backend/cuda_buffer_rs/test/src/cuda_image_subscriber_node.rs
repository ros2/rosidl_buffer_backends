// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use cuda_buffer_rs::from_input_buffer;
use cuda_core::CudaContext;
use rclrs::{Executor, Node, Subscription, SubscriptionOptions};
use ros_env::{
    sensor_msgs::msg::{self, buffer::Image},
    std_msgs::msg::UInt32,
};

use crate::common::{acknowledgement_topic, Backend, BYTES, HEIGHT, IMAGE_TOPIC, SAMPLES, WIDTH};

fn record_sample(received: &AtomicU32, sequence: u32) {
    assert!(sequence <= SAMPLES, "unexpected image sequence {sequence}");
    if sequence == 0 {
        assert_eq!(
            received.load(Ordering::SeqCst),
            0,
            "negotiation image after numbered samples"
        );
    } else {
        assert_eq!(
            sequence,
            received.fetch_add(1, Ordering::SeqCst) + 1,
            "duplicate or out-of-order image"
        );
    }
}

fn verify_pixels(sequence: u32, pixels: &[u8]) {
    assert_eq!(pixels.len(), BYTES);
    for (index, &value) in pixels.iter().enumerate() {
        assert_eq!(
            value,
            (index as u32 * 17 + sequence * 31) as u8,
            "pixel {index}"
        );
    }
}

fn gpu_subscriber(node: &Node, id: usize) -> (Subscription<Image>, Arc<AtomicU32>) {
    let context = CudaContext::new(0).unwrap();
    let streams = [context.new_stream().unwrap(), context.default_stream()];
    let ack = node
        .create_publisher::<UInt32>(&*acknowledgement_topic(id))
        .unwrap();
    let received = Arc::new(AtomicU32::new(0));
    let count = received.clone();
    let subscription = node
        .create_subscription::<Image, _>(
            SubscriptionOptions::new(IMAGE_TOPIC).acceptable_buffer_backends("cuda"),
            move |message: Image| {
                let sequence = message.header.stamp.sec as u32;
                if sequence == 0
                    && (message.data.is_empty() || message.data.backend_name().unwrap() != "cuda")
                {
                    return;
                }
                record_sample(&count, sequence);
                assert_eq!(message.data.backend_name().unwrap(), "cuda");
                assert_eq!(
                    (message.width, message.height, message.step),
                    (WIDTH, HEIGHT, WIDTH)
                );
                assert_eq!(message.encoding, "mono8");
                let stream = &streams[(sequence > SAMPLES / 2) as usize];
                let input = from_input_buffer::<u8>(&message.data, stream).unwrap();
                assert_eq!(input.stream().cu_stream(), stream.cu_stream());
                verify_pixels(sequence, &input.to_host_vec().unwrap());
                ack.publish(UInt32 { data: sequence }).unwrap();
            },
        )
        .unwrap();
    (subscription, received)
}

fn cpu_subscriber(node: &Node, id: usize) -> (Subscription<msg::Image>, Arc<AtomicU32>) {
    let ack = node
        .create_publisher::<UInt32>(&*acknowledgement_topic(id))
        .unwrap();
    let received = Arc::new(AtomicU32::new(0));
    let count = received.clone();
    let subscription = node
        .create_subscription::<msg::Image, _>(IMAGE_TOPIC, move |message: msg::Image| {
            let sequence = message.header.stamp.sec as u32;
            record_sample(&count, sequence);
            assert_eq!(
                (message.width, message.height, message.step),
                (WIDTH, HEIGHT, WIDTH)
            );
            assert_eq!(message.encoding, "mono8");
            verify_pixels(sequence, &message.data);
            ack.publish(UInt32 { data: sequence }).unwrap();
        })
        .unwrap();
    (subscription, received)
}

pub struct CudaImageSubscriber {
    _node: Node,
    _gpu_subscription: Option<Subscription<Image>>,
    _cpu_subscription: Option<Subscription<msg::Image>>,
    received: Arc<AtomicU32>,
    id: usize,
    backend: Backend,
}

impl CudaImageSubscriber {
    pub fn new(executor: &mut Executor, id: usize, backend: Backend) -> Self {
        let node = executor
            .create_node(&*format!("cuda_image_subscriber_{id}"))
            .unwrap();
        match backend {
            Backend::Cuda => {
                let (subscription, received) = gpu_subscriber(&node, id);
                Self {
                    _node: node,
                    _gpu_subscription: Some(subscription),
                    _cpu_subscription: None,
                    received,
                    id,
                    backend: Backend::Cuda,
                }
            }
            Backend::Cpu => {
                let (subscription, received) = cpu_subscriber(&node, id);
                Self {
                    _node: node,
                    _gpu_subscription: None,
                    _cpu_subscription: Some(subscription),
                    received,
                    id,
                    backend: Backend::Cpu,
                }
            }
        }
    }

    pub fn is_complete(&self) -> bool {
        self.received.load(Ordering::SeqCst) == SAMPLES
    }

    pub fn assert_complete(&self) {
        assert!(self.is_complete(), "subscriber did not receive all samples");
        println!(
            "{}_SUBSCRIBER_PASS id={} samples={SAMPLES}",
            self.backend.label(),
            self.id
        );
    }
}
