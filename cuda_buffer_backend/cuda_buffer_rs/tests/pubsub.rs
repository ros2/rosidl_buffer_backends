use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use cuda_buffer_rs::{read_primitive_sequence, CudaBuffer, CudaStream};
use rclrs::{Context, CreateBasicExecutor, SpinOptions, SubscriptionOptions};
use std_msgs::msg::rmw::UInt8MultiArray;

const SUBSCRIBER_TOPIC: &str = "CUDA_BUFFER_RS_SUBSCRIBER_TOPIC";

#[test]
fn cuda_buffer_survives_rust_pubsub() {
    if let Ok(topic) = std::env::var(SUBSCRIBER_TOPIC) {
        run_subscriber(&topic);
        return;
    }

    let mut publisher_executor = Context::default().create_basic_executor();
    let publisher_node_name = format!("cuda_buffer_rs_publisher_{}", std::process::id());
    let publisher_node = publisher_executor
        .create_node(&*publisher_node_name)
        .expect("create publisher node");
    let topic = format!("cuda_buffer_rs_topic_{}", std::process::id());
    let publisher = publisher_node
        .create_publisher::<UInt8MultiArray>(&topic)
        .expect("create publisher");
    let mut subscriber = Command::new(std::env::current_exe().expect("test executable"))
        .args(["--exact", "cuda_buffer_survives_rust_pubsub", "--nocapture"])
        .env(SUBSCRIBER_TOPIC, &topic)
        .spawn()
        .expect("start subscriber");

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let mut message = UInt8MultiArray::default();
        message.data = CudaBuffer::allocate(4096)
            .expect("allocate CUDA buffer")
            .into_primitive_sequence();
        publisher.publish(message).expect("publish CUDA buffer");
        publisher_executor.spin(SpinOptions::spin_once().timeout(Duration::from_millis(10)));
        if let Some(status) = subscriber.try_wait().expect("query subscriber") {
            assert!(status.success(), "subscriber failed: {status}");
            return;
        }
        thread::sleep(Duration::from_millis(50));
        assert!(Instant::now() < deadline, "CUDA message was not received");
    }
}

fn run_subscriber(topic: &str) {
    let mut executor = Context::default().create_basic_executor();
    let node = executor
        .create_node("cuda_buffer_rs_subscriber")
        .expect("create subscriber node");
    let received = Arc::new(AtomicBool::new(false));
    let callback_received = Arc::clone(&received);
    let _subscription = node
        .create_subscription::<UInt8MultiArray, _>(
            SubscriptionOptions::new(topic).acceptable_buffer_backends("cuda"),
            move |message: UInt8MultiArray| {
                let valid = message.data.is_rosidl_buffer()
                    && read_primitive_sequence(&message.data, CudaStream::INTERNAL)
                        .map(|guard| guard.len() == 4096 && !guard.device_ptr().is_null())
                        .unwrap_or(false);
                callback_received.store(valid, Ordering::Release);
            },
        )
        .expect("create CUDA subscription");

    let deadline = Instant::now() + Duration::from_secs(10);
    while !received.load(Ordering::Acquire) {
        executor.spin(SpinOptions::spin_once().timeout(Duration::from_millis(100)));
        assert!(Instant::now() < deadline, "CUDA message was not received");
    }
}
