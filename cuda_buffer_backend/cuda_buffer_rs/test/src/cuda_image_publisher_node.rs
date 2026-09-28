// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cuda_buffer_rs::{allocate_buffer, from_output_buffer};
use cuda_core::{launch_kernel_on_stream, CudaContext, CudaFunction, CudaStream};
use rclrs::{Executor, Node, Publisher, Subscription};
use ros_env::{sensor_msgs::msg::buffer::Image, std_msgs::msg::UInt32};

use crate::common::{
    acknowledgement_topic, spin, spin_for, BYTES, HEIGHT, IMAGE_TOPIC, NEGOTIATION_RETRY_INTERVAL,
    SAMPLES, SHUTDOWN_GRACE, WIDTH,
};

const THREADS_PER_BLOCK: u32 = 256;
const PRODUCER_DELAY: Duration = Duration::from_millis(30);

const FILL: &str = r#"
.version 7.0
.target sm_52
.address_size 64
.visible .entry fill(.param .u64 address, .param .u32 size, .param .u32 seed) {
    .reg .u32 i, block, threads, n, v;
    .reg .u64 p, offset;
    .reg .pred done;
    mov.u32 i, %tid.x;
    mov.u32 block, %ctaid.x;
    mov.u32 threads, %ntid.x;
    mad.lo.u32 i, block, threads, i;
    ld.param.u32 n, [size];
    setp.ge.u32 done, i, n;
    @done bra end;
    ld.param.u64 p, [address];
    ld.param.u32 v, [seed];
    mul.lo.u32 v, v, 31;
    mad.lo.u32 v, i, 17, v;
    cvt.u64.u32 offset, i;
    add.u64 p, p, offset;
    st.global.u8 [p], v;
end:
    ret;
}
"#;

fn publish_image(
    publisher: &Publisher<Image>,
    sequence: u32,
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
) {
    let mut image = Image {
        height: HEIGHT,
        width: WIDTH,
        encoding: "mono8".into(),
        step: WIDTH,
        data: allocate_buffer(BYTES).unwrap(),
        ..Default::default()
    };
    image.header.stamp.sec = sequence as i32;
    let mut output = from_output_buffer::<u8>(&mut image.data, stream).unwrap();
    // SAFETY: borrow only the address; the facade is never replaced or freed.
    // All writes below use the handle's stream, before publishing its owner.
    let mut address = unsafe { output.as_device_buffer_mut() }.cu_deviceptr();
    let mut size = BYTES as u32;
    let mut seed = sequence;
    let mut args = [
        (&mut address as *mut u64).cast::<c_void>(),
        (&mut size as *mut u32).cast(),
        (&mut seed as *mut u32).cast(),
    ];
    // Deliberately delay the producer to exercise cross-stream event ordering.
    stream
        .launch_host_function(|| std::thread::sleep(PRODUCER_DELAY))
        .unwrap();
    // SAFETY: the kernel writes exactly BYTES bytes; the caller retains its module.
    unsafe {
        launch_kernel_on_stream(
            function,
            ((BYTES as u32).div_ceil(THREADS_PER_BLOCK), 1, 1),
            (THREADS_PER_BLOCK, 1, 1),
            0,
            stream,
            &mut args,
        )
    }
    .unwrap();
    publisher.publish(image).unwrap();
}

pub struct CudaImagePublisher {
    _node: Node,
    publisher: Publisher<Image>,
    _ack_subscriptions: Vec<Subscription<UInt32>>,
    acks: Arc<Mutex<Vec<Option<u32>>>>,
}

impl CudaImagePublisher {
    pub fn new(executor: &mut Executor, total: usize) -> Self {
        assert!(total > 0);
        let node = executor.create_node("cuda_image_publisher").unwrap();
        // None means the subscriber has not acknowledged negotiation yet.
        let acks = Arc::new(Mutex::new(vec![None; total]));
        let mut ack_subscriptions = Vec::new();
        for id in 0..total {
            let progress = acks.clone();
            ack_subscriptions.push(
                node.create_subscription::<UInt32, _>(
                    &*acknowledgement_topic(id),
                    move |ack: UInt32| {
                        let mut counts = progress.lock().unwrap();
                        let sequence = ack.data;
                        if sequence == 0 {
                            assert!(
                                matches!(counts[id], None | Some(0)),
                                "negotiation acknowledgement after numbered samples"
                            );
                        } else {
                            assert_eq!(
                                counts[id],
                                Some(sequence - 1),
                                "duplicate or out-of-order acknowledgement from subscriber {id}"
                            );
                        }
                        counts[id] = Some(sequence);
                    },
                )
                .unwrap(),
            );
        }
        let publisher = node.create_publisher::<Image>(IMAGE_TOPIC).unwrap();
        Self {
            _node: node,
            publisher,
            _ack_subscriptions: ack_subscriptions,
            acks,
        }
    }

    pub fn run(&self, executor: &mut Executor, deadline: Instant) {
        let publisher = &self.publisher;
        let acks = &self.acks;
        let total = acks.lock().unwrap().len();
        let context = CudaContext::new(0).unwrap();
        let streams = [context.default_stream(), context.new_stream().unwrap()];
        let module = context.load_module_from_ptx_src(FILL).unwrap();
        let function = module.load_function("fill").unwrap();
        while publisher.get_subscription_count().unwrap() < total {
            spin(executor, deadline);
        }
        // Complete backend descriptor negotiation before numbered samples.
        while acks.lock().unwrap().iter().any(Option::is_none) {
            publish_image(publisher, 0, &streams[0], &function);
            spin_for(executor, NEGOTIATION_RETRY_INTERVAL, deadline);
        }
        for sequence in 1..=SAMPLES {
            let stream = &streams[(sequence > SAMPLES / 2) as usize];
            publish_image(publisher, sequence, stream, &function);
            while acks.lock().unwrap().iter().any(|&n| n != Some(sequence)) {
                spin(executor, deadline);
            }
        }
        spin_for(executor, SHUTDOWN_GRACE, deadline);

        context.check_err().unwrap();
        println!("PUBLISHER_PASS samples={SAMPLES} subscribers={total}");
    }
}
