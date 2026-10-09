// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

use rclrs::{Executor, SpinOptions};
use std::time::{Duration, Instant};

pub const IMAGE_TOPIC: &str = "test_cuda_image";
pub const WIDTH: u32 = 256;
pub const HEIGHT: u32 = 128;
pub const BYTES: usize = (WIDTH * HEIGHT) as usize;
pub const SAMPLES: u32 = 10;
pub const TEST_TIMEOUT: Duration = Duration::from_secs(45);
pub const NEGOTIATION_RETRY_INTERVAL: Duration = Duration::from_millis(100);
// Allow reliable acknowledgements to leave the process and observe late duplicates.
pub const SHUTDOWN_GRACE: Duration = Duration::from_millis(300);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Backend {
    Cuda,
    Cpu,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cuda => "GPU",
            Self::Cpu => "CPU",
        }
    }
}

pub fn acknowledgement_topic(subscriber_id: usize) -> String {
    format!("{IMAGE_TOPIC}/ack/subscriber_{subscriber_id}")
}

pub fn spin(executor: &mut Executor, deadline: Instant) {
    assert!(Instant::now() < deadline, "image delivery timed out");
    let errors = executor.spin(SpinOptions::spin_once().timeout(Duration::from_millis(10)));
    assert!(
        errors.iter().all(rclrs::RclrsError::is_timeout),
        "{errors:?}"
    );
}

pub fn spin_for(executor: &mut Executor, duration: Duration, deadline: Instant) {
    let until = Instant::now() + duration;
    while Instant::now() < until {
        spin(executor, deadline);
    }
}
