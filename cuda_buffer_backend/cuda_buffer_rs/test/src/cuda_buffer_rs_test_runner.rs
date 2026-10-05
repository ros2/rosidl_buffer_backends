// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

use rclrs::{Context, CreateBasicExecutor};
use std::process::ExitCode;
use std::time::Instant;

mod common;
mod cuda_image_publisher_node;
mod cuda_image_subscriber_node;

use common::{Backend, SHUTDOWN_GRACE, TEST_TIMEOUT};
use cuda_image_publisher_node::CudaImagePublisher;
use cuda_image_subscriber_node::CudaImageSubscriber;

const USAGE: &str = "Usage: cuda_buffer_rs_test_runner publisher <subscriber-count>\n\
                    cuda_buffer_rs_test_runner subscriber <id> [cuda|cpu]\n\
                    cuda_buffer_rs_test_runner composed <subscriber-count>";

enum Mode {
    Publisher(usize),
    Subscriber { id: usize, backend: Backend },
    Composed(usize),
}

fn parse_mode(args: &[&str]) -> Result<Mode, String> {
    let number = |value: &str| {
        value
            .parse::<usize>()
            .map_err(|_| format!("invalid number: {value}"))
    };
    match args {
        [mode @ ("publisher" | "composed"), count] => {
            let count = number(count)?;
            if count == 0 {
                return Err("subscriber count must be positive".into());
            }
            Ok(if *mode == "publisher" {
                Mode::Publisher(count)
            } else {
                Mode::Composed(count)
            })
        }
        ["subscriber", id] | ["subscriber", id, "cuda"] => Ok(Mode::Subscriber {
            id: number(id)?,
            backend: Backend::Cuda,
        }),
        ["subscriber", id, "cpu"] => Ok(Mode::Subscriber {
            id: number(id)?,
            backend: Backend::Cpu,
        }),
        _ => Err("invalid mode or arguments".into()),
    }
}

fn main() -> ExitCode {
    // launch_ros appends ROS arguments; leave those for rclrs to process.
    let args: Vec<_> = std::env::args()
        .skip(1)
        .take_while(|arg| arg != "--ros-args")
        .collect();
    let args: Vec<_> = args.iter().map(String::as_str).collect();
    if args == ["--help"] {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let mode = match parse_mode(&args) {
        Ok(mode) => mode,
        Err(error) => {
            eprintln!("{error}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let mut executor = Context::default_from_env().unwrap().create_basic_executor();
    let deadline = Instant::now() + TEST_TIMEOUT;
    match mode {
        Mode::Publisher(count) => {
            CudaImagePublisher::new(&mut executor, count).run(&mut executor, deadline);
        }
        Mode::Subscriber { id, backend } => {
            let subscriber = CudaImageSubscriber::new(&mut executor, id, backend);
            while !subscriber.is_complete() {
                common::spin(&mut executor, deadline);
            }
            common::spin_for(&mut executor, SHUTDOWN_GRACE, deadline);
            subscriber.assert_complete();
        }
        Mode::Composed(count) => {
            // Use the same node implementations, hosted by one executor.
            let subscribers: Vec<_> = (0..count)
                .map(|id| CudaImageSubscriber::new(&mut executor, id, Backend::Cuda))
                .collect();
            CudaImagePublisher::new(&mut executor, count).run(&mut executor, deadline);
            for subscriber in subscribers {
                subscriber.assert_complete();
            }
        }
    }
    ExitCode::SUCCESS
}
