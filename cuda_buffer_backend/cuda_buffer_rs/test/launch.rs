// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

fn run_launch_test(name: &str) {
    let executable = std::env::current_exe().unwrap();
    let profile_dir = executable.parent().unwrap().parent().unwrap();
    let runner = profile_dir.join("examples/cuda_buffer_rs_test_runner");
    assert!(
        runner.is_file(),
        "missing test runner: {}",
        runner.display()
    );

    let results = profile_dir.join("test_results/cuda_buffer_rs");
    std::fs::create_dir_all(&results).unwrap();
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test")
        .join(format!("{name}.py"));
    let output = std::process::Command::new("launch_test")
        .arg("--package-name")
        .arg("cuda_buffer_rs")
        .arg("--junit-xml")
        .arg(results.join(format!("{name}.xml")))
        .arg(script)
        .arg(format!("test_runner:={}", runner.display()))
        .output()
        .expect("failed to run launch_test; source the ROS workspace before testing");
    assert!(
        output.status.success(),
        "{name} failed ({}):\n{}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn cuda_image_inter_pubsub_fastrtps() {
    run_launch_test("test_cuda_image_inter_pubsub_fastrtps_launch");
}

#[test]
fn cuda_image_intra_pubsub_fastrtps() {
    run_launch_test("test_cuda_image_intra_pubsub_fastrtps_launch");
}

#[test]
fn cuda_image_cpu_fallback_fastrtps() {
    run_launch_test("test_cuda_image_cpu_fallback_fastrtps_launch");
}
