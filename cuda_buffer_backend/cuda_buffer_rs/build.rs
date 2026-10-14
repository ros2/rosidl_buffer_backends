// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn cuda_search_path(path: &Path) {
    if path.is_dir() {
        println!("cargo:rustc-link-search=native={}", path.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath-link,{}", path.display());
    }
}

fn main() {
    let mut prefixes = Vec::new();
    let mut directories = Vec::new();
    println!("cargo:rerun-if-env-changed=AMENT_PREFIX_PATH");
    if let Some(prefixes) = env::var_os("AMENT_PREFIX_PATH") {
        directories.extend(env::split_paths(&prefixes).map(|prefix| prefix.join("lib")));
    }
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    if let Some(prefix) = env::var_os("CONDA_PREFIX") {
        directories.push(PathBuf::from(prefix).join("lib"));
    }

    prefixes.extend(
        directories
            .iter()
            .filter_map(|directory| directory.parent().map(Path::to_path_buf)),
    );
    let mut build = cxx_build::bridge("src/bridge.rs");
    build.file("src/bridge.cpp").std("c++20");
    for (package, header) in [
        ("cuda_buffer", "cuda_buffer/cuda_buffer_api.hpp"),
        ("rosidl_buffer", "rosidl_buffer/buffer.hpp"),
        ("rmw", "rmw/types.h"),
        ("rcutils", "rcutils/allocator.h"),
        (
            "rosidl_runtime_c",
            "rosidl_runtime_c/message_type_support_struct.h",
        ),
    ] {
        let include = prefixes
            .iter()
            .flat_map(|prefix| [prefix.join("include").join(package), prefix.join("include")])
            .find(|directory| directory.join(header).is_file())
            .unwrap_or_else(|| panic!("{package} headers are missing from the ROS installation"));
        println!("cargo:rerun-if-changed={}", include.display());
        build.include(include);
    }
    let cuda_roots: Vec<_> = ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"]
        .iter()
        .filter_map(env::var_os)
        .map(PathBuf::from)
        .chain([PathBuf::from("/usr/local/cuda"), PathBuf::from("/usr")])
        .collect();
    let cuda_include = cuda_roots
        .iter()
        .map(|root| root.join("include"))
        .find(|directory| directory.join("cuda_runtime.h").is_file())
        .expect("CUDA headers are missing; set CUDA_TOOLKIT_PATH");
    build
        .include(&cuda_include)
        .compile("cuda_buffer_rs_bridge");
    println!("cargo:rerun-if-changed={}", cuda_include.display());
    for path in ["src/bridge.rs", "src/bridge.cpp", "src/bridge.hpp"] {
        println!("cargo:rerun-if-changed={path}");
    }
    for library in ["cudart", "cuda", "rcutils"] {
        println!("cargo:rustc-link-lib=dylib={library}");
    }

    // Give rustdoc one directory containing the selected native libraries, so
    // reordering search paths cannot select libraries from another ROS overlay.
    let links =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set")).join("native-libraries");
    fs::create_dir_all(&links).expect("create native library directory");
    for library in ["cuda_buffer", "rosidl_buffer"] {
        let filename = format!("lib{library}.so");
        let source = directories
            .iter()
            .map(|directory| directory.join(&filename))
            .find(|path| path.is_file())
            .unwrap_or_else(|| {
                panic!("{library} was not found in AMENT_PREFIX_PATH or CONDA_PREFIX")
            });
        println!("cargo:rerun-if-changed={}", source.display());
        let target = links.join(filename);
        if target.symlink_metadata().is_ok() {
            fs::remove_file(&target).expect("remove previous native library link");
        }
        std::os::unix::fs::symlink(
            fs::canonicalize(&source).expect("resolve native library"),
            target,
        )
        .expect("link native library");
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath-link,{}",
            source.parent().unwrap().display()
        );
        println!("cargo:rustc-link-lib=dylib={library}");
    }
    println!("cargo:rustc-link-search=native={}", links.display());

    for variable in ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"] {
        println!("cargo:rerun-if-env-changed={variable}");
        if let Some(root) = env::var_os(variable) {
            cuda_search_path(&PathBuf::from(root).join("lib64"));
        }
    }
    cuda_search_path(Path::new("/usr/local/cuda/lib64"));
}
