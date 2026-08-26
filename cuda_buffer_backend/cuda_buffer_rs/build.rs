use std::env;
use std::path::{Path, PathBuf};

const NATIVE_LIBRARIES: &[&str] = &["cuda_buffer", "rosidl_buffer"];

fn add_library_path(path: &Path) {
    if path.is_dir() {
        println!("cargo:rustc-link-search=native={}", path.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath-link,{}", path.display());
    }
}

fn add_rpath_link(path: &Path) {
    if path.is_dir() {
        println!("cargo:rustc-link-arg=-Wl,-rpath-link,{}", path.display());
    }
}

fn main() {
    for prefix in env::var("AMENT_PREFIX_PATH")
        .unwrap_or_default()
        .split(':')
        .filter(|prefix| !prefix.is_empty())
    {
        let library_path = PathBuf::from(prefix).join("lib");
        add_library_path(&library_path);
    }

    if let Some(prefix) = env::var_os("CONDA_PREFIX") {
        add_library_path(&PathBuf::from(prefix).join("lib"));
    }

    for cuda_root in [
        env::var_os("CUDA_HOME").map(PathBuf::from),
        env::var_os("CUDA_PATH").map(PathBuf::from),
        Some(PathBuf::from("/usr/local/cuda")),
        Some(PathBuf::from("/usr/local/cuda-11.8")),
    ]
    .into_iter()
    .flatten()
    {
        add_library_path(&cuda_root.join("lib64"));
        add_library_path(&cuda_root.join("targets/x86_64-linux/lib"));
    }
    let system_library_path = Path::new("/lib/x86_64-linux-gnu");
    add_rpath_link(system_library_path);

    for library in NATIVE_LIBRARIES {
        println!("cargo:rustc-link-lib=dylib={library}");
    }
    for library in ["stdc++", "atomic", "cudart"] {
        println!("cargo:rustc-link-lib=dylib={library}");
    }
    let cuda_driver = system_library_path.join("libcuda.so.1");
    if cuda_driver.is_file() {
        println!("cargo:rustc-link-arg={}", cuda_driver.display());
    } else {
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    println!("cargo:rerun-if-env-changed=AMENT_PREFIX_PATH");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
