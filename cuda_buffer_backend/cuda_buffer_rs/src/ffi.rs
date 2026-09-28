// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Raw declarations for the `cuda_buffer` C ABI.
//!
//! `buffer` arguments are opaque `rosidl::Buffer<uint8_t> *` values and
//! `cuda_stream` arguments are raw `cudaStream_t` values. Null means CUDA default
//! stream 0. To use the backend's internal stream, pass the pointer returned by
//! `cuda_buffer_internal_stream`.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_void};

pub type cuda_buffer_ret_t = c_int;

pub const CUDA_BUFFER_RET_OK: cuda_buffer_ret_t = 0;
pub const CUDA_BUFFER_RET_INVALID_ARGUMENT: cuda_buffer_ret_t = 1;
pub const CUDA_BUFFER_RET_BAD_ALLOC: cuda_buffer_ret_t = 2;
pub const CUDA_BUFFER_RET_CUDA_ERROR: cuda_buffer_ret_t = 3;
pub const CUDA_BUFFER_RET_ERROR: cuda_buffer_ret_t = 4;

pub const CUDA_BUFFER_COPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_BUFFER_COPY_DEVICE_TO_DEVICE: c_int = 3;

#[repr(C)]
pub struct cuda_buffer_read_handle_t {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct cuda_buffer_write_handle_t {
    _opaque: [u8; 0],
}

extern "C" {
    pub fn cuda_buffer_error_message() -> *const c_char;

    pub fn cuda_buffer_internal_stream(cuda_stream: *mut *mut c_void) -> cuda_buffer_ret_t;

    pub fn cuda_buffer_allocate(byte_count: usize, buffer: *mut *mut c_void) -> cuda_buffer_ret_t;

    pub fn cuda_buffer_is_cuda_backed(buffer: *const c_void) -> bool;

    pub fn cuda_buffer_device_id(buffer: *const c_void, device_id: *mut c_int)
        -> cuda_buffer_ret_t;

    pub fn cuda_buffer_acquire_read(
        buffer: *const c_void,
        cuda_stream: *mut c_void,
        handle: *mut *mut cuda_buffer_read_handle_t,
    ) -> cuda_buffer_ret_t;

    pub fn cuda_buffer_acquire_write(
        buffer: *mut *mut c_void,
        cuda_stream: *mut c_void,
        handle: *mut *mut cuda_buffer_write_handle_t,
    ) -> cuda_buffer_ret_t;

    pub fn cuda_buffer_to_buffer_on_stream(
        source: *const c_void,
        byte_count: usize,
        handle: *mut cuda_buffer_write_handle_t,
        cuda_stream: *mut c_void,
        kind: c_int,
    ) -> cuda_buffer_ret_t;

    pub fn cuda_buffer_read_handle_data(handle: *const cuda_buffer_read_handle_t) -> *const u8;

    pub fn cuda_buffer_write_handle_data(handle: *mut cuda_buffer_write_handle_t) -> *mut u8;

    pub fn cuda_buffer_read_handle_size(handle: *const cuda_buffer_read_handle_t) -> usize;

    pub fn cuda_buffer_write_handle_size(handle: *const cuda_buffer_write_handle_t) -> usize;

    pub fn cuda_buffer_read_handle_destroy(handle: *mut cuda_buffer_read_handle_t);

    pub fn cuda_buffer_write_handle_destroy(handle: *mut cuda_buffer_write_handle_t);

    /// Canonical destruction function for `rosidl::Buffer<uint8_t> *`, exported by
    /// `rosidl_buffer`. Accepts null.
    pub fn rosidl_buffer_uint8_destroy(buffer: *mut c_void);
}
