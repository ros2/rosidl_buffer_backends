# Design

## Introduction

`qc_buffer_backend` is a `rosidl::Buffer<T>` storage backend that allocates the message payload in Qualcomm ION / dma-buf memory and delivers it to a subscriber **without serialization or host copies** when the runtime conditions allow it. When they don't, the system falls back to the default CPU path automatically.

The optimized path is taken when the publisher and subscriber are on the **same device**. The ION buffer is CPU-accessible and, through its dma-buf fd, can be shared with the HTP (Qualcomm NPU) accelerator so it reads the exact same physical pages the CPU wrote — this is what makes HTP-CPU zero-copy possible. Both same-process and cross-process subscribers on the same device are supported.

## Class diagram

Classes are grouped by origin. **ROS2 framework classes** (from `rosidl` / `rosidl_buffer`) are shown in the `rosidl` package. **`qc_buffer_backend` classes** (implemented in this repository) are shown in the `qc_buffer_backend` package.

```mermaid
classDiagram
    direction TB

    namespace rosidl {
        class BufferBackend {
            <<interface, rosidl_buffer_backend>>
            +get_backend_type() string
            +create_descriptor_with_endpoint(impl, endpoint) shared_ptr
            +from_descriptor_with_endpoint(desc, endpoint) unique_ptr
        }

        class Buffer_T {
            <<rosidl_buffer>>
            +get_backend_type() string
            +size() size_t
            +get_impl() BufferImplBase_T*
        }

        class BufferImplBase_T {
            <<interface, rosidl_buffer>>
            +get_backend_type() string
            +size() size_t
            +to_cpu() unique_ptr
            +clone() unique_ptr
        }
    }

    namespace qc_buffer_backend {
        class QcBufferBackend {
            <<qc_buffer_backend>>
            +get_backend_type() string
            +create_descriptor_with_endpoint(impl, endpoint) shared_ptr
            +from_descriptor_with_endpoint(desc, endpoint) unique_ptr
        }

        class QcBufferImpl_T {
            <<qc_buffer>>
            -size_ size_t
            -qc_buffer_ shared_ptr~QcBuffer~
            +get_backend_type() string
            +size() size_t
            +to_cpu() unique_ptr
            +clone() unique_ptr
            +get_qc_buffer() shared_ptr~QcBuffer~
        }

        class QcBuffer {
            <<qc_buffer>>
            -ptr_ uint8_t*
            -fd_ int
            -size_ size_t
            -uid_ uint64_t
            -origin_ Origin
            +data() uint8_t*
            +dmabuf_fd() int
            +size() size_t
            +uid() uint64_t
        }

        class FdBroker {
            <<qc_buffer>>
            -table_ unordered_map
            -order_ deque
            -server_fd_ int
            -socket_path_ string
            -running_ atomic~bool~
            -active_connections_ atomic~int~
            +instance()$ FdBroker&
            +register_buffer(uid, fd, size)
            +socket_path() string
            +running() bool
        }

        class RpcMemLoader {
            <<qc_buffer>>
            -available_ bool
            -lib_ void*
            +instance()$ RpcMemLoader&
            +available() bool
            +alloc(heap_id, flags, size) void*
            +free(ptr)
            +to_fd(ptr) int
        }

        class QcBufferDescriptor {
            <<qc_buffer_backend_msgs>>
            +uid uint64_t
            +pid int32_t
            +size uint64_t
            +dmabuf_size uint64_t
            +dmabuf_fd int32_t
            +vaddr uint64_t
            +use_ipc bool
            +ipc_socket_path string
            +element_type_name string
        }
    }

    BufferBackend <|.. QcBufferBackend : implements
    BufferImplBase_T <|.. QcBufferImpl_T : implements
    Buffer_T o-- BufferImplBase_T : holds pimpl
    QcBufferImpl_T *-- QcBuffer : owns
    QcBufferBackend ..> QcBufferImpl_T : creates / inspects
    QcBufferBackend ..> FdBroker : register_buffer
    QcBufferBackend ..> RpcMemLoader : available()
    QcBufferBackend ..> QcBufferDescriptor : creates / reads
    QcBuffer ..> RpcMemLoader : alloc/free (kRpcmem)
```

## Class descriptions of `qc_buffer_backend`

These classes are implemented in folder `qc_buffer_backend`. They are split across three ROS2 packages: `qc_buffer` (core library), `qc_buffer_backend` (plugin), and `qc_buffer_backend_msgs` (message definition).

#### `QcBufferBackend` — `qc_buffer_backend` package

The pluginlib plugin entry point. On the publish path, `create_descriptor_with_endpoint()` extracts the dma-buf fd from the `QcBufferImpl`, registers it with `FdBroker`, and fills a `QcBufferDescriptor`. On the subscribe path, `from_descriptor_with_endpoint()` connects to the publisher's broker socket, receives a fd via `SCM_RIGHTS`, calls `mmap()`, and constructs a new `QcBufferImpl`. Any failure returns an empty impl, triggering a DDS CPU fallback.

#### `QcBufferImpl<T>` — `qc_buffer` package

The Qualcomm implementation of `BufferImplBase<T>`, holding a `shared_ptr<QcBuffer>`. Three construction paths:
- **Default**: empty impl (size 0, no allocation).
- **By size**: publisher path — calls `rpcmem_alloc` via `QcBuffer(byte_size)`.
- **By fd**: subscriber path — calls `mmap(fd)` and wraps the result in `QcBuffer(ptr, size, fd)`.

Destruction delegates to `QcBuffer::reset()`, which calls either `rpcmem_free` or `munmap + close(fd)` depending on the `Origin` tag.

#### `QcBuffer` — `qc_buffer` package

RAII owner of a single ION / dma-buf allocation. Non-copyable. Two origin variants:
- **kRpcmem**: allocated by the publisher via `rpcmem_alloc`; `reset()` calls `rpcmem_free`.
- **kMmap**: mapped by the subscriber via `mmap(fd)`; `reset()` calls `munmap + close(fd)`.

`uid()` is a process-monotonic identifier used by `FdBroker` for fd lookup. `dmabuf_fd()` is the kernel fd passed to the HTP accelerator for zero-copy access.

#### `FdBroker` — `qc_buffer` package

A process-level singleton Unix socket server in the publisher process. `register_buffer()` calls `dup(fd)` on each newly published buffer's dma-buf fd, keeping the kernel dma-buf object alive after `QcBuffer` destructs. Any subscriber (same or different process on the same device) connects, sends the uid, and receives a fd copy via `sendmsg + SCM_RIGHTS`; the broker then closes its own dup. A rolling FIFO window (64 slots) bounds memory retention. The destructor spins on `active_connections_` to ensure all detached connection threads finish before tearing down internal state.

#### `RpcMemLoader` — `qc_buffer` package

A process-level singleton that `dlopen`s `libcdsprpc.so` and resolves `rpcmem_alloc`, `rpcmem_free`, and `rpcmem_to_fd`. When `available()` returns false (library absent, e.g. on a developer host), the entire backend falls back to the CPU path transparently.

#### `QcBufferDescriptor` — `qc_buffer_backend_msgs` package

The lightweight descriptor transmitted from publisher to subscriber over DDS. Key fields: `uid` for broker lookup, `ipc_socket_path` for the broker Unix socket, `dmabuf_size` for the `mmap()` call, and `use_ipc` to signal whether the FdBroker path is active. The `pid`, `vaddr`, and `dmabuf_fd` fields carry debug information about the publisher-side allocation.

## High-level architecture

```mermaid
flowchart LR
  Pub["Publisher node"]
  Sub["Subscriber node\n(same or different process)"]

  Pub -->|"allocate_buffer + CPU write"| Buffer["rosidl::Buffer&lt;uint8_t&gt;"]
  Buffer -->|"backend = 'qc'"| Plugin["qc_buffer_backend plugin"]
  Plugin -->|"create_descriptor"| Descriptor["QcBufferDescriptor\nuse_ipc=true, socket_path, uid, size"]
  Descriptor -->|"published over RMW"| Sub
  Plugin -.->|"rpcmem_alloc / free"| Loader["RpcMemLoader"]
  Plugin -.->|"register_buffer(uid, fd)"| Broker["FdBroker"]
  Sub -.->|"connect + SCM_RIGHTS"| Broker
  Sub -.->|"mmap(fd)"| Mem["Same physical pages"]
  Sub -.->|"get_dmabuf_fd to HTP"| HTP["HTP inference / app layer"]
```

## Publish / subscribe flow

```mermaid
sequenceDiagram
  autonumber
  participant Pub as Publisher
  participant Broker as FdBroker
  participant Sub as Subscriber
  participant App as App layer / HTP

  Pub->>Pub: buf = allocate_buffer(N), rpcmem_alloc returns VA + fd + uid
  Pub->>Pub: write data to buf via CPU pointer, no copy
  Pub->>Pub: msg.data = buf, then publish(msg)
  Pub->>Broker: register_buffer(uid, fd) — broker stores dup(fd)
  Note over Pub,Sub: descriptor = use_ipc=true, socket_path, uid, dmabuf_size, size
  Pub->>Pub: QcBuffer destructs, original fd closed, broker's dup keeps dma-buf alive
  Sub->>Broker: connect(socket_path), send(uid)
  Broker-->>Sub: SCM_RIGHTS(dup_fd) — subscriber receives new fd
  Broker->>Broker: close own dup_fd (subscriber holds its own reference)
  Sub->>Sub: mmap(fd, dmabuf_size) — maps same physical pages, zero-copy
  Sub->>App: get_dmabuf_fd(buf) returns fd
  App->>App: register fd with HTP, which reads the same memory
  Sub->>Sub: QcBufferImpl destructs: munmap + close(fd)
  Note over Sub: dma-buf refcount reaches 0 only after all subscribers close their fds
```

The subscriber performs **no memcpy**: its `mmap()`-ed address and the publisher's original `vaddr` point at the same physical pages (zero-copy).

## Fallback decision

The backend never breaks functionality; the worst case is a CPU copy. It falls back (returns `nullptr` / an empty impl, letting the serialization layer use the CPU path) when any of the following holds:

```mermaid
flowchart TB
  Create["create_descriptor_with_endpoint(impl)"] --> A{"impl is QcBufferImpl?"}
  A -->|no| Fallback["CPU fallback"]
  A -->|yes| B{"RpcMemLoader available?"}
  B -->|no| Fallback
  B -->|yes| C{"dma-buf fd valid?"}
  C -->|no| Fallback
  C -->|yes| D{"FdBroker running?"}
  D -->|no| Fallback
  D -->|yes| Emit["register uid with FdBroker, emit descriptor"]

  From["from_descriptor_with_endpoint(desc)"] --> E{"element type matches?"}
  E -->|no| Drop["empty impl, CPU"]
  E -->|yes| F{"use_ipc and socket_path present?"}
  F -->|no| Drop
  F -->|yes| G{"broker connect + uid found?"}
  G -->|no| Drop
  G -->|yes| H{"mmap(fd) succeeds?"}
  H -->|no| Drop
  H -->|yes| Reuse["QcBufferImpl via mmap, zero-copy"]
```

A **type or contract violation** (e.g. a non-`uint8_t` element type name, or an explicit qc allocation that fails) is treated as an error — allocation throws `QcError`, while `from_descriptor` logs and drops. An **environmental limitation** (no rpcmem, cross-device, broker miss) is a normal expected downgrade and only logs at WARN (once where appropriate).

## Buffer lifetime

```mermaid
flowchart TD
  A["rpcmem_alloc creates QcBuffer: VA + fd + uid"] --> B["shared_ptr held by published msg"]
  B --> C["create_descriptor: FdBroker stores dup(fd)"]
  C --> D["descriptor published over DDS"]
  B --> E["QcBuffer destructs: rpcmem_free + close(original fd)"]
  E --> F["dma-buf kept alive by broker's dup_fd"]

  D --> G["subscriber connects to broker, requests uid"]
  G --> H{"uid still in broker window?"}
  H -->|yes| I["broker sends dup_fd via SCM_RIGHTS, closes own copy"]
  H -->|no| J["MISS — CPU fallback"]

  I --> K["subscriber: mmap(fd) — same physical pages"]
  K --> L["subscriber callback runs"]
  L --> M["QcBufferImpl destructs: munmap + close(fd)"]
  M --> N["dma-buf refcount 0 when all subscribers done — memory freed"]
```

The `FdBroker` retention window (64 slots, FIFO eviction) bounds how long a buffer's fd is kept available for subscribers. At 30 Hz a window of 64 covers over 2 seconds. A subscriber that arrives after eviction sees a miss and falls back to CPU.
