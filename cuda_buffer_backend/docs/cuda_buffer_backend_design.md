# Design

## Introduction

`cuda_buffer_backend` is a `rosidl::Buffer<T>` storage backend that allocates
the message payload in CUDA Virtual Memory (VMM) and delivers it to a
co-located subscriber **without serialization or host copies** when the
runtime conditions allow it. When they don't, the system falls back to the
default CPU path automatically.

The optimized path is taken when the publisher and subscriber are on the
**same host, same CUDA device, same Linux user**, and at least one of the
two endpoints is intra-process or inter-process-on-the-same-host.

## High-level architecture

```mermaid
flowchart LR
  Pub["Publisher node"]
  Sub["Subscriber node"]

  Pub -->|"allocate_buffer + from_output_buffer"| Buffer["rosidl::Buffer&lt;uint8_t&gt;"]
  Buffer -->|"backend = 'cuda'"| Plugin[cuda_buffer_backend plugin]
  Plugin -->|"create_descriptor"| Descriptor["CudaBufferDescriptor<br/>(pid, block_id, socket, uid, event_handle)"]
  Descriptor -->|"published over RMW"| Sub
  Plugin -.->|"VMM allocations"| Pool[CudaMemoryPool]
  Plugin -.->|"FD passing via SCM_RIGHTS"| IPC[CudaVmmIPCManager]
  Plugin -.->|"endpoint registry in /dev/shm"| HEM[HostEndpointManager]
```

Three runtime singletons make this work:

- `CudaMemoryPool` — VMM allocator with a free-list. Each block has a
  stable `(pid, block_id)` identity for its publisher's lifetime.
- `CudaVmmIPCManager` — per-process FD dispatcher. Listens on per-block
  Unix-domain sockets and serves the exported VMM FD via `SCM_RIGHTS` when
  a subscriber connects.
- `HostEndpointManager` — host-wide shared-memory registry of ROS 2
  endpoints (pid, gid, device, uid, intra-process flag). Used by the
  backend to decide IPC capability for any discovered remote endpoint.

## Publish / subscribe flow

```mermaid
sequenceDiagram
  autonumber
  participant Pub as Publisher
  participant Pool as CudaMemoryPool
  participant Sub as Subscriber
  participant Cache as Subscriber import cache

  Pub->>Pool: allocate(byte_count)
  Pool-->>Pub: VmmBlock (va, exported_fd, ipc_meta, uid)
  Pub->>Pub: write data to GPU memory
  Pub-->>Sub: publish CudaBufferDescriptor

  Sub->>Cache: lookup (pid, block_id)
  alt cache hit
    Cache-->>Sub: existing import (no FD round-trip)
  else cache miss
    Sub->>Pub: connect(socket_path)
    Pub->>Sub: sendmsg with SCM_RIGHTS (FD)
    Sub->>Sub: cuMemImportFromShareableHandle + map
    Sub->>Sub: mmap IPCMetadata
    Sub->>Cache: insert
  end

  Sub->>Sub: ipc_meta->refcount.fetch_add(1)
  Sub-->>Sub: deliver CUDA-backed Buffer to user

  Note over Sub: on Buffer destroy
  Sub->>Sub: ipc_meta->refcount.fetch_sub(1)
  Note over Pub: pool may now recycle the block
```

The first transmission of a `(pid, block_id)` pair does the full FD
handshake; every subsequent transmission of the same pair is a cache hit.

## IPC capability decision

When a remote endpoint is discovered, the backend decides whether to use
the optimized path or fall back to CPU:

```mermaid
flowchart TB
  Discover["on_discovering_endpoint(remote)"] --> A{remote supports 'cuda'?}
  A -->|no| Fallback[CPU fallback]
  A -->|yes| B{pool is IPC capable?}
  B -->|no| Fallback
  B -->|yes| C{locality of remote endpoint}
  C -->|INTRA_PROCESS| Use[use CUDA IPC]
  C -->|INTER_PROCESS_SAME_HOST| D{same device AND same uid?}
  D -->|yes| Use
  D -->|no| Fallback
  C -->|other| Fallback
```

Locality comes from `HostEndpointManager`: every backend's
`on_creating_endpoint` publishes its (pid, gid, device, uid, intra-process
flag) into the host-wide shared registry, and `on_discovering_endpoint`
queries it for the remote gid.

## CUDA buffer lifetime

```mermaid
flowchart TD
  A["VMM block in pool"] --> B["Publisher CudaBuffer"]
  B --> C["WriteHandle records write event"]
  C --> D["Descriptor published"]

  D --> E["Subscriber imports block"]
  E --> F{"Descriptor UID matches shared-memory UID?"}
  F -->|"yes"| G["IPC refcount increments"]
  F -->|"no"| S["stale buffer, discard"]
  G --> H["Local CudaBuffer and ReadHandle"]
  H --> I["Local recycler waits read events"]
  I --> J["IPC refcount decrements"]

  J --> K{"Safe to recycle?"}
  K -->|"refcount zero and grace elapsed"| A
  K -->|"otherwise"| K
```

A CUDA buffer is a pooled VMM block wrapped in `CudaBuffer`. When the block is
imported in another process, the import increments the shared-memory IPC
refcount, while local `shared_ptr` ownership keeps the imported `CudaBuffer`
alive inside that process. When the last local owner is released, the recycler
waits for recorded CUDA read events before decrementing the IPC refcount.
The publisher may recycle the block only after the IPC refcount returns to
zero and a short grace window has elapsed. If a late subscriber still races
with reuse, it detects stale data on first access by comparing the descriptor
UID with the UID stored in shared memory.
