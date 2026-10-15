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

  D --> E["Subscriber acquires open UID generation"]
  E -->|"accepted"| G["IPC refcount increments"]
  E -->|"closed or stale"| S["discard descriptor before event import"]
  G --> H["Import event; local CudaBuffer and ReadHandle"]
  H --> I["Local recycler waits read events"]
  I --> J["Destroy imported event; decrement IPC refcount"]

  D --> P["Pool retains exported event with returned block"]
  P --> K{"Grace elapsed?"}
  K -->|"yes"| L["Close generation by setting shared UID to zero"]
  K -->|"no"| K
  L --> M{"IPC refcount zero and producer event complete?"}
  M -->|"yes"| N["Destroy exported event and reuse block with a new UID"]
  M -->|"no"| M
  N --> A
```

A CUDA buffer is a pooled VMM block wrapped in `CudaBuffer`. Before importing a
block, a subscriber atomically acquires its published UID generation. The
acquisition increments the shared-memory IPC refcount only while that
generation remains open. Local `shared_ptr` ownership then keeps the imported
`CudaBuffer` alive. When the last local owner is released, the recycler waits
for recorded CUDA read events, destroys the imported event handle, and
decrements the IPC refcount.

When the publisher returns a block, the pool retains any exported write event
with that block. After the grace window, the pool closes the generation to new
imports by setting the shared UID to zero. It destroys the exported event and
reuses the block only after the producer event has completed and the IPC
refcount is zero. Reuse installs the next UID before reopening the generation.
A delayed subscriber therefore observes a closed or different UID and discards
the descriptor before calling `cudaIpcOpenEventHandle`.
