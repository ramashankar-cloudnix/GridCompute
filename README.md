# GridCompute

A distributed, browser-based compute grid. Any device that opens the URL becomes a worker node — contributing its CPU and GPU to a shared task queue coordinated by a central Node.js server over WebSockets (Socket.IO).

Originally built for 3D Mandelbulb fractal rendering, the grid now supports **generic ML inference workloads** via [LiteRT.js](https://github.com/google-ai-edge/LiteRT) — Google's high-performance `.tflite` runtime for the browser.

---

## Features

| Capability | Detail |
|---|---|
| **~90% System Resource Saturation** | Dynamically scales worker pool to ~90% of host capacity with configurable target slider (20%–100%) |
| **Collective dGPU + iGPU Computing** | WebGPU dual-adapter initialization harnesses discrete and integrated GPUs simultaneously |
| **Legacy 5–7yr Hardware Fallback** | Universal WebGL2 GPGPU + multi-core CPU workers run reliably on 2019+ devices |
| **2026 GFLOPS Benchmark Engine** | Tests FP32, FP16 (`shader-f16`), and INT8 quantized GOPS for edge AI workloads |
| **Background Tab Persistence** | Screen Wake Lock + silent Web Audio loop + Web Locks lease keep workers active even when minimized |
| **Unified Task Push REST API** | Push distributed GEMM, Monte Carlo, ML inference, and custom kernels via `/api/v1/jobs` |
| **Standalone Demo Client App** | Interactive demo web app (`/demo`) and CLI client (`demo_cli.js`) to submit and monitor tasks |
| **3D Mandelbulb rendering** | Distributed raymarching across all connected nodes |
| **LiteRT.js ML inference** | Run `.tflite` models (WebGPU → XNNPACK CPU fallback) across the grid |
| **Redundancy & consensus** | Tasks can be replicated across N nodes; majority-vote resolves conflicts |
| **Fault tolerance** | Disconnected nodes have in-flight tasks automatically re-queued |

---

## Architecture

```
Browser Nodes (workers)                 Node.js Server (coordinator)
┌─────────────────────────────────┐     ┌─────────────────────────────────────────┐
│ worker_webgpu.js (dGPU / iGPU)  │◄───►│  Socket.IO task dispatcher              │
├─────────────────────────────────┤     │  ├─ Double-buffered prefetch queue      │
│ worker_gpu.js (WebGL2 fallback) │◄───►│  └─ GFLOPS-weighted load balancing      │
├─────────────────────────────────┤     │                                         │
│ worker.js (Multi-core CPU)      │◄───►│  Task Queue                             │
├─────────────────────────────────┤     │  ├─ GEMM matrix multiplication tiles    │
│ worker_litert.js (ML inference) │◄───►│  ├─ Monte Carlo simulation chunks       │
└─────────────────────────────────┘     │  ├─ Mandelbulb render chunks            │
                 ▲                      │  └─ Custom JS / Wasm kernels            │
                 │                      │                                         │
        Background Engine               │  Unified REST API (v1)                  │
        ├─ Screen Wake Lock             │  ├─ POST /api/v1/jobs                   │
        ├─ Silent Web Audio             │  ├─ GET  /api/v1/jobs/:jobId            │
        └─ Web Locks Lease              │  ├─ GET  /api/v1/jobs/:jobId/results    │
                                        │  ├─ GET  /api/v1/grid/stats             │
                                        │  └─ GET  /api/v1/grid/nodes             │
                                        └─────────────────────────────────────────┘
                                                             ▲
                                                             │ REST / Socket.IO
                                                ┌────────────────────────────┐
                                                │ Demo App (/demo) & CLI SDK │
                                                └────────────────────────────┘
```

The server never computes — it only routes tasks and aggregates results.

---

## Quick Start

### Requirements

- Node.js ≥ 18
- npm

### Install & Run

```bash
git clone https://github.com/ramashankar-cloudnix/GridCompute.git
cd GridCompute
npm install
npm start
```

Server starts at **http://localhost:8080**

Open that URL on any device to join as a worker node. Open it on multiple devices/tabs to scale the grid.

---

## Worker Types

### `worker_gpu.js` — WebGL2 GPU Worker *(default for Mandelbulb)*
Renders Mandelbulb chunks using GLSL fragment shaders via an `OffscreenCanvas`. Achieves the highest throughput on desktop GPUs.

### `worker.js` — CPU JavaScript Worker *(Mandelbulb fallback)*
Pure JavaScript sphere-tracer. Used when WebGL2 is unavailable. Capped at 60 ray steps to protect the main thread budget.

### `worker_litert.js` — LiteRT.js Inference Worker *(ML workloads)*
Runs `.tflite` model inference distributed across browser nodes.

- **Primary backend:** WebGPU via LiteRT.js ML Drift engine (5–60× faster than CPU for large models)
- **Fallback backend:** XNNPACK (CPU via WebAssembly) — works in any modern browser
- **Model cache:** compiled models are cached per-worker so repeated tasks on the same model are near-instant
- **Node registration:** nodes using this worker include `{ litert: true }` in their profile so the server routes inference tasks only to capable nodes

---

## ML Inference with LiteRT.js

### 1 — Install dependencies

```bash
npm install
```

This installs `multer` (model uploads) and `@litertjs/core` (provides the Wasm runtime assets served at `/litert-wasm/` and `/litert-js/`).

> If `@litertjs/core` is not yet published to npm, run `npm install multer` and the worker will automatically fall back to loading LiteRT.js from the `unpkg.com` CDN.

### 2 — Upload a `.tflite` model

```bash
curl -F "model=@mobilenet_v1.tflite" http://localhost:8080/api/models/upload
```

Response:
```json
{
  "name": "mobilenet_v1.tflite",
  "url": "/models/mobilenet_v1.tflite",
  "sizeBytes": 4276033
}
```

### 3 — Submit an inference job

```bash
curl -X POST http://localhost:8080/api/submit-litert-job \
  -H "Content-Type: application/json" \
  -d '{
    "modelName": "mobilenet_v1.tflite",
    "redundancy": 1,
    "tasks": [
      {
        "taskId": "img_batch_0",
        "inputs": [
          {
            "data": [0.12, 0.45, 0.88, ...],
            "shape": [1, 224, 224, 3],
            "dtype": "float32"
          }
        ]
      }
    ]
  }'
```

Response:
```json
{
  "jobId": "litert_a3f9bc12",
  "status": "queued",
  "totalTasks": 1,
  "modelUrl": "/models/mobilenet_v1.tflite",
  "redundancy": 1,
  "liteRtNodesOnline": 2
}
```

### 4 — Poll for results

```bash
curl http://localhost:8080/api/job-status/litert_a3f9bc12
```

Each completed task result contains:
```json
{
  "outputs": [
    { "data": [0.003, 0.98, ...], "shape": [1, 1001], "dtype": "float32" }
  ],
  "inferenceMs": 12.4,
  "modelUrl": "/models/mobilenet_v1.tflite",
  "accelerator": "webgpu"
}
```

### Input tensor format

| Field | Type | Description |
|---|---|---|
| `data` | `number[]` | Flat array of tensor values |
| `shape` | `number[]` | Tensor dimensions, e.g. `[1, 224, 224, 3]` |
| `dtype` | `string` | `"float32"` (default) · `"int32"` · `"uint8"` |

---

## REST API Reference

### Models

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/models` | List all uploaded `.tflite` models |
| `POST` | `/api/models/upload` | Upload a `.tflite` model (`multipart/form-data`, field: `model`) |
| `DELETE` | `/api/models/:name` | Delete a model by filename |

### Jobs

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/submit-litert-job` | Submit a distributed LiteRT inference job |
| `POST` | `/api/submit-job` | Submit a generic custom JavaScript job |
| `POST` | `/api/crack` | Submit a distributed SHA-256 brute-force job |
| `GET` | `/api/job-status/:jobId` | Poll job progress and results |

### Static Assets (served automatically)

| Path | Serves |
|---|---|
| `/litert-wasm/` | `@litertjs/core` Wasm binaries (Wasm runtime for browser nodes) |
| `/litert-js/` | `@litertjs/core` JS bundle (UMD, imported by `worker_litert.js`) |
| `/models/` | Uploaded `.tflite` model files |
| `/worker_litert.js` | LiteRT inference Web Worker script |
| `/worker_gpu.js` | WebGL2 GPU Web Worker script |
| `/worker.js` | CPU JavaScript Web Worker script |

---

## Socket.IO Events

### Server → Client

| Event | Payload | Description |
|---|---|---|
| `process_task` | Task object | Assigned computing task for the node (GEMM, Monte Carlo, Custom JS, LiteRT) |
| `no_more_tasks` | — | Task queue is empty (workers idle/standby) |
| `custom_task_update` | `{ jobId, taskId, result, error }` | Consensus result for a custom/litert task |
| `gemm_tile_completed` | `{ jobId, taskId, completedCount, totalCount }` | Real-time GEMM tile resolution event |
| `progress_update` | `{ completedTasks, total }` | Overall cluster job progress |
| `update_dashboard` | Node list | Live node telemetry, hardware specs & GFLOPS rating |
| `log_event` | `string` | Human-readable activity log entry |

### Client → Server

| Event | Payload | Description |
|---|---|---|
| `register_node` | `{ name, cpu, gpu, dgpu, igpu, gcu, benchmarks, resourceTarget, backends, concurrency }` | Join the grid with full hardware profile |
| `task_completed` | `{ jobId, taskId, backend, result, error? }` | Report finished chunk/tile result |

---

## Node Registration Profile

When a browser node connects and completes its benchmark, it sends `register_node` with:

```js
{
  name:           "💻 Windows PC",      // display name
  cpu:            "16 Logical Cores",   // CPU core count
  gpu:            "NVIDIA GeForce RTX", // GPU string
  dgpu:           "NVIDIA GeForce RTX", // Discrete GPU (if present)
  igpu:           "Intel UHD Graphics", // Integrated GPU (if present)
  gcu:            4250.5,               // Total cluster GFLOPS score
  benchmarks: {                         // Comprehensive 2026 GFLOPS rating
    fp32_gflops:    2100.2,
    fp16_gflops:    1850.0,
    fp16_supported: true,
    int8_gops:      2900.5,
    cpu_gflops:     125.4,
    total_gflops:   4250.5
  },
  resourceTarget: 90,                   // Target system resource utilization (~90%)
  backends:       ["webgpu-high-performance", "webgpu-low-power", "cpu"],
  concurrency:    16                    // Parallel pipeline count
}
```

## Redundancy & Consensus

For jobs with `redundancy N > 1`, each logical task is replicated across N different nodes. Results are compared using a majority-vote algorithm:

- If `⌈N/2⌉` or more nodes return identical results → **consensus reached**, result accepted
- If no majority → **consensus failed**, all replicas are re-queued automatically

This protects against faulty or malicious nodes returning incorrect results.

---

## Security & Deployment Notice

> [!WARNING]
> **Distributed Script Execution Threat Model**
> `GridCompute` distributes computation tasks across connected browser nodes. By design, generic compute tasks (`type: 'custom'`) execute JavaScript payloads inside browser Web Workers (`worker.js`).
> 
> - **Local / Intranet Use**: The default configuration (`cors: '*'`, unauthenticated `/api/v1/jobs`) is designed for development, research clusters, LAN compute pools, and controlled laboratory environments.
> - **Public Network Exposure**: If exposing the coordinator server to the public internet, **you must place it behind a reverse proxy (e.g. Nginx, Caddy, Cloudflare) with an authentication layer** (API keys, OAuth, or HTTP Basic Auth) to prevent untrusted actors from dispatching arbitrary scripts to connected worker browsers.
> - **Sandbox Scope**: Tasks run inside isolated Web Workers without direct DOM access; however, workers have network access (`fetch`, `importScripts`). Ensure task scripts originate from trusted submitters.

---

## File Overview

```
GridCompute/
├── server.js               # Node.js coordinator (Express, Socket.IO, task dispatcher)
├── index.html              # Worker dashboard & live GFLOPS hardware telemetry UI
├── demo_app.html           # Interactive consumer demo application (/demo)
├── demo_cli.js             # Node.js CLI tool demonstrating matrix & Monte Carlo runs
├── GridClient.js           # Client SDK for programmatic job submission & polling
├── worker_webgpu.js        # WebGPU worker (Dual dGPU + iGPU FP32/FP16/INT8 pipelines)
├── worker_gpu.js           # WebGL2 GPGPU worker (Universal fallback for 2019+ hardware)
├── worker.js               # Multi-core CPU Web Worker (Pure JS compute engine)
├── worker_litert.js        # LiteRT.js inference worker (Distributed .tflite ML models)
├── NoSleep.min.js          # Keeps mobile/desktop screens and tabs awake (MIT License)
├── examples/               # Standalone real-world client scripts (e.g. Distributed OCR)
├── models/                 # Uploaded .tflite model repository
├── package.json            # NPM dependencies and project metadata
├── LICENSE                 # GNU Affero General Public License v3 (AGPL-3.0)
└── README.md               # Architecture documentation and API reference
```

---

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0) — see the [LICENSE](LICENSE) file for details.
Third-party bundled libraries (`NoSleep.min.js`) are subject to their respective licenses (MIT).