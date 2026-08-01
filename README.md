# GridCompute

A distributed, browser-based compute grid. Any device that opens the URL becomes a worker node — contributing its CPU and GPU to a shared task queue coordinated by a central Node.js server over WebSockets (Socket.IO).

Originally built for 3D Mandelbulb fractal rendering, the grid now supports **generic ML inference workloads** via [LiteRT.js](https://github.com/google-ai-edge/LiteRT) — Google's high-performance `.tflite` runtime for the browser.

---

## Features

| Capability | Detail |
|---|---|
| **3D Mandelbulb rendering** | Distributed raymarching across all connected nodes |
| **WebGL2 GPU workers** | Each node renders via GLSL fragment shaders on its own GPU |
| **CPU fallback workers** | JavaScript sphere-tracer for devices without WebGL2 |
| **LiteRT.js ML inference** | Run `.tflite` models (WebGPU → XNNPACK CPU fallback) across the grid |
| **Custom JS jobs** | Submit arbitrary async JavaScript to be executed on worker nodes |
| **SHA-256 brute-force** | Built-in distributed hash cracker demo |
| **Redundancy & consensus** | Tasks can be replicated across N nodes; majority-vote resolves conflicts |
| **Fault tolerance** | Disconnected nodes have in-flight tasks automatically re-queued |
| **Real-time dashboard** | Live node status, progress, and event log via Socket.IO |

---

## Architecture

```
Browser Nodes (workers)          Node.js Server (coordinator)
┌─────────────────────┐          ┌──────────────────────────────┐
│  worker_gpu.js      │◄────────►│  Socket.IO task dispatcher   │
│  (WebGL2 GLSL)      │          │                              │
├─────────────────────┤          │  Task queue (heavyTaskQueue) │
│  worker.js          │◄────────►│  ├─ mandelbulb chunks        │
│  (CPU JavaScript)   │          │  ├─ litert inference tasks   │
├─────────────────────┤          │  └─ custom JS tasks          │
│  worker_litert.js   │◄────────►│                              │
│  (LiteRT.js WebGPU) │          │  REST API                    │
└─────────────────────┘          │  ├─ /api/models/*            │
         ▲                       │  ├─ /api/submit-litert-job   │
         │ opens URL             │  ├─ /api/submit-job          │
   Any browser/device            │  └─ /api/crack               │
                                 └──────────────────────────────┘
```

The server never computes — it only routes tasks and aggregates results.

---

## Quick Start

### Requirements

- Node.js ≥ 18
- npm

### Install & Run

```bash
git clone <repo-url>
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
| `process_task` | Task object | Assigned task for the node to execute |
| `no_more_tasks` | — | Queue is empty |
| `row_completed` | `{ yStart, chunkHeight, pixels }` | Mandelbulb chunk rendered |
| `custom_task_update` | `{ jobId, taskId, result, error }` | Consensus result for a custom/litert task |
| `progress_update` | `{ completedTasks, total }` | Overall progress |
| `update_dashboard` | Node list | Node status for UI dashboard |
| `log_event` | `string` | Human-readable activity log entry |
| `camera_updated` | `{ camera, power }` | Fractal camera state sync |

### Client → Server

| Event | Payload | Description |
|---|---|---|
| `register_node` | `{ name, cpu, gpu, gcu, concurrency, litert }` | Join the grid |
| `task_completed` | `{ jobId, taskId, result, pixels?, error? }` | Report task result |
| `set_camera` | `{ theta, phi, dist }` | Orbit Mandelbulb camera |
| `set_power` | `number` | Change Mandelbulb exponent (4/6/8/12/16) |
| `set_quality` | `number` | Change ray-march step budget (40–200) |
| `inject_tasks` | — | Reset to default Mandelbulb view |

---

## Node Registration Profile

When a browser node connects it sends `register_node` with:

```js
{
  name:        "My Device",      // display name (max 32 chars)
  cpu:         "Apple M3",       // CPU string (max 64 chars)
  gpu:         "Apple GPU",      // GPU string (max 256 chars)
  gcu:         1500,             // Grid Compute Units (arbitrary perf score)
  concurrency: 2,                // max simultaneous tasks (1–32)
  litert:      true              // declare LiteRT.js inference support
}
```

Setting `litert: true` opts the node into receiving LiteRT inference tasks. Nodes without this flag only receive Mandelbulb and custom JS tasks.

---

## Redundancy & Consensus

For jobs with `redundancy N > 1`, each logical task is replicated across N different nodes. Results are compared using a majority-vote algorithm:

- If `⌈N/2⌉` or more nodes return identical results → **consensus reached**, result accepted
- If no majority → **consensus failed**, all replicas are re-queued automatically

This protects against faulty or malicious nodes returning incorrect results.

---

## File Overview

```
GridCompute/
├── server.js          # Node.js coordinator (Express + Socket.IO)
├── index.html         # Browser UI (dashboard + canvas)
├── worker_gpu.js      # WebGL2 GPU Web Worker (Mandelbulb)
├── worker.js          # CPU JavaScript Web Worker (Mandelbulb)
├── worker_litert.js   # LiteRT.js inference Web Worker (ML tasks)
├── GridClient.js      # Client-side Socket.IO helper
├── NoSleep.min.js     # Prevent mobile display sleep during compute
├── models/            # Uploaded .tflite model files (auto-created)
├── package.json
└── README.md
```

---

## License

See [LICENSE](LICENSE).