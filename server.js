const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');
const fs   = require('fs');
const multer = require('multer');

const app = express();
// Support large JSON payloads (e.g. GEMM matrices and tensor arrays up to 100MB)
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// Ensure Express never returns HTML error pages for API requests
app.use((err, req, res, next) => {
    if (err) {
        console.error('⚠️ Express parsing error:', err.message);
        return res.status(err.status || 400).json({ error: err.message || 'Invalid request body' });
    }
    next();
});
const server = http.createServer(app);
const io = new Server(server, {
    cors: { origin: '*' },
    pingTimeout: 20000,
    pingInterval: 10000
});

let networkNodes = {};
const CANVAS_WIDTH  = 1280;
const CANVAS_HEIGHT = 360;
const CHUNK_SIZE    = 4;   // Smaller chunks to prevent GPU TDR timeouts
const MAX_LOG_EVENTS = 200;

// ─── Models directory (for uploaded .tflite files) ────────────────────────────
const MODELS_DIR = path.join(__dirname, 'models');
if (!fs.existsSync(MODELS_DIR)) fs.mkdirSync(MODELS_DIR, { recursive: true });

// ─── Multer: save uploaded .tflite models to /models/ ────────────────────────
const modelStorage = multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, MODELS_DIR),
    filename:    (_req,  file, cb) => {
        // Sanitise filename — only alphanumerics, dots, dashes, underscores
        const safe = path.basename(file.originalname).replace(/[^a-zA-Z0-9._-]/g, '_');
        cb(null, safe);
    }
});
const uploadModel = multer({
    storage:    modelStorage,
    fileFilter: (_req, file, cb) => {
        if (!file.originalname.match(/\.tflite$/i))
            return cb(new Error('Only .tflite model files are accepted'));
        cb(null, true);
    },
    limits: { fileSize: 50 * 1024 * 1024 }  // 50 MB cap per model
});

// ─── 3D Camera State (spherical coordinates) ──────────────────────────────────
let camera = {
    theta: 0.5,   // azimuth angle in radians
    phi:   0.3,   // elevation angle in radians (clamped ±1.4 to avoid gimbal)
    dist:  2.5    // distance from origin
};

let fractalPower = 8;  // Mandelbulb power: 4, 6, 8, 12, 16
let maxSteps     = 80; // ray march steps per pixel

function generateTasks() {
    const tasks = [];
    for (let y = 0; y < CANVAS_HEIGHT; y += CHUNK_SIZE) {
        const h = Math.min(CHUNK_SIZE, CANVAS_HEIGHT - y);
        tasks.push({
            jobId:       'mandelbulb',
            taskId:      'yStart_' + y,
            type:        'mandelbulb',
            yStart:      y,
            chunkHeight: h,
            width:       CANVAS_WIDTH,
            height:      CANVAS_HEIGHT,
            // Camera parameters
            camTheta:    camera.theta,
            camPhi:      camera.phi,
            camDist:     camera.dist,
            // Fractal parameters
            power:       fractalPower,
            maxSteps:    maxSteps
        });
    }
    return tasks;
}

let heavyTaskQueue  = [];
let totalTasksCount = 0;
let completedTasks  = 0;

// ─── Static file serving ──────────────────────────────────────────────────────
app.get('/',             (req, res) => res.sendFile(__dirname + '/index.html'));
app.get('/worker.js',    (req, res) => res.sendFile(__dirname + '/worker.js'));
app.get('/worker_gpu.js',(req, res) => res.sendFile(__dirname + '/worker_gpu.js'));
app.get('/worker_webgpu.js', (req, res) => res.sendFile(__dirname + '/worker_webgpu.js'));
app.get('/NoSleep.min.js',(req,res) => res.sendFile(__dirname + '/NoSleep.min.js'));
app.get('/worker_litert.js', (req, res) => res.sendFile(__dirname + '/worker_litert.js'));
app.get('/demo',         (req, res) => res.sendFile(__dirname + '/demo_app.html'));
app.get('/demo_app.html',(req, res) => res.sendFile(__dirname + '/demo_app.html'));

// ─── LiteRT.js — serve wasm runtime assets to browser nodes ──────────────────
// Browser workers call loadLiteRt('/litert-wasm/') which fetches these binaries.
const LITERT_WASM_DIR = path.join(__dirname, 'node_modules', '@litertjs', 'core', 'wasm');
const LITERT_JS_DIR   = path.join(__dirname, 'node_modules', '@litertjs', 'core', 'dist');
if (fs.existsSync(LITERT_WASM_DIR)) {
    app.use('/litert-wasm', express.static(LITERT_WASM_DIR));
    console.log('🧠 LiteRT wasm runtime served at /litert-wasm/');
} else {
    console.warn('⚠️  @litertjs/core wasm not found — run: npm install');
}
if (fs.existsSync(LITERT_JS_DIR)) {
    app.use('/litert-js', express.static(LITERT_JS_DIR));
    console.log('🧠 LiteRT JS bundle served at /litert-js/');
}

// ─── Uploaded model files — served to browser workers ─────────────────────────
app.use('/models', express.static(MODELS_DIR));

// ─── Distributed Compute API & Jobs State ─────────────────────────────────────
const jobs = {};

// ─── Unified Distributed Task Push & Grid Telemetry API (v1) ─────────────────

// API: Get online cluster nodes with hardware & GFLOPS telemetry
app.get('/api/v1/grid/nodes', (_req, res) => {
    res.json({
        nodes: Object.values(networkNodes),
        totalCount: Object.keys(networkNodes).length
    });
});

// API: Get cluster-wide stats and aggregate GFLOPS (FP32, FP16, INT8)
app.get('/api/v1/grid/stats', (_req, res) => {
    const nodes = Object.values(networkNodes);
    let totalFP32 = 0, totalFP16 = 0, totalINT8 = 0, totalGflops = 0, totalWorkers = 0;

    nodes.forEach(n => {
        totalWorkers += (n.concurrency || 1);
        if (n.benchmarks) {
            totalFP32 += (n.benchmarks.fp32_gflops || 0);
            totalFP16 += (n.benchmarks.fp16_gflops || 0);
            totalINT8 += (n.benchmarks.int8_gops || 0);
            totalGflops += (n.benchmarks.total_gflops || 0);
        } else {
            totalGflops += (n.gcu || 0);
        }
    });

    res.json({
        onlineNodes: nodes.length,
        totalWorkers,
        totalFP32Gflops: parseFloat(totalFP32.toFixed(2)),
        totalFP16Gflops: parseFloat(totalFP16.toFixed(2)),
        totalINT8Gops: parseFloat(totalINT8.toFixed(2)),
        totalGflops: parseFloat(totalGflops.toFixed(2)),
        queuedTasks: heavyTaskQueue.length,
        completedTasks,
        activeJobs: Object.keys(jobs).length
    });
});

// API: Check status of any job
app.get('/api/job-status/:jobId', (req, res) => {
    const job = jobs[req.params.jobId];
    if (!job) return res.status(404).json({ error: 'Job not found' });
    res.json({
        jobId: job.id,
        type: job.type,
        status: job.status,
        progress: {
            completed: job.completedCount,
            total: job.totalCount,
            percent: job.totalCount > 0 ? Math.round((job.completedCount / job.totalCount) * 100) : 100
        },
        results: job.results,
        assembledResult: job.assembledResult || null,
        createdAt: new Date(job.createdAt).toISOString(),
        completedAt: job.completedAt ? new Date(job.completedAt).toISOString() : null
    });
});
app.get('/api/v1/jobs/:jobId', (req, res) => {
    const job = jobs[req.params.jobId];
    if (!job) return res.status(404).json({ error: 'Job not found' });
    res.json({
        jobId: job.id,
        type: job.type,
        status: job.status,
        progress: {
            completed: job.completedCount,
            total: job.totalCount,
            percent: job.totalCount > 0 ? Math.round((job.completedCount / job.totalCount) * 100) : 100
        },
        resultsCount: Object.keys(job.results).length,
        createdAt: new Date(job.createdAt).toISOString(),
        completedAt: job.completedAt ? new Date(job.completedAt).toISOString() : null
    });
});

// API: Get assembled results of a completed job
app.get('/api/v1/jobs/:jobId/results', (req, res) => {
    const job = jobs[req.params.jobId];
    if (!job) return res.status(404).json({ error: 'Job not found' });
    res.json({
        jobId: job.id,
        status: job.status,
        type: job.type,
        completedCount: job.completedCount,
        totalCount: job.totalCount,
        assembledResult: job.assembledResult || null,
        rawResults: job.results
    });
});

// API: Cancel / delete a job
app.delete('/api/v1/jobs/:jobId', (req, res) => {
    const jobId = req.params.jobId;
    if (!jobs[jobId]) return res.status(404).json({ error: 'Job not found' });
    
    // Purge queued tasks belonging to this job
    const beforeCount = heavyTaskQueue.length;
    heavyTaskQueue = heavyTaskQueue.filter(t => t.jobId !== jobId);
    const removed = beforeCount - heavyTaskQueue.length;
    jobs[jobId].status = 'cancelled';
    io.emit('log_event', `🛑 Job ${jobId} cancelled (${removed} queued chunks purged).`);
    res.json({ cancelled: jobId, purgedTasks: removed });
});

// API: Unified Task Push (v1) — Supports GEMM, Monte Carlo, Custom JS, and Array compute
app.post('/api/v1/jobs', (req, res) => {
    const { type, matrixA, matrixB, M, K, N, tileRows, totalSamples, chunkSize, script, imports, tasks, redundancy, requiredBackend } = req.body;
    const jobRedundancy = Math.max(1, Math.min(parseInt(redundancy, 10) || 1, 5));
    const jobId = 'job_' + Math.random().toString(36).substring(2, 10);

    let queuedTasks = [];
    let jobAssembledResult = null;
    let jobParams = {};

    // 1. GEMM Matrix Multiplication Workload
    if (type === 'gemm') {
        const rowsM = parseInt(M, 10);
        const colsK = parseInt(K, 10);
        const colsN = parseInt(N, 10);
        if (!Array.isArray(matrixA) || !Array.isArray(matrixB) || !rowsM || !colsK || !colsN) {
            return res.status(400).json({ error: 'Invalid GEMM request: matrixA, matrixB, M, K, N required' });
        }
        const tRows = Math.max(1, parseInt(tileRows, 10) || 32);
        jobParams = { M: rowsM, K: colsK, N: colsN, tileRows: tRows };
        jobAssembledResult = {
            M: rowsM,
            K: colsK,
            N: colsN,
            matrixC: new Array(rowsM * colsN).fill(0),
            achievedGflops: 0,
            durationSeconds: 0
        };

        for (let r = 0; r < rowsM; r += tRows) {
            const rowCount = Math.min(tRows, rowsM - r);
            const subA = matrixA.slice(r * colsK, (r + rowCount) * colsK);
            for (let rep = 0; rep < jobRedundancy; rep++) {
                queuedTasks.push({
                    jobId,
                    taskId: `gemm_tile_${r}`,
                    type: 'gemm',
                    requiredBackend: requiredBackend || null,
                    params: {
                        tileRowStart: r,
                        tileRowCount: rowCount,
                        K: colsK,
                        N: colsN,
                        subA,
                        matrixB
                    },
                    replicaIndex: rep
                });
            }
        }
    }
    // 2. Monte Carlo Simulation Workload (e.g. Distributed Pi Estimation)
    else if (type === 'monte_carlo') {
        const samples = parseInt(totalSamples, 10) || 5000000;
        const cSize = Math.max(10000, parseInt(chunkSize, 10) || 500000);
        const numChunks = Math.ceil(samples / cSize);
        jobParams = { totalSamples: samples, chunkSize: cSize };
        jobAssembledResult = { totalSamples: samples, totalHits: 0, estimatedPi: 0 };

        const mcScript = `
            const count = params.samples;
            let hits = 0;
            for (let i = 0; i < count; i++) {
                const x = Math.random();
                const y = Math.random();
                if (x * x + y * y <= 1.0) hits++;
            }
            return { hits, samples: count, localPi: (4 * hits) / count };
        `;

        for (let i = 0; i < numChunks; i++) {
            const thisChunk = Math.min(cSize, samples - i * cSize);
            for (let rep = 0; rep < jobRedundancy; rep++) {
                queuedTasks.push({
                    jobId,
                    taskId: `mc_chunk_${i}`,
                    type: 'custom',
                    script: mcScript,
                    params: { chunkIndex: i, samples: thisChunk },
                    replicaIndex: rep
                });
            }
        }
    }
    // 3. Generic Custom Tasks or raw tasks array
    else {
        if (!tasks || !Array.isArray(tasks)) {
            return res.status(400).json({ error: 'Tasks array or valid compute type required' });
        }
        const jobImports = Array.isArray(imports) ? imports : [];
        tasks.forEach(t => {
            for (let rep = 0; rep < jobRedundancy; rep++) {
                queuedTasks.push({
                    jobId,
                    taskId: t.taskId,
                    type: type || 'custom',
                    script: t.script || script || '',
                    imports: t.imports || jobImports,
                    params: t.params,
                    requiredBackend: t.requiredBackend || requiredBackend || null,
                    replicaIndex: rep
                });
            }
        });
    }

    if (queuedTasks.length === 0) {
        return res.status(400).json({ error: 'No tasks generated' });
    }

    const logicalCount = queuedTasks.length / jobRedundancy;
    const newJob = {
        id: jobId,
        type: type || 'custom',
        status: 'queued',
        params: jobParams,
        redundancy: jobRedundancy,
        tasks: queuedTasks,
        results: {},
        assembledResult: jobAssembledResult,
        completedCount: 0,
        totalCount: logicalCount,
        createdAt: Date.now()
    };

    jobs[jobId] = newJob;
    heavyTaskQueue.push(...queuedTasks);
    totalTasksCount += queuedTasks.length;

    io.emit('log_event', `📥 [API] Job ${jobId} (${newJob.type}) queued: ${logicalCount} tasks | Redundancy N=${jobRedundancy}`);
    io.emit('progress_update', { completedTasks, total: totalTasksCount });
    wakeAllNodes();

    res.status(202).json({
        jobId,
        type: newJob.type,
        status: 'queued',
        totalTasks: logicalCount,
        queuedTasks: queuedTasks.length,
        redundancy: jobRedundancy
    });
});

// ─── LiteRT Model Management API ──────────────────────────────────────────────

// API: List all uploaded .tflite models
app.get('/api/models', (_req, res) => {
    const files = fs.existsSync(MODELS_DIR)
        ? fs.readdirSync(MODELS_DIR).filter(f => f.endsWith('.tflite'))
        : [];
    const models = files.map(f => {
        const stat = fs.statSync(path.join(MODELS_DIR, f));
        return {
            name:       f,
            url:        `/models/${f}`,
            sizeBytes:  stat.size,
            uploadedAt: stat.mtime
        };
    });
    res.json({ models, count: models.length });
});

// API: Upload a .tflite model file (multipart/form-data, field name: "model")
app.post('/api/models/upload', (req, res, next) => {
    uploadModel.single('model')(req, res, (err) => {
        if (err) return res.status(400).json({ error: err.message });
        if (!req.file) return res.status(400).json({ error: 'No model file provided. Use field name "model".' });
        const info = {
            name:      req.file.filename,
            url:       `/models/${req.file.filename}`,
            sizeBytes: req.file.size
        };
        io.emit('log_event', `📦 Model uploaded: ${info.name} (${(info.sizeBytes / 1024).toFixed(1)} KB)`);
        console.log(`📦 Model stored: ${info.name}`);
        res.json(info);
    });
});

// API: Delete a .tflite model
app.delete('/api/models/:name', (req, res) => {
    const name = path.basename(req.params.name);
    const filePath = path.join(MODELS_DIR, name);
    if (!fs.existsSync(filePath)) return res.status(404).json({ error: 'Model not found' });
    fs.unlinkSync(filePath);
    io.emit('log_event', `🗑️ Model deleted: ${name}`);
    res.json({ deleted: name });
});

// API: Submit a custom generic JavaScript task job
app.post('/api/submit-job', (req, res) => {
    const { type, script, imports, tasks, redundancy } = req.body;
    if (!tasks || !Array.isArray(tasks)) {
        return res.status(400).json({ error: 'Invalid tasks array' });
    }

    const jobRedundancy = Math.max(1, Math.min(parseInt(redundancy, 10) || 1, 5)); // cap at 5
    const jobId = 'job_' + Math.random().toString(36).substring(2, 10);
    const jobImports = Array.isArray(imports) ? imports : [];
    
    const queuedTasks = [];
    tasks.forEach(t => {
        for (let r = 0; r < jobRedundancy; r++) {
            queuedTasks.push({
                jobId,
                taskId: t.taskId,
                type: type || 'custom',
                script: script || '',
                imports: t.imports || jobImports,
                params: t.params,
                replicaIndex: r
            });
        }
    });

    const newJob = {
        id: jobId,
        type: type || 'custom',
        status: 'queued',
        script: script || '',
        imports: jobImports,
        redundancy: jobRedundancy,
        tasks: queuedTasks,
        results: {},
        completedCount: 0,
        totalCount: tasks.length, // total logical tasks
        createdAt: Date.now()
    };

    jobs[jobId] = newJob;
    heavyTaskQueue.push(...queuedTasks);
    totalTasksCount += queuedTasks.length;

    io.emit('log_event', `📥 Custom Job ${jobId} submitted: ${tasks.length} logical tasks with redundancy N=${jobRedundancy} (${queuedTasks.length} queued).`);
    io.emit('progress_update', { completedTasks, total: totalTasksCount });
    wakeAllNodes();

    res.status(202).json({ jobId, status: 'queued', totalTasks: tasks.length, redundancy: jobRedundancy });
});



// ─── LiteRT Inference Job API ──────────────────────────────────────────────────

// API: Submit a distributed ML inference job (runs on LiteRT-capable browser nodes)
//
// Request body:
//   modelName  {string}  — filename of a previously uploaded .tflite model
//   tasks      {Array}   — [{ taskId, inputs: [{ data, shape, dtype }] }]
//   redundancy {number}  — optional fault-tolerance N (default 1, max 5)
//
// Each task's `inputs` is an array of tensors (matching model input signatures):
//   data  — flat number array (will be cast to Float32Array / Int32Array / Uint8Array)
//   shape — dimension array, e.g. [1, 224, 224, 3]
//   dtype — 'float32' | 'int32' | 'uint8'  (default: 'float32')
//
app.post('/api/submit-litert-job', (req, res) => {
    const { modelName, tasks, redundancy } = req.body;

    if (!modelName || typeof modelName !== 'string') {
        return res.status(400).json({ error: 'Required: modelName (string) — filename of an uploaded .tflite model' });
    }
    if (!tasks || !Array.isArray(tasks) || tasks.length === 0) {
        return res.status(400).json({ error: 'Required: tasks (non-empty array)' });
    }

    const modelFile = path.basename(modelName);
    const modelPath = path.join(MODELS_DIR, modelFile);
    if (!fs.existsSync(modelPath)) {
        return res.status(404).json({
            error: `Model "${modelFile}" not found. Upload it first via POST /api/models/upload`
        });
    }

    const modelUrl      = `/models/${modelFile}`;
    const jobRedundancy = Math.max(1, Math.min(parseInt(redundancy, 10) || 1, 5));
    const jobId         = 'litert_' + Math.random().toString(36).substring(2, 10);

    const queuedTasks = [];
    tasks.forEach(t => {
        if (!t.taskId || !Array.isArray(t.inputs)) {
            console.warn(`⚠️ Skipping malformed litert task:`, t);
            return;
        }
        for (let r = 0; r < jobRedundancy; r++) {
            queuedTasks.push({
                jobId,
                taskId:       t.taskId,
                type:         'litert',        // routed to LiteRT-capable nodes only
                modelUrl,
                inputs:       t.inputs,        // [{ data[], shape[], dtype }]
                replicaIndex: r,
                requiresLiteRt: true           // dispatch guard flag
            });
        }
    });

    if (queuedTasks.length === 0) {
        return res.status(400).json({ error: 'No valid tasks after validation' });
    }

    const newJob = {
        id:            jobId,
        type:          'litert',
        status:        'queued',
        modelUrl,
        redundancy:    jobRedundancy,
        tasks:         queuedTasks,
        results:       {},
        completedCount: 0,
        totalCount:    tasks.length,   // logical task count (pre-redundancy)
        createdAt:     Date.now()
    };

    jobs[jobId] = newJob;
    heavyTaskQueue.push(...queuedTasks);
    totalTasksCount += queuedTasks.length;

    const liteRtNodes = Object.values(networkNodes).filter(n => n.litert).length;
    io.emit('log_event',
        `🧠 LiteRT Job ${jobId}: ${tasks.length} inference tasks | model: ${modelFile} | ` +
        `redundancy N=${jobRedundancy} | ${liteRtNodes} LiteRT-capable node(s) online`);
    io.emit('progress_update', { completedTasks, total: totalTasksCount });
    wakeAllNodes();

    res.status(202).json({
        jobId,
        status:     'queued',
        totalTasks: tasks.length,
        modelUrl,
        redundancy: jobRedundancy,
        liteRtNodesOnline: liteRtNodes
    });
});

// API: Submit a built-in SHA-256 Brute Force cracker job
app.post('/api/crack', (req, res) => {
    const { targetHash, prefix, maxNonce, redundancy } = req.body;
    const hash = targetHash || 'd7c7112040db42371a3962b9a7b97a3cf9367fbcf0b62d3a95d46e330e7ccfb0'; // 'grid1337'
    const pref = prefix || 'grid';
    const limit = maxNonce || 5000000;
    const chunkSize = 250000;
    const jobRedundancy = Math.max(1, Math.min(parseInt(redundancy, 10) || 1, 5));

    const tasks = [];
    let taskIdCounter = 0;
    for (let start = 0; start < limit; start += chunkSize) {
        const end = Math.min(start + chunkSize, limit);
        tasks.push({
            taskId: `crack_chunk_${taskIdCounter++}`,
            params: {
                targetHash: hash,
                prefix: pref,
                startNonce: start,
                endNonce: end
            }
        });
    }

    const script = `
        const { targetHash, prefix, startNonce, endNonce } = params;
        for (let nonce = startNonce; nonce < endNonce; nonce++) {
            const candidate = prefix + nonce;
            const msgBuffer = new TextEncoder().encode(candidate);
            const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
            const hashArray = Array.from(new Uint8Array(hashBuffer));
            const hash = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
            if (hash === targetHash) {
                return { found: true, password: candidate, nonce };
            }
        }
        return { found: false };
    `;

    const queuedTasks = [];
    tasks.forEach(t => {
        for (let r = 0; r < jobRedundancy; r++) {
            queuedTasks.push({
                jobId:       '', // filled below
                taskId:      t.taskId,
                type:        'custom',
                script,
                params:      t.params,
                replicaIndex: r
            });
        }
    });

    const jobId = 'job_crack_' + Math.random().toString(36).substring(2, 10);
    queuedTasks.forEach(t => t.jobId = jobId);

    const newJob = {
        id: jobId,
        type: 'custom',
        status: 'queued',
        script,
        redundancy: jobRedundancy,
        tasks: queuedTasks,
        results: {},
        completedCount: 0,
        totalCount: tasks.length, // total logical tasks
        createdAt: Date.now()
    };

    jobs[jobId] = newJob;
    heavyTaskQueue.push(...queuedTasks);

    completedTasks = 0;
    totalTasksCount = queuedTasks.length;

    io.emit('start_crack_job', { jobId, targetHash: hash, totalTasks: tasks.length, redundancy: jobRedundancy });
    io.emit('log_event', `🔑 SHA-256 crack job started for hash ${hash} (redundancy N=${jobRedundancy})`);
    io.emit('progress_update', { completedTasks, total: totalTasksCount });

    wakeAllNodes();

    res.status(202).json({ jobId, status: 'queued', totalTasks: tasks.length, redundancy: jobRedundancy });
});

// ─── Helper: broadcast dashboard ─────────────────────────────────────────────
function broadcastDashboard() {
    io.emit('update_dashboard', Object.values(networkNodes));
}

// ─── Helper: assign next task(s) to a socket until concurrency filled ─────────
function sendNextTask(socket) {
    if (!networkNodes[socket.id]) return;
    const node = networkNodes[socket.id];

    // Double-buffered prefetch: keep concurrency * 2 tasks in flight so workers never starve
    const targetBuffer = Math.max(2, (node.concurrency || 2) * 2);

    let taskIndex = 0;
    while (heavyTaskQueue.length > taskIndex && node.activeTasks.length < targetBuffer) {
        const task = heavyTaskQueue[taskIndex];

        // ── LiteRT routing guard ──────────────────────────────────────────────
        if (task.requiresLiteRt && !node.litert) {
            taskIndex++;
            continue;
        }

        // ── Specific backend requirement guard ────────────────────────────────
        if (task.requiredBackend && Array.isArray(node.backends) && !node.backends.includes(task.requiredBackend)) {
            taskIndex++;
            continue;
        }
        
        // Ensure this node does not process the same logical task twice (for redundancy consensus check)
        const alreadyAssigned = node.activeTasks.some(t => t.jobId === task.jobId && t.taskId === task.taskId) ||
                                (jobs[task.jobId] && 
                                 jobs[task.jobId].results[task.taskId] && 
                                 jobs[task.jobId].results[task.taskId].submissions && 
                                 jobs[task.jobId].results[task.taskId].submissions[socket.id]);
        
        if (alreadyAssigned) {
            taskIndex++;
            continue;
        }

        // Remove the task from the queue and assign to node
        heavyTaskQueue.splice(taskIndex, 1);
        node.status = task.requiresLiteRt ? 'Inferring' : 'Computing';
        node.activeTasks.push(task);
        socket.emit('process_task', task);
        console.log(`📤 Task jobId=${task.jobId} taskId=${task.taskId} type=${task.type} replica=${task.replicaIndex || 0} → ${node.name}`);
    }

    if (heavyTaskQueue.length === 0 && node.activeTasks.length === 0) {
        socket.emit('no_more_tasks');
        node.status = 'Idle';
    }
}

// ─── Helper: send tasks to ALL connected nodes ────────────────────────────────
function wakeAllNodes() {
    Object.keys(networkNodes).forEach(nodeId => {
        const nodeSocket = io.sockets.sockets.get(nodeId);
        if (nodeSocket) sendNextTask(nodeSocket);
    });
}

// ─── Helper: do a full render reset ──────────────────────────────────────────
function resetRender(broadcastViewport = true) {
    heavyTaskQueue  = generateTasks();
    completedTasks  = 0;
    totalTasksCount = heavyTaskQueue.length;

    jobs.mandelbulb = {
        id: 'mandelbulb',
        type: 'mandelbulb',
        status: 'processing',
        tasks: [],
        results: {},
        completedCount: 0,
        totalCount: heavyTaskQueue.length,
        createdAt: Date.now()
    };

    Object.keys(networkNodes).forEach(nodeId => {
        networkNodes[nodeId].completed    = 0;
        networkNodes[nodeId].activeTasks  = [];
    });

    io.emit('reset_canvas');
    io.emit('progress_update', { completedTasks, total: totalTasksCount });
    if (broadcastViewport) {
        io.emit('camera_updated', { camera, power: fractalPower });
    }
    broadcastDashboard();
    wakeAllNodes();
}

// ─── Socket.io connection handler ─────────────────────────────────────────────
io.on('connection', (socket) => {
    console.log(`📡 Node connected: ${socket.id}`);
    io.emit('log_event', `📡 New connection: Node ${socket.id.substring(0, 6)}... joined`);

    // Sync state on connect
    socket.emit('camera_updated', { camera, power: fractalPower });
    socket.emit('progress_update', { completedTasks, total: totalTasksCount });

    socket.on('log_event', () => { /* no-op — prevent log injection */ });

    // 1. Device registration
    socket.on('register_node', (profile) => {
        if (!profile || typeof profile.name !== 'string' || typeof profile.gcu !== 'number') {
            console.warn(`⚠️ Invalid register_node from ${socket.id}`);
            return;
        }

        const isNew = !networkNodes[socket.id];
        networkNodes[socket.id] = {
            id:             socket.id,
            name:           profile.name.substring(0, 32),
            cpu:            profile.cpu ? profile.cpu.substring(0, 64)   : 'Unknown CPU',
            gpu:            profile.gpu ? profile.gpu.substring(0, 256)  : 'Unknown GPU',
            dgpu:           profile.dgpu ? profile.dgpu.substring(0, 128) : null,
            igpu:           profile.igpu ? profile.igpu.substring(0, 128) : null,
            gcu:            Math.max(0, Math.min(profile.gcu, 99999)),
            benchmarks:     profile.benchmarks || {
                fp32_gflops: 0,
                fp16_gflops: 0,
                fp16_supported: false,
                int8_gops: 0,
                cpu_gflops: 0,
                total_gflops: 0
            },
            resourceTarget: typeof profile.resourceTarget === 'number' ? profile.resourceTarget : 90,
            backends:       Array.isArray(profile.backends) ? profile.backends : [],
            concurrency:    typeof profile.concurrency === 'number'
                                ? Math.max(1, Math.min(profile.concurrency, 64))
                                : 1,
            litert:         profile.litert === true,
            status:         'Ready',
            completed:      isNew ? 0 : (networkNodes[socket.id]?.completed || 0),
            activeTasks:    [],
            connectedAt:    Date.now()
        };

        const totalGflops = profile.benchmarks ? profile.benchmarks.total_gflops : profile.gcu;
        console.log(`✅ Registered: ${profile.name} (${totalGflops} GFLOPS/GCUs) [new=${isNew}]`);
        io.emit('log_event', `✅ Node joined: ${profile.name} — ${totalGflops} GFLOPS (Target: ${networkNodes[socket.id].resourceTarget}%)`);
        broadcastDashboard();

        sendNextTask(socket);
        socket.emit('camera_updated', { camera, power: fractalPower });
    });

    // 2. Task completion
    socket.on('task_completed', (payload) => {
        if (!networkNodes[socket.id]) return;
        const node = networkNodes[socket.id];

        if (!payload || !payload.jobId || payload.taskId === undefined) {
            console.warn(`⚠️ Invalid task_completed from ${socket.id}`);
            sendNextTask(socket);
            return;
        }

        if (!Array.isArray(node.activeTasks)) node.activeTasks = [];
        
        const taskIndex = node.activeTasks.findIndex(t => t.jobId === payload.jobId && t.taskId === payload.taskId);
        if (taskIndex === -1) {
            // Stale task from a previous render generation — ignore it
            sendNextTask(socket);
            return;
        }
        
        // Remove from active tasks
        const task = node.activeTasks.splice(taskIndex, 1)[0];

        node.completed++;
        completedTasks++;

        const job = jobs[payload.jobId];
        if (job) {
            // Ensure result sub-object exists for this task
            if (!job.results[payload.taskId]) {
                job.results[payload.taskId] = {
                    status: 'pending',
                    submissions: {}
                };
            }

            // Save this node's submission
            job.results[payload.taskId].submissions[socket.id] = {
                nodeName: node.name,
                backend: payload.backend || 'unknown',
                success: !payload.error,
                error: payload.error,
                value: payload.result
            };

            // ── Workload-specific assembly ────────────────────────────────────
            if (job.type === 'gemm' && payload.result && payload.result.tileResult && job.assembledResult) {
                const { tileRowStart, tileRowCount, tileResult } = payload.result;
                const N = job.params.N;
                for (let r = 0; r < tileRowCount; r++) {
                    const rowOffset = (tileRowStart + r) * N;
                    const tileOffset = r * N;
                    for (let c = 0; c < N; c++) {
                        job.assembledResult.matrixC[rowOffset + c] = tileResult[tileOffset + c];
                    }
                }
            } else if (job.type === 'monte_carlo' && payload.result && job.assembledResult) {
                job.assembledResult.totalHits += (payload.result.hits || 0);
            }

            const submissionsCount = Object.keys(job.results[payload.taskId].submissions).length;

            if (job.type === 'mandelbulb') {
                // Mandelbulb tasks have redundancy = 1, resolve immediately
                job.completedCount++;
                job.results[payload.taskId].status = 'success';
                job.results[payload.taskId].value = payload.result;
                
                io.emit('row_completed', {
                    yStart: task.yStart,
                    chunkHeight: task.chunkHeight,
                    pixels: payload.pixels || []
                });
            } else if (job.type === 'gemm') {
                // GEMM tile tasks resolve immediately
                job.completedCount++;
                job.results[payload.taskId].status = payload.error ? 'failed' : 'success';
                job.results[payload.taskId].value = payload.result;

                io.emit('gemm_tile_completed', {
                    jobId: job.id,
                    taskId: task.taskId,
                    completedCount: job.completedCount,
                    totalCount: job.totalCount,
                    nodeName: node.name,
                    backend: payload.backend
                });
            } else {
                // For custom tasks, check if we have reached the redundancy target
                if (submissionsCount >= job.redundancy) {
                    const subs = Object.values(job.results[payload.taskId].submissions);
                    
                    // Group similar results (stringified comparison)
                    const votes = {};
                    subs.forEach(s => {
                        const key = s.success ? JSON.stringify(s.value) : `__ERROR__:${s.error}`;
                        if (!votes[key]) {
                            votes[key] = { count: 0, success: s.success, value: s.value, error: s.error };
                        }
                        votes[key].count++;
                    });
                    
                    // Find candidate with majority votes
                    let winner = null;
                    let maxVotes = 0;
                    for (const key in votes) {
                        if (votes[key].count > maxVotes) {
                            maxVotes = votes[key].count;
                            winner = votes[key];
                        }
                    }
                    
                    const threshold = Math.ceil(job.redundancy / 2);
                    if (maxVotes >= threshold) {
                        job.completedCount++;
                        job.results[payload.taskId].status = winner.success ? 'success' : 'failed';
                        job.results[payload.taskId].value = winner.value;
                        job.results[payload.taskId].error = winner.error;
                        
                        console.log(`✅ Consensus reached for Task ${payload.taskId} (Job ${job.id}): ${maxVotes}/${job.redundancy} matches.`);
                        io.emit('log_event', `✅ Consensus reached: Task ${payload.taskId} resolved (${maxVotes}/${job.redundancy} matches)`);
                        
                        io.emit('custom_task_update', {
                            jobId: job.id,
                            taskId: task.taskId,
                            completedCount: job.completedCount,
                            totalCount: job.totalCount,
                            nodeName: node.name,
                            result: winner.value,
                            error: winner.error
                        });
                    } else {
                        console.warn(`⚠️ Consensus failed for Task ${payload.taskId} (Job ${job.id})! Re-queuing...`);
                        io.emit('log_event', `⚠️ Consensus FAILED: Task ${payload.taskId} got conflicting results. Re-queuing...`);
                        
                        job.results[payload.taskId].submissions = {};
                        job.results[payload.taskId].status = 'pending';
                        
                        for (let r = 0; r < job.redundancy; r++) {
                            heavyTaskQueue.unshift({
                                jobId: job.id,
                                taskId: task.taskId,
                                type: job.type,
                                script: job.script,
                                params: task.params,
                                replicaIndex: r
                            });
                            totalTasksCount++;
                        }
                    }
                }
            }

            if (job.completedCount === job.totalCount) {
                job.status = 'completed';
                job.completedAt = Date.now();
                const duration = ((job.completedAt - job.createdAt) / 1000).toFixed(2);
                let extraStats = '';

                if (job.type === 'gemm' && job.params) {
                    const { M, K, N } = job.params;
                    const totalOps = 2 * M * K * N;
                    const durationSec = Math.max(0.01, (job.completedAt - job.createdAt) / 1000);
                    const gflops = ((totalOps / durationSec) / 1e9).toFixed(2);
                    job.assembledResult.achievedGflops = parseFloat(gflops);
                    job.assembledResult.durationSeconds = parseFloat(duration);
                    extraStats = ` | Rate: ${gflops} GFLOPS`;
                } else if (job.type === 'monte_carlo' && job.assembledResult) {
                    job.assembledResult.estimatedPi = (4 * job.assembledResult.totalHits) / job.assembledResult.totalSamples;
                    extraStats = ` | π ≈ ${job.assembledResult.estimatedPi.toFixed(6)}`;
                }

                io.emit('job_completed', {
                    jobId: job.id,
                    type: job.type,
                    duration,
                    assembledResult: job.assembledResult
                });
                io.emit('log_event', `🎉 Job ${job.id} (${job.type}) completed in ${duration}s!${extraStats}`);
                console.log(`🎉 Job ${job.id} completed in ${duration}s${extraStats}`);
            }
        }

        io.emit('progress_update', { completedTasks, total: totalTasksCount });

        if (completedTasks % Math.ceil(totalTasksCount / 10) === 0 || completedTasks === totalTasksCount) {
            io.emit('log_event', `✅ Task ${payload.taskId} solved by ${node.name} (${completedTasks}/${totalTasksCount})`);
        }

        broadcastDashboard();
        sendNextTask(socket);
    });

    // 3. Camera orbit — sent by any client dragging/scrolling
    socket.on('set_camera', (params) => {
        if (typeof params.theta !== 'number' || typeof params.phi !== 'number' || typeof params.dist !== 'number') return;
        camera.theta = params.theta;
        camera.phi   = Math.max(-1.4, Math.min(1.4, params.phi));  // clamp to avoid gimbal lock
        camera.dist  = Math.max(1.2, Math.min(8.0, params.dist));   // clamp distance

        io.emit('log_event', `🎥 Camera: θ=${camera.theta.toFixed(2)} φ=${camera.phi.toFixed(2)} d=${camera.dist.toFixed(2)}`);
        resetRender();
    });

    // 4. Fractal power change
    socket.on('set_power', (power) => {
        const p = parseInt(power, 10);
        if (![4, 6, 8, 12, 16].includes(p)) return;
        fractalPower = p;
        io.emit('power_updated', fractalPower);
        io.emit('log_event', `🌀 Mandelbulb power set to ${fractalPower}`);
        resetRender();
    });

    // 5. Quality (max steps) change
    socket.on('set_quality', (steps) => {
        const s = parseInt(steps, 10);
        if (s >= 40 && s <= 200) {
            maxSteps = s;
            io.emit('quality_updated', maxSteps);
            io.emit('log_event', `⚙️ Quality set to ${maxSteps} ray steps`);
            resetRender();
        }
    });

    // 6. Full reset (inject_tasks)
    socket.on('inject_tasks', () => {
        camera = { theta: 0.5, phi: 0.3, dist: 2.5 };
        fractalPower = 8;
        maxSteps     = 80;
        io.emit('power_updated', fractalPower);
        io.emit('quality_updated', maxSteps);
        io.emit('log_event', `📥 Full reset — default Mandelbulb view`);
        resetRender();
    });

    // 7. Disconnect — re-queue node's in-flight tasks
    socket.on('disconnect', (reason) => {
        if (networkNodes[socket.id]) {
            const node = networkNodes[socket.id];
            if (node.activeTasks && node.activeTasks.length > 0) {
                [...node.activeTasks].reverse().forEach(task => {
                    heavyTaskQueue.unshift(task);
                    console.log(`♻️ Re-queued task taskId=${task.taskId} [${reason}]`);
                });
                io.emit('log_event', `♻️ Re-queued ${node.activeTasks.length} tasks from ${node.name}`);
            }
            io.emit('log_event', `❌ Node left: ${node.name}`);
            delete networkNodes[socket.id];
            broadcastDashboard();
        }
        console.log(`❌ Disconnected: ${socket.id} [${reason}]`);
    });
});

const PORT = process.env.PORT || 8080;
const HOST = process.env.HOST || '0.0.0.0';

// ─── Error handling & graceful shutdown ───────────────────────────────────────
server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        console.error(`\n🚫 Port ${PORT} is already in use.\n`);
        process.exit(1);
    } else {
        console.error('Server error:', err);
    }
});

function gracefulShutdown(signal) {
    console.log(`\n🛑 Received ${signal}. Shutting down GridCompute server...`);
    io.emit('log_event', '🛑 Server shutting down. Reconnect shortly.');
    server.close(() => { console.log('✅ Closed cleanly.'); process.exit(0); });
    setTimeout(() => process.exit(1), 5000);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT',  () => gracefulShutdown('SIGINT'));
process.on('uncaughtException',  (err) => console.error('💥 Uncaught exception:', err));
process.on('unhandledRejection', (r)   => console.error('💥 Unhandled rejection:', r));

server.listen(PORT, HOST, () => {
    console.log(`🚀 GridCompute running on http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORT}`);
    console.log('   Press Ctrl+C to stop.');
});