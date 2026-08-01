const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');
const fs   = require('fs');
const multer = require('multer');

const app = express();
app.use(express.json()); // Support JSON body parsing for API submissions
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

let heavyTaskQueue  = generateTasks();
let totalTasksCount = heavyTaskQueue.length;
let completedTasks  = 0;

// ─── Static file serving ──────────────────────────────────────────────────────
app.get('/',             (req, res) => res.sendFile(__dirname + '/index.html'));
app.get('/worker.js',    (req, res) => res.sendFile(__dirname + '/worker.js'));
app.get('/worker_gpu.js',(req, res) => res.sendFile(__dirname + '/worker_gpu.js'));
app.get('/NoSleep.min.js',(req,res) => res.sendFile(__dirname + '/NoSleep.min.js'));
app.get('/worker_litert.js', (req, res) => res.sendFile(__dirname + '/worker_litert.js'));

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
const jobs = {
    mandelbulb: {
        id: 'mandelbulb',
        type: 'mandelbulb',
        status: 'processing',
        tasks: [],
        results: {},
        completedCount: 0,
        totalCount: 90,
        createdAt: Date.now()
    }
};

// API: Check status of a job
app.get('/api/job-status/:jobId', (req, res) => {
    const job = jobs[req.params.jobId];
    if (!job) {
        return res.status(404).json({ error: 'Job not found' });
    }
    res.json({
        jobId: job.id,
        type: job.type,
        status: job.status,
        progress: {
            completed: job.completedCount,
            total: job.totalCount,
            percent: Math.round((job.completedCount / job.totalCount) * 100)
        },
        results: job.results,
        createdAt: new Date(job.createdAt).toISOString(),
        completedAt: job.completedAt ? new Date(job.completedAt).toISOString() : null
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
    const { type, script, tasks, redundancy } = req.body;
    if (!tasks || !Array.isArray(tasks)) {
        return res.status(400).json({ error: 'Invalid tasks array' });
    }

    const jobRedundancy = Math.max(1, Math.min(parseInt(redundancy, 10) || 1, 5)); // cap at 5
    const jobId = 'job_' + Math.random().toString(36).substring(2, 10);
    
    const queuedTasks = [];
    tasks.forEach(t => {
        for (let r = 0; r < jobRedundancy; r++) {
            queuedTasks.push({
                jobId,
                taskId: t.taskId,
                type: type || 'custom',
                script: script || '',
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

    let taskIndex = 0;
    while (heavyTaskQueue.length > taskIndex && node.activeTasks.length < node.concurrency) {
        const task = heavyTaskQueue[taskIndex];

        // ── LiteRT routing guard ──────────────────────────────────────────────
        // LiteRT inference tasks can only run on nodes that loaded worker_litert.js
        // and confirmed LiteRT.js support via register_node { litert: true }.
        if (task.requiresLiteRt && !node.litert) {
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
            id:          socket.id,
            name:        profile.name.substring(0, 32),
            cpu:         profile.cpu ? profile.cpu.substring(0, 64)   : 'Unknown CPU',
            gpu:         profile.gpu ? profile.gpu.substring(0, 256)  : 'Unknown GPU',
            gcu:         Math.max(0, Math.min(profile.gcu, 99999)),
            concurrency: typeof profile.concurrency === 'number'
                             ? Math.max(1, Math.min(profile.concurrency, 32))
                             : 1,
            // LiteRT.js capability — true when node is running worker_litert.js
            // and has successfully initialised the LiteRT Wasm runtime.
            litert:      profile.litert === true,
            status:      'Ready',
            completed:   isNew ? 0 : (networkNodes[socket.id]?.completed || 0),
            activeTasks: [],
            connectedAt: Date.now()
        };

        console.log(`✅ Registered: ${profile.name} (${profile.gcu} GCUs) [new=${isNew}]`);
        io.emit('log_event', `✅ Node joined: ${profile.name} — ${profile.gcu} GCUs`);
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
                success: !payload.error,
                error: payload.error,
                value: payload.result
            };

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
                    
                    // Find the candidate with the majority votes
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
                        // Consensus reached!
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
                        // Consensus failed! No majority vote. Re-queue task replicas!
                        console.warn(`⚠️ Consensus failed for Task ${payload.taskId} (Job ${job.id})! Re-queuing...`);
                        io.emit('log_event', `⚠️ Consensus FAILED: Task ${payload.taskId} got conflicting results. Re-queuing...`);
                        
                        // Clear the submissions for this task
                        job.results[payload.taskId].submissions = {};
                        job.results[payload.taskId].status = 'pending';
                        
                        // Re-queue task replicas
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
                io.emit('log_event', `🎉 Job ${job.id} (${job.type}) completed in ${duration}s!`);
                console.log(`🎉 Job ${job.id} completed in ${duration}s`);
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

// ─── Error handling & graceful shutdown ───────────────────────────────────────
server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        console.error(`\n🚫 Port 8080 is already in use.\n`);
        process.exit(1);
    } else {
        console.error('Server error:', err);
    }
});

function gracefulShutdown(signal) {
    console.log(`\n🛑 Received ${signal}. Shutting down GridTorrent server...`);
    io.emit('log_event', '🛑 Server shutting down. Reconnect shortly.');
    server.close(() => { console.log('✅ Closed cleanly.'); process.exit(0); });
    setTimeout(() => process.exit(1), 5000);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT',  () => gracefulShutdown('SIGINT'));
process.on('uncaughtException',  (err) => console.error('💥 Uncaught exception:', err));
process.on('unhandledRejection', (r)   => console.error('💥 Unhandled rejection:', r));

server.listen(8080, '0.0.0.0', () => {
    console.log('🚀 GridTorrent 3D running on http://localhost:8080');
    console.log('   Press Ctrl+C to stop.');
});