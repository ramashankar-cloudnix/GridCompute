const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: { origin: '*' },
    pingTimeout: 20000,
    pingInterval: 10000
});

let networkNodes = {};
const CANVAS_WIDTH = 600;
const CANVAS_HEIGHT = 300;
const MAX_LOG_EVENTS = 200; // Server-side log throttle

let viewport = {
    xmin: -2.0,
    xmax: 0.5,
    ymin: -1.2,
    ymax: 1.2
};

function generateMandelbrotTasks() {
    return Array.from({ length: CANVAS_HEIGHT }, (_, i) => ({
        id: i,
        y: i,
        width: CANVAS_WIDTH,
        height: CANVAS_HEIGHT,
        xmin: viewport.xmin,
        xmax: viewport.xmax,
        ymin: viewport.ymin,
        ymax: viewport.ymax,
        maxIterations: 1000 // 8x complexity with 4x4 SSAA
    }));
}

let heavyTaskQueue = generateMandelbrotTasks();
let totalTasksCount = CANVAS_HEIGHT;
let completedTasks = 0;

// --- Static file serving ---
app.get('/', (req, res) => res.sendFile(__dirname + '/index.html'));
app.get('/worker.js', (req, res) => res.sendFile(__dirname + '/worker.js'));
app.get('/NoSleep.min.js', (req, res) => res.sendFile(__dirname + '/NoSleep.min.js'));

// --- Helper: broadcast dashboard once per event ---
function broadcastDashboard() {
    io.emit('update_dashboard', Object.values(networkNodes));
}

// --- Helper: assign next task or mark idle ---
function sendNextTask(socket) {
    if (!networkNodes[socket.id]) return;

    if (heavyTaskQueue.length > 0) {
        const task = heavyTaskQueue.shift();
        networkNodes[socket.id].status = 'Computing';
        networkNodes[socket.id].activeTask = task;
        socket.emit('process_task', task);
        console.log(`📤 Row ${task.y} → ${networkNodes[socket.id].name}`);
    } else {
        socket.emit('no_more_tasks');
        networkNodes[socket.id].status = 'Idle';
        networkNodes[socket.id].activeTask = null;
    }
}

// --- Helper: wake up all idle/ready nodes for a fresh render ---
function wakeAllNodes() {
    Object.keys(networkNodes).forEach(nodeId => {
        const node = networkNodes[nodeId];
        if (node.status === 'Idle' || node.status === 'Ready') {
            const nodeSocket = io.sockets.sockets.get(nodeId);
            if (nodeSocket) sendNextTask(nodeSocket);
        }
    });
}

// --- Helper: validate zoom_to viewport payload ---
function isValidViewport(vp) {
    if (!vp || typeof vp !== 'object') return false;
    const { xmin, xmax, ymin, ymax } = vp;
    if (![xmin, xmax, ymin, ymax].every(v => typeof v === 'number' && isFinite(v))) return false;
    if (xmin >= xmax || ymin >= ymax) return false;
    return true;
}

// --- Socket.io connection handler ---
io.on('connection', (socket) => {
    console.log(`📡 Node connected: ${socket.id}`);
    io.emit('log_event', `📡 New connection: Node ${socket.id.substring(0, 6)}... joined`);

    // Sync viewport coordinates on connection
    socket.emit('viewport_updated', viewport);
    socket.emit('progress_update', { completedTasks, total: totalTasksCount });

    // BUG 7 FIX: Accept client log events but do NOT relay them to all clients.
    // Only server-authoritative events should be broadcast. This prevents log injection.
    socket.on('log_event', () => { /* intentionally no-op */ });

    // 1. Device Handshake & Benchmark registration
    // BUG 2 FIX: Only reset 'completed' if this is a brand-new socket ID.
    //            Re-registering an existing socket (rapid reconnect) preserves progress.
    socket.on('register_node', (hardwareProfile) => {
        if (!hardwareProfile || typeof hardwareProfile.name !== 'string' || typeof hardwareProfile.gcu !== 'number') {
            console.warn(`⚠️ Invalid register_node payload from ${socket.id}`);
            return;
        }

        const isNewNode = !networkNodes[socket.id];
        networkNodes[socket.id] = {
            id: socket.id,
            name: hardwareProfile.name.substring(0, 32), // sanitize length
            gcu: Math.max(0, Math.min(hardwareProfile.gcu, 99999)), // clamp
            status: 'Ready',
            completed: isNewNode ? 0 : (networkNodes[socket.id]?.completed || 0),
            activeTask: null,
            connectedAt: Date.now()
        };

        console.log(`✅ Registered: ${hardwareProfile.name} (${hardwareProfile.gcu} GCUs) [new=${isNewNode}]`);
        io.emit('log_event', `✅ Node joined: ${hardwareProfile.name} — ${hardwareProfile.gcu} GCUs`);
        broadcastDashboard();

        // Feed first task
        sendNextTask(socket);
    });

    // 2. Handle completed micro-tasks
    socket.on('task_completed', (payload) => {
        if (!networkNodes[socket.id]) return;

        const node = networkNodes[socket.id];

        // Validate payload
        if (!payload || typeof payload.y !== 'number' || !Array.isArray(payload.iterations)) {
            console.warn(`⚠️ Invalid task_completed payload from ${socket.id}`);
            sendNextTask(socket); // still give next task
            return;
        }

        node.completed++;
        node.activeTask = null;

        // Broadcast row to all canvases
        io.emit('row_completed', { y: payload.y, iterations: payload.iterations });

        completedTasks++;
        io.emit('log_event', `⚡ Row ${payload.y} ← ${node.name}`);

        // BUG 6 FIX: emit dashboard and progress only once per completion
        broadcastDashboard();
        io.emit('progress_update', { completedTasks, total: totalTasksCount });

        // Assign next work
        sendNextTask(socket);
    });

    // 3. Interactive zoom handler
    // BUG 8 FIX: Validate viewport before applying
    socket.on('zoom_to', (newViewport) => {
        if (!isValidViewport(newViewport)) {
            console.warn(`⚠️ Invalid zoom_to payload from ${socket.id}:`, newViewport);
            socket.emit('log_event', `⚠️ Invalid zoom payload rejected.`);
            return;
        }

        viewport = {
            xmin: newViewport.xmin,
            xmax: newViewport.xmax,
            ymin: newViewport.ymin,
            ymax: newViewport.ymax
        };

        heavyTaskQueue = generateMandelbrotTasks();
        completedTasks = 0;
        totalTasksCount = CANVAS_HEIGHT;

        // Reset all nodes' active tasks back into the queue
        Object.keys(networkNodes).forEach(nodeId => {
            networkNodes[nodeId].completed = 0;
            if (networkNodes[nodeId].activeTask) {
                heavyTaskQueue.unshift(networkNodes[nodeId].activeTask);
                networkNodes[nodeId].activeTask = null;
            }
        });

        io.emit('viewport_updated', viewport);
        io.emit('reset_canvas');
        io.emit('log_event', `🔍 Zoom applied. Rendering ${totalTasksCount} rows...`);
        io.emit('progress_update', { completedTasks, total: totalTasksCount });
        broadcastDashboard();

        wakeAllNodes();
    });

    // 4. Reset and start a fresh full render job
    socket.on('inject_tasks', () => {
        viewport = { xmin: -2.0, xmax: 0.5, ymin: -1.2, ymax: 1.2 };

        heavyTaskQueue = generateMandelbrotTasks();
        completedTasks = 0;
        totalTasksCount = CANVAS_HEIGHT;

        Object.keys(networkNodes).forEach(nodeId => {
            networkNodes[nodeId].completed = 0;
            if (networkNodes[nodeId].activeTask) {
                networkNodes[nodeId].activeTask = null;
            }
        });

        io.emit('viewport_updated', viewport);
        io.emit('reset_canvas');
        io.emit('log_event', `📥 Full reset — rendering ${totalTasksCount} rows at default viewport`);
        io.emit('progress_update', { completedTasks, total: totalTasksCount });
        broadcastDashboard();

        wakeAllNodes();
    });

    // 5. Disconnect — re-queue unfinished task
    socket.on('disconnect', (reason) => {
        if (networkNodes[socket.id]) {
            const node = networkNodes[socket.id];
            if (node.activeTask) {
                heavyTaskQueue.unshift(node.activeTask); // put task back at front
                io.emit('log_event', `♻️ Re-queued Row ${node.activeTask.y} from ${node.name}`);
                console.log(`♻️ Re-queued row ${node.activeTask.y} [reason: ${reason}]`);
            }
            io.emit('log_event', `❌ Node left: ${node.name}`);
            delete networkNodes[socket.id];
            broadcastDashboard();
        }
        console.log(`❌ Disconnected: ${socket.id} [${reason}]`);
    });
});

// BUG 12 FIX: Graceful shutdown + EADDRINUSE handling
server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        console.error(`\n🚫 Port 8080 is already in use.\n   Kill the other process or change the port and restart.\n`);
        process.exit(1);
    } else {
        console.error('Server error:', err);
    }
});

function gracefulShutdown(signal) {
    console.log(`\n🛑 Received ${signal}. Shutting down GridTorrent server...`);
    io.emit('log_event', '🛑 Server is shutting down. Please reconnect shortly.');
    server.close(() => {
        console.log('✅ Server closed cleanly.');
        process.exit(0);
    });
    // Force kill after 5s if hanging
    setTimeout(() => process.exit(1), 5000);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));
process.on('uncaughtException', (err) => {
    console.error('💥 Uncaught exception:', err);
});
process.on('unhandledRejection', (reason) => {
    console.error('💥 Unhandled rejection:', reason);
});

server.listen(8080, '0.0.0.0', () => {
    console.log('🚀 GridTorrent running on http://localhost:8080');
    console.log('   Press Ctrl+C to stop.');
});