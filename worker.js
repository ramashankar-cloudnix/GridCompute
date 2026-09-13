// GridCompute CPU Web Worker — JavaScript Compute & Raymarcher Fallback
// Renders a chunk of rows via SDF sphere-tracing in JavaScript.
// Output: RGBA Uint8Array — same format as the GPU worker.

// ── Mandelbulb Distance Estimator ────────────────────────────────────────────
function mandelbulbDE(px, py, pz, pw) {
    let zx = px, zy = py, zz = pz;
    let dr = 1.0, r = 0.0;

    for (let i = 0; i < 15; i++) {
        r = Math.sqrt(zx*zx + zy*zy + zz*zz);
        if (r > 2.0) break;
        if (r < 0.00001) break; // Prevent divide-by-zero / NaN at origin

        let theta = Math.acos(clamp(zz / r, -1, 1));
        let phi   = Math.atan2(zy, zx);
        dr = Math.pow(r, pw - 1.0) * pw * dr + 1.0;

        const zr = Math.pow(r, pw);
        theta *= pw;
        phi   *= pw;

        zx = zr * Math.sin(theta) * Math.cos(phi);
        zy = zr * Math.sin(theta) * Math.sin(phi);
        zz = zr * Math.cos(theta);

        zx += px; zy += py; zz += pz;
    }
    if (r < 0.00001) return 0.0; // Safe distance at origin
    return 0.5 * Math.log(r) * r / dr;
}

function sdf(x, y, z, power) { return mandelbulbDE(x, y, z, power); }

// ── Normal estimation (central finite differences) ───────────────────────────
function calcNormal(x, y, z, power) {
    const e = 0.0008;
    const nx = sdf(x+e,y,z,power) - sdf(x-e,y,z,power);
    const ny = sdf(x,y+e,z,power) - sdf(x,y-e,z,power);
    const nz = sdf(x,y,z+e,power) - sdf(x,y,z-e,power);
    const len = Math.sqrt(nx*nx+ny*ny+nz*nz) || 1;
    return [nx/len, ny/len, nz/len];
}

// ── Ambient occlusion ─────────────────────────────────────────────────────────
function calcAO(px, py, pz, nx, ny, nz, power) {
    let occ = 0, sca = 1;
    for (let i = 0; i < 4; i++) {
        const h = 0.01 + 0.1 * i / 3;
        occ += (h - sdf(px+nx*h, py+ny*h, pz+nz*h, power)) * sca;
        sca *= 0.95;
    }
    return Math.max(0, Math.min(1, 1 - 3 * occ));
}

// ── Soft shadow ───────────────────────────────────────────────────────────────
function softShadow(rox,roy,roz, rdx,rdy,rdz, mint, maxt, power) {
    let res = 1, t = mint;
    for (let i = 0; i < 12; i++) {
        const h = sdf(rox+rdx*t, roy+rdy*t, roz+rdz*t, power);
        res = Math.min(res, 10 * h / t);
        t  += Math.max(h, 0.005);
        if (res < 0.005 || t > maxt) break;
    }
    return Math.max(0, Math.min(1, res));
}

// ── Clamp helper ──────────────────────────────────────────────────────────────
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ── Colour mapping (matches GPU palette) ──────────────────────────────────────
function toSRGB(v) { return Math.pow(Math.max(0, v), 1/2.2); }

// ─── CPU GFLOPS Benchmark (Scalable 1x to 16x) ───────────────────────────
// scale: 1 to 16, where 1x is the default baseline and 16x is peak intensity
function benchmarkCPU(scale = 1) {
    const factor = Math.max(1, Math.min(16, Number(scale) || 1));
    const N = 3200000; // 3.2M Float32 elements
    const a = new Float32Array(N);
    const b = new Float32Array(N);
    for (let i = 0; i < N; i++) {
        a[i] = i * 0.001 + 0.1;
        b[i] = 1.001;
    }

    // Iterations scale from 40 (at 1x) to 640 (at 16x)
    const iterations = Math.max(40, Math.round(640 * (factor / 16)));
    const tStart = performance.now();
    for (let iter = 0; iter < iterations; iter++) {
        for (let i = 0; i < N; i += 4) {
            a[i]   = a[i]   * b[i]   + 0.0001;
            a[i+1] = a[i+1] * b[i+1] + 0.0001;
            a[i+2] = a[i+2] * b[i+2] + 0.0001;
            a[i+3] = a[i+3] * b[i+3] + 0.0001;
        }
    }
    const elapsed = performance.now() - tStart;
    const totalOps = iterations * N * 2;
    const gflops = (totalOps / (Math.max(1, elapsed) / 1000)) / 1e9;
    return parseFloat(gflops.toFixed(2));
}

// ─── Main message handler ─────────────────────────────────────────────────────
self.onmessage = function(e) {
    const task = e.data;
    if (!task) return;

    if (task.action === 'benchmark') {
        try {
            const scale = Math.max(1, Math.min(16, Number(task.scale) || 1));
            self.postMessage({ type: 'benchmark_progress', phase: 'cpu', percent: 88, msg: `Evaluating CPU Multi-Thread Worker (${scale}x)...` });
            const gflops = benchmarkCPU(scale);
            self.postMessage({
                type: 'cpu_benchmark_complete',
                scale: scale,
                fp32_gflops: gflops,
                int8_gops: parseFloat((gflops * 1.2).toFixed(2))
            });
        } catch (err) {
            self.postMessage({ type: 'cpu_benchmark_error', error: err.message });
        }
        return;
    }

    if (task.type === 'gemm') {
        try {
            const { tileRowStart, tileRowCount, K, N, subA, matrixB } = task.params;
            const tStart = performance.now();
            const out = new Float32Array(tileRowCount * N);
            for (let r = 0; r < tileRowCount; r++) {
                const rOffset = r * K;
                const outOffset = r * N;
                for (let c = 0; c < N; c++) {
                    let s = 0.0;
                    for (let k = 0; k < K; k++) {
                        s += subA[rOffset + k] * matrixB[k * N + c];
                    }
                    out[outOffset + c] = s;
                }
            }
            const durationMs = performance.now() - tStart;
            self.postMessage({
                jobId: task.jobId,
                taskId: task.taskId,
                type: 'gemm',
                backend: 'cpu-worker',
                result: {
                    tileRowStart,
                    tileRowCount,
                    durationMs: parseFloat(durationMs.toFixed(2)),
                    tileResult: Array.from(out)
                }
            });
        } catch (err) {
            self.postMessage({ jobId: task.jobId, taskId: task.taskId, type: 'gemm', error: err.message });
        }
        return;
    }

    try {
        if (task.type === 'custom') {
            if (task.imports && Array.isArray(task.imports)) {
                task.imports.forEach(url => {
                    try {
                        self.importScripts(url);
                    } catch (e) {
                        console.warn('Worker importScripts warning:', url, e.message);
                    }
                });
            }
            const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
            const computeFn = new AsyncFunction('params', task.script);
            computeFn(task.params).then(result => {
                self.postMessage({
                    jobId: task.jobId,
                    taskId: task.taskId,
                    result: result
                });
            }).catch(err => {
                self.postMessage({
                    jobId: task.jobId,
                    taskId: task.taskId,
                    error: err.message || 'Compute error'
                });
            });
            return;
        }

        const { yStart, chunkHeight, width, height,
                camTheta, camPhi, camDist, power, maxSteps } = task;

        if (typeof yStart !== 'number' || typeof chunkHeight !== 'number') {
            self.postMessage({ error: 'Invalid task parameters', yStart: task.yStart ?? -1 });
            return;
        }

        // Build camera (match GPU shader exactly)
        const cosT = Math.cos(camTheta), sinT = Math.sin(camTheta);
        const cosP = Math.cos(camPhi),   sinP = Math.sin(camPhi);
        const rox = camDist * sinT * cosP;
        const roy = camDist * sinP;
        const roz = camDist * cosT * cosP;

        // Camera basis vectors
        const cwx = -rox, cwy = -roy, cwz = -roz;
        const cwLen = Math.sqrt(cwx*cwx+cwy*cwy+cwz*cwz)||1;
        const cwNx = cwx/cwLen, cwNy = cwy/cwLen, cwNz = cwz/cwLen;

        // cu = normalize(cw × up)
        const upx = 0, upy = 1, upz = 0;
        let cux = cwNy*upz - cwNz*upy;
        let cuy = cwNz*upx - cwNx*upz;
        let cuz = cwNx*upy - cwNy*upx;
        const cuLen = Math.sqrt(cux*cux+cuy*cuy+cuz*cuz)||1;
        cux /= cuLen; cuy /= cuLen; cuz /= cuLen;

        // cv = cu × cw
        const cvx = cuy*cwNz - cuz*cwNy;
        const cvy = cuz*cwNx - cux*cwNz;
        const cvz = cux*cwNy - cuy*cwNx;

        const aspect = width / height;
        const pixels  = new Uint8Array(width * chunkHeight * 4);

        for (let row = 0; row < chunkHeight; row++) {
            const canvasY  = yStart + row;
            const bufBase  = row * width * 4;

            for (let col = 0; col < width; col++) {
                const canvasX = col;

                // NDC coordinates (same as GPU shader)
                const uvx = ((canvasX / width)  * 2 - 1) * aspect;
                const uvy =  1 - (canvasY / height) * 2;

                // Ray direction: normalize(uvx*cu + uvy*cv + 1.8*cw)
                const fov = 1.8;
                let rdx = uvx*cux + uvy*cvx + fov*cwNx;
                let rdy = uvx*cuy + uvy*cvy + fov*cwNy;
                let rdz = uvx*cuz + uvy*cvz + fov*cwNz;
                const rdLen = Math.sqrt(rdx*rdx+rdy*rdy+rdz*rdz)||1;
                rdx /= rdLen; rdy /= rdLen; rdz /= rdLen;

                // ── Sphere-trace ───────────────────────────────────────────
                let t = 0, hit = false, step = 0;
                const cpuMaxSteps = Math.min(maxSteps, 60); // CPU budget cap
                for (step = 0; step < cpuMaxSteps; step++) {
                    const h = sdf(rox+rdx*t, roy+rdy*t, roz+rdz*t, power);
                    if (h < 0.0004) { hit = true; break; }
                    if (t > 6.0)    break;
                    t += h;
                }

                let r8, g8, b8;

                if (hit) {
                    const px = rox+rdx*t, py = roy+rdy*t, pz = roz+rdz*t;
                    const [nx, ny, nz] = calcNormal(px, py, pz, power);

                    // Light direction
                    const lix = 0.5, liy = 0.8, liz = 0.4;
                    const liLen = Math.sqrt(lix*lix+liy*liy+liz*liz);
                    const lx = lix/liLen, ly = liy/liLen, lz = liz/liLen;

                    const dif = clamp(nx*lx + ny*ly + nz*lz, 0, 1);
                    const sha = softShadow(px+nx*0.001, py+ny*0.001, pz+nz*0.001, lx,ly,lz, 0.01, 4, power);
                    const ao  = calcAO(px,py,pz, nx,ny,nz, power);

                    // Fresnel rim
                    const rimDot = clamp(1 + (nx*rdx + ny*rdy + nz*rdz), 0, 1);
                    const rim    = rimDot * rimDot * rimDot;

                    // Neon palette (deep teal ↔ purple)
                    const t2 = clamp(ny * 0.5 + 0.5, 0, 1);
                    const baseR = 0.04 + t2 * 0.36;
                    const baseG = 0.35 - t2 * 0.27;
                    const baseB = 0.30 + t2 * 0.20;

                    let cr = baseR * (0.25*ao + 0.75*dif*sha);
                    let cg = baseG * (0.25*ao + 0.75*dif*sha);
                    let cb = baseB * (0.25*ao + 0.75*dif*sha);

                    // Rim glow (emerald)
                    cr += 0.35 * rim * 0.25 * sha * ao;
                    cg += 0.35 * rim * 0.95 * sha * ao;
                    cb += 0.35 * rim * 0.75 * sha * ao;

                    // Sky bounce
                    cr += 0.08 * ao * 0.08;
                    cg += 0.08 * ao * 0.18;
                    cb += 0.08 * ao * 0.28;

                    // Depth tinting
                    const dep = step / cpuMaxSteps;
                    cg += dep * 0.4 * (0.6 * cg - cg);
                    cb += dep * 0.4 * (0.5 * cb - cb);

                    // Gamma correction
                    r8 = Math.round(toSRGB(cr) * 255);
                    g8 = Math.round(toSRGB(cg) * 255);
                    b8 = Math.round(toSRGB(cb) * 255);
                } else {
                    // Deep-space background
                    const t2  = 0.5 * (rdy + 1);
                    const bgR = (0.01 + t2 * 0.01) * 255;
                    const bgG = (0.02 + t2 * 0.01) * 255;
                    const bgB = (0.06 + t2 * 0.03) * 255;
                    r8 = Math.round(bgR);
                    g8 = Math.round(bgG);
                    b8 = Math.round(bgB);
                }

                const off = bufBase + col * 4;
                pixels[off]   = clamp(r8, 0, 255);
                pixels[off+1] = clamp(g8, 0, 255);
                pixels[off+2] = clamp(b8, 0, 255);
                pixels[off+3] = 255;
            }
        }

        self.postMessage({ jobId: 'mandelbulb', yStart, chunkHeight, pixels: Array.from(pixels) });

    } catch (err) {
        if (e.data && e.data.type === 'custom') {
            self.postMessage({
                jobId: e.data.jobId,
                taskId: e.data.taskId,
                error: err.message || 'CPU execution error'
            });
        } else {
            self.postMessage({
                error:  err.message || 'CPU raymarcher error',
                yStart: (e.data && typeof e.data.yStart === 'number') ? e.data.yStart : -1
            });
        }
    }
};
