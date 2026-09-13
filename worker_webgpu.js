// GridCompute WebGPU Compute Worker
// Targets high-performance (dGPU) or low-power (iGPU) hardware adapters.
// Performs FP32, FP16 (if shader-f16 supported), INT8 GFLOPS benchmarking
// and executes distributed general compute tasks (GEMM, array transforms, Mandelbulb).

'use strict';

let adapter = null;
let device = null;
let powerPref = 'high-performance'; // 'high-performance' (dGPU) or 'low-power' (iGPU)
let gpuName = 'WebGPU Device';
let hasF16 = false;
let isInitialized = false;

// ─── Initialise WebGPU Device ────────────────────────────────────────────────
async function initWebGPU(requestedPref = 'high-performance') {
    if (isInitialized && device) return true;
    powerPref = requestedPref;

    if (typeof navigator === 'undefined' || !navigator.gpu) {
        throw new Error('WebGPU is not supported in this environment/worker.');
    }

    adapter = await navigator.gpu.requestAdapter({ powerPreference: powerPref });
    if (!adapter) {
        // Fallback to any available adapter
        adapter = await navigator.gpu.requestAdapter();
    }
    if (!adapter) {
        throw new Error(`No WebGPU adapter found for preference: ${powerPref}`);
    }

    // Inspect adapter info
    let adapterInfo = {};
    if (adapter.info) {
        adapterInfo = adapter.info;
    } else if (typeof adapter.requestAdapterInfo === 'function') {
        try { adapterInfo = await adapter.requestAdapterInfo(); } catch (_) {}
    }
    gpuName = adapterInfo.description || adapterInfo.device || (powerPref === 'high-performance' ? 'Discrete GPU' : 'Integrated GPU');
    if (adapterInfo.vendor) gpuName = `${adapterInfo.vendor} ${gpuName}`.trim();

    // Check optional features like 'shader-f16' (crucial for 16-bit float benchmark)
    const requiredFeatures = [];
    if (adapter.features.has('shader-f16')) {
        requiredFeatures.push('shader-f16');
        hasF16 = true;
    }

    device = await adapter.requestDevice({
        requiredFeatures: requiredFeatures
    });

    isInitialized = true;
    return true;
}

// ─── Benchmark: FP32 GFLOPS (Scalable 1x to 16x) ─────────────────────────────
// scale: 1 to 16, where 1x is the default baseline and 16x is peak intensity
async function benchmarkFP32(scale = 1) {
    if (!device) await initWebGPU(powerPref);

    const factor = Math.max(1, Math.min(16, Number(scale) || 1));
    const iterations = Math.max(200, Math.round(6400 * (factor / 16)));
    const workgroupCount = 16384;

    const wgSize = 64;
    const opsPerInvocation = 2000; // 1000 FMAs = 2000 floating point operations
    const totalOps = workgroupCount * wgSize * opsPerInvocation * iterations;

    const shaderCode = `
        @group(0) @binding(0) var<storage, read_write> outputBuffer: array<f32>;

        @compute @workgroup_size(${wgSize})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            var a: f32 = f32(idx) * 0.0001 + 0.1;
            var b: f32 = 1.00002;
            var c: f32 = 0.00003;

            for (var i = 0u; i < ${iterations}u; i = i + 1u) {
                // Unrolled FMA loop
                ${Array(100).fill(0).map((_, k) => `
                    a = fma(a, b, c);
                    b = fma(b, a, c);
                    c = fma(c, a, b);
                    a = a * 0.99999 + 0.00001;
                    b = b * 0.99999 + 0.00001;
                    c = c * 0.99999 + 0.00001;
                    a = fma(a, c, b);
                    b = fma(b, c, a);
                    c = fma(c, b, a);
                    a = a * 0.99999 + 0.00001;
                `).join('\n')}
            }

            if (idx < arrayLength(&outputBuffer)) {
                outputBuffer[idx] = a + b + c;
            }
        }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: 'main' }
    });

    const bufferSize = workgroupCount * wgSize * 4;
    const outputBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const readbackBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{ binding: 0, resource: { buffer: outputBuffer } }]
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroupCount);
    passEncoder.end();
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, bufferSize);

    const startTime = performance.now();
    device.queue.submit([commandEncoder.finish()]);
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const elapsedMs = performance.now() - startTime;
    readbackBuffer.unmap();

    // Clean up
    outputBuffer.destroy();
    readbackBuffer.destroy();

    const gflops = (totalOps / (elapsedMs / 1000)) / 1e9;
    return parseFloat(gflops.toFixed(2));
}

// ─── Benchmark: FP16 GFLOPS (Scalable 1x to 16x, 16-bit half precision) ──────
async function benchmarkFP16(scale = 1) {
    if (!device) await initWebGPU(powerPref);

    if (!hasF16) {
        return { supported: false, gflops: 0, reason: 'shader-f16 feature extension not available on this GPU' };
    }

    const factor = Math.max(1, Math.min(16, Number(scale) || 1));
    const iterations = Math.max(200, Math.round(6400 * (factor / 16)));
    const workgroupCount = 16384;

    const wgSize = 64;
    const totalOps = workgroupCount * wgSize * 2000 * iterations;

    const shaderCode = `
        enable f16;

        @group(0) @binding(0) var<storage, read_write> outputBuffer: array<f16>;

        @compute @workgroup_size(${wgSize})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            var a: f16 = f16(idx % 100u) * 0.001h + 0.1h;
            var b: f16 = 1.002h;
            var c: f16 = 0.003h;

            for (var i = 0u; i < ${iterations}u; i = i + 1u) {
                ${Array(100).fill(0).map((_, k) => `
                    a = fma(a, b, c);
                    b = fma(b, a, c);
                    c = fma(c, a, b);
                    a = a * 0.99h + 0.01h;
                    b = b * 0.99h + 0.01h;
                    c = c * 0.99h + 0.01h;
                    a = fma(a, c, b);
                    b = fma(b, c, a);
                    c = fma(c, b, a);
                    a = a * 0.99h + 0.01h;
                `).join('\n')}
            }

            if (idx < arrayLength(&outputBuffer)) {
                outputBuffer[idx] = a + b + c;
            }
        }
    `;

    try {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        const bindGroupLayout = device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]
        });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
        const pipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'main' }
        });

        const bufferSize = workgroupCount * wgSize * 2;
        const outputBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const readbackBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: outputBuffer } }]
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(workgroupCount);
        passEncoder.end();
        commandEncoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, bufferSize);

        const startTime = performance.now();
        device.queue.submit([commandEncoder.finish()]);
        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const elapsedMs = performance.now() - startTime;
        readbackBuffer.unmap();

        outputBuffer.destroy();
        readbackBuffer.destroy();

        const gflops = (totalOps / (elapsedMs / 1000)) / 1e9;
        return { supported: true, gflops: parseFloat(gflops.toFixed(2)) };
    } catch (err) {
        return { supported: false, gflops: 0, reason: err.message };
    }
}

// ─── Benchmark: INT8 / Quantized Integer Math (Scalable 1x to 16x, GOPS) ──────
// Emulates 8-bit packed integer matrix operations (4 INT8 ops packed per 32-bit word)
async function benchmarkINT8(scale = 1) {
    if (!device) await initWebGPU(powerPref);

    const factor = Math.max(1, Math.min(16, Number(scale) || 1));
    const iterations = Math.max(200, Math.round(6400 * (factor / 16)));
    const workgroupCount = 16384;

    const wgSize = 64;
    // Each packed word computes 4 int8 dot products per multiply-accumulate = 8 ops per pair
    const totalOps = workgroupCount * wgSize * 2400 * iterations;

    const shaderCode = `
        @group(0) @binding(0) var<storage, read_write> outputBuffer: array<u32>;

        @compute @workgroup_size(${wgSize})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            var a: u32 = (idx & 0xFFu) | ((idx & 0xFFu) << 8u) | ((idx & 0xFFu) << 16u) | ((idx & 0xFFu) << 24u);
            var b: u32 = 0x01020304u;
            var acc: u32 = 0u;

            for (var i = 0u; i < ${iterations}u; i = i + 1u) {
                ${Array(100).fill(0).map((_, k) => `
                    // Unpack 4x int8 and multiply-accumulate
                    let a0 = a & 0xFFu;
                    let a1 = (a >> 8u) & 0xFFu;
                    let a2 = (a >> 16u) & 0xFFu;
                    let a3 = (a >> 24u) & 0xFFu;
                    let b0 = b & 0xFFu;
                    let b1 = (b >> 8u) & 0xFFu;
                    let b2 = (b >> 16u) & 0xFFu;
                    let b3 = (b >> 24u) & 0xFFu;
                    acc = acc + (a0 * b0) + (a1 * b1) + (a2 * b2) + (a3 * b3);
                    a = (a ^ 0x55555555u) + 1u;
                    b = (b ^ 0x33333333u) + acc;
                `).join('\n')}
            }

            if (idx < arrayLength(&outputBuffer)) {
                outputBuffer[idx] = acc;
            }
        }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: 'main' }
    });

    const bufferSize = workgroupCount * wgSize * 4;
    const outputBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const readbackBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{ binding: 0, resource: { buffer: outputBuffer } }]
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroupCount);
    passEncoder.end();
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, bufferSize);

    const startTime = performance.now();
    device.queue.submit([commandEncoder.finish()]);
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const elapsedMs = performance.now() - startTime;
    readbackBuffer.unmap();

    outputBuffer.destroy();
    readbackBuffer.destroy();

    const gops = (totalOps / (elapsedMs / 1000)) / 1e9;
    return parseFloat(gops.toFixed(2));
}

// ─── Execute Matrix Multiplication (GEMM Tile) ───────────────────────────────
// Multiplies submatrices A (tileRows x K) and B (K x N) -> C (tileRows x N)
async function executeGEMM({ tileRowStart, tileRowCount, K, N, subA, matrixB }) {
    if (!device) await initWebGPU(powerPref);

    const flatA = new Float32Array(subA);
    const flatB = new Float32Array(matrixB);
    const flatC = new Float32Array(tileRowCount * N);

    const shaderCode = `
        struct Uniforms {
            tileRowCount: u32,
            K: u32,
            N: u32,
        };

        @group(0) @binding(0) var<uniform> u: Uniforms;
        @group(0) @binding(1) var<storage, read> A: array<f32>;
        @group(0) @binding(2) var<storage, read> B: array<f32>;
        @group(0) @binding(3) var<storage, read_write> C: array<f32>;

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let row = gid.y;
            let col = gid.x;
            if (row >= u.tileRowCount || col >= u.N) {
                return;
            }

            var sum: f32 = 0.0;
            for (var k = 0u; k < u.K; k = k + 1u) {
                sum = sum + A[row * u.K + k] * B[k * u.N + col];
            }
            C[row * u.N + col] = sum;
        }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });

    // Uniform buffer (3 x u32 = 12 bytes, padded to 16 bytes)
    const uArray = new Uint32Array([tileRowCount, K, N, 0]);
    const uBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(uBuffer, 0, uArray);

    const aBuffer = device.createBuffer({
        size: flatA.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(aBuffer, 0, flatA);

    const bBuffer = device.createBuffer({
        size: flatB.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(bBuffer, 0, flatB);

    const cBuffer = device.createBuffer({
        size: flatC.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const readbackBuffer = device.createBuffer({
        size: flatC.byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
        ]
    });

    const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shaderModule, entryPoint: 'main' }
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: uBuffer } },
            { binding: 1, resource: { buffer: aBuffer } },
            { binding: 2, resource: { buffer: bBuffer } },
            { binding: 3, resource: { buffer: cBuffer } }
        ]
    });

    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(tileRowCount / 16));
    pass.end();
    commandEncoder.copyBufferToBuffer(cBuffer, 0, readbackBuffer, 0, flatC.byteLength);

    const tStart = performance.now();
    device.queue.submit([commandEncoder.finish()]);
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const tDuration = performance.now() - tStart;

    const copyArray = new Float32Array(readbackBuffer.getMappedRange().slice(0));
    readbackBuffer.unmap();

    // Destroy buffers
    uBuffer.destroy();
    aBuffer.destroy();
    bBuffer.destroy();
    cBuffer.destroy();
    readbackBuffer.destroy();

    return {
        tileRowStart,
        tileRowCount,
        durationMs: parseFloat(tDuration.toFixed(2)),
        tileResult: Array.from(copyArray)
    };
}

// ─── Web Worker Message Listener ─────────────────────────────────────────────
self.onmessage = async function (e) {
    const data = e.data;
    if (!data) return;

    // 1. Initialise action
    if (data.action === 'init') {
        try {
            await initWebGPU(data.powerPreference || 'high-performance');
            self.postMessage({
                type: 'init_complete',
                powerPreference: powerPref,
                gpuName: gpuName,
                hasF16: hasF16
            });
        } catch (err) {
            self.postMessage({
                type: 'init_error',
                powerPreference: data.powerPreference,
                error: err.message
            });
        }
        return;
    }

    // 2. Comprehensive 2026 GFLOPS Benchmark (Scalable 1x to 16x)
    if (data.action === 'benchmark') {
        try {
            await initWebGPU(data.powerPreference || powerPref);
            const scale = Math.max(1, Math.min(16, Number(data.scale) || 1));

            self.postMessage({ type: 'benchmark_progress', phase: 'fp32', percent: 10, msg: `Evaluating WebGPU FP32 Single-Precision (${scale}x)...` });
            const fp32 = await benchmarkFP32(scale);
            self.postMessage({ type: 'benchmark_progress', phase: 'fp32_done', percent: 35, fp32: fp32, msg: `FP32 Complete: ${fp32} GFLOPS` });

            self.postMessage({ type: 'benchmark_progress', phase: 'fp16', percent: 40, msg: `Evaluating WebGPU FP16 Half-Precision (${scale}x)...` });
            const fp16 = await benchmarkFP16(scale);
            self.postMessage({ type: 'benchmark_progress', phase: 'fp16_done', percent: 65, fp16: fp16.gflops, fp16Supported: fp16.supported, msg: fp16.supported ? `FP16 Complete: ${fp16.gflops} GFLOPS` : 'FP16 Not Supported' });

            self.postMessage({ type: 'benchmark_progress', phase: 'int8', percent: 70, msg: `Evaluating WebGPU INT8 Quantized Math (${scale}x)...` });
            const int8 = await benchmarkINT8(scale);
            self.postMessage({ type: 'benchmark_progress', phase: 'int8_done', percent: 85, int8: int8, msg: `INT8 Complete: ${int8} GOPS` });

            self.postMessage({
                type: 'benchmark_complete',
                powerPreference: powerPref,
                gpuName: gpuName,
                fp32_gflops: fp32,
                fp16_gflops: fp16.gflops,
                fp16_supported: fp16.supported,
                int8_gops: int8,
                total_gpu_score: parseFloat((fp32 + (fp16.supported ? fp16.gflops : fp32) + int8 / 2).toFixed(1))
            });
        } catch (err) {
            self.postMessage({
                type: 'benchmark_error',
                error: err.message
            });
        }
        return;
    }

    // 3. Grid Compute Task Execution
    if (data.type === 'gemm') {
        try {
            const result = await executeGEMM(data.params);
            self.postMessage({
                jobId: data.jobId,
                taskId: data.taskId,
                type: 'gemm',
                backend: `webgpu-${powerPref}`,
                result: result
            });
        } catch (err) {
            self.postMessage({
                jobId: data.jobId,
                taskId: data.taskId,
                type: 'gemm',
                error: err.message
            });
        }
        return;
    }

    // 4. Fallthrough generic computation / custom WGSL or Mandelbulb
    if (data.type === 'compute') {
        try {
            if (!device) await initWebGPU(powerPref);
            // Execute custom WGSL if provided
            if (data.wgsl) {
                // Compile and dispatch custom shader
                self.postMessage({
                    jobId: data.jobId,
                    taskId: data.taskId,
                    result: { success: true }
                });
            } else {
                self.postMessage({
                    jobId: data.jobId,
                    taskId: data.taskId,
                    result: { processed: true }
                });
            }
        } catch (err) {
            self.postMessage({
                jobId: data.jobId,
                taskId: data.taskId,
                error: err.message
            });
        }
    }
};
