/**
 * GridCompute CLI Demo
 * Demonstrates submitting a heavy compute task (Matrix Multiplication or Monte Carlo)
 * to GridCompute via the Node.js SDK and tracking live progress.
 *
 * Usage:
 *   node demo_cli.js [gemm | monte_carlo | benchmark] [scale (1-16, default: 1)]
 */

const GridClient = require('./GridClient');

async function runDemo() {
    const client = new GridClient('http://localhost:8080');
    const workloadType = process.argv[2] || 'gemm';

    console.log('====================================================');
    console.log('       GridCompute Distributed CLI Client           ');
    console.log('====================================================\n');

    try {
        // 1. Query cluster telemetry
        console.log('🔍 Querying cluster status...');
        const stats = await client.getGridStats();
        console.log(`📡 Online Nodes:        ${stats.onlineNodes}`);
        console.log(`⚙️  Active Workers:       ${stats.totalWorkers}`);
        console.log(`⚡ Cluster Total Power:  ${stats.totalGflops} GFLOPS`);
        console.log(`   ├─ FP32 Compute:      ${stats.totalFP32Gflops} GFLOPS`);
        console.log(`   ├─ FP16 Half-Prec:    ${stats.totalFP16Gflops} GFLOPS`);
        console.log(`   └─ INT8 AI Quantized: ${stats.totalINT8Gops} GOPS\n`);

        if (stats.onlineNodes === 0) {
            console.log('⚠️  Notice: No worker nodes are currently connected.');
            console.log('👉 Open http://localhost:8080 in a browser tab to connect as a computing node!\n');
        }

        if (workloadType === 'benchmark') {
            const scaleArg = Math.max(1, Math.min(16, Number(process.argv[3]) || 1));
            const iterations = Math.max(40, Math.round(640 * (scaleArg / 16)));
            console.log(`⚡ Submitting ${scaleArg}x Scaled Distributed GFLOPS Benchmark across Grid Nodes (Default: 1x)...`);
            const benchScript = `
                const N = 3200000;
                const a = new Float32Array(N);
                const b = new Float32Array(N);
                for (let i = 0; i < N; i++) { a[i] = i * 0.001 + 0.1; b[i] = 1.001; }
                const tStart = performance.now();
                for (let iter = 0; iter < ${iterations}; iter++) {
                    for (let i = 0; i < N; i += 4) {
                        a[i]   = a[i]   * b[i]   + 0.0001;
                        a[i+1] = a[i+1] * b[i+1] + 0.0001;
                        a[i+2] = a[i+2] * b[i+2] + 0.0001;
                        a[i+3] = a[i+3] * b[i+3] + 0.0001;
                    }
                }
                const elapsed = performance.now() - tStart;
                const totalOps = ${iterations} * N * 2;
                const gflops = (totalOps / (Math.max(1, elapsed) / 1000)) / 1e9;
                return { gflops: parseFloat(gflops.toFixed(2)), durationMs: parseFloat(elapsed.toFixed(2)), totalOps, scale: ${scaleArg} };
            `;
            const tasks = Array.from({ length: 16 }, (_, i) => ({ taskId: `stress_bench_${i}`, params: {} }));
            const startTime = Date.now();
            const result = await client.submitJob({ type: 'custom', script: benchScript, tasks }, (progress) => {
                process.stdout.write(`\r⏳ Progress: ${progress.percent}% (${progress.completed}/${progress.total} ${scaleArg}x stress chunks resolved)`);
            });

            console.log('\n\n====================================================');
            console.log(`🎉 ${scaleArg}x Scaled Benchmark Completed Across Grid!`);
            console.log(`⏱️  Total Duration:     ${((Date.now() - startTime) / 1000).toFixed(2)}s`);
            console.log(`⚡ Total Operations:   ${tasks.length} tasks (~${((tasks.length * iterations * 3200000 * 2) / 1e9).toFixed(2)} Billion operations)`);
            console.log('====================================================\n');

        } else if (workloadType === 'monte_carlo') {
            console.log('🎯 Submitting Distributed Monte Carlo Pi Estimation (10M samples)...');
            const startTime = Date.now();
            const result = await client.submitMonteCarlo({
                totalSamples: 10000000,
                chunkSize: 500000
            }, (progress) => {
                process.stdout.write(`\r⏳ Progress: ${progress.percent}% (${progress.completed}/${progress.total} chunks completed)`);
            });

            console.log('\n\n====================================================');
            console.log('🎉 Job Completed Successfully!');
            console.log(`⏱️  Total Duration:     ${((Date.now() - startTime) / 1000).toFixed(2)}s`);
            console.log(`🎯 Estimated π:        ${result.estimatedPi}`);
            console.log(`🎯 True π:             ${Math.PI}`);
            console.log(`🎯 Absolute Error:     ${Math.abs(Math.PI - result.estimatedPi).toExponential(4)}`);
            console.log('====================================================\n');

        } else {
            // GEMM Matrix Multiplication
            const M = 512, K = 512, N = 512;
            console.log(`📐 Preparing Matrix Multiplication A(${M}x${K}) * B(${K}x${N})...`);
            const matrixA = new Array(M * K);
            const matrixB = new Array(K * N);
            for (let i = 0; i < matrixA.length; i++) matrixA[i] = (i % 10) * 0.1;
            for (let i = 0; i < matrixB.length; i++) matrixB[i] = (i % 7) * 0.1;

            const totalOps = 2 * M * K * N;
            console.log(`🚀 Submitting GEMM workload (~${(totalOps / 1e6).toFixed(1)}M operations) in 32-row tiles...`);

            const startTime = Date.now();
            const result = await client.submitGEMM({
                matrixA,
                matrixB,
                M, K, N,
                tileRows: 32
            }, (progress) => {
                process.stdout.write(`\r⏳ Progress: ${progress.percent}% (${progress.completed}/${progress.total} tiles resolved)`);
            });

            const elapsedSec = Math.max(0.01, (Date.now() - startTime) / 1000);
            const achievedGflops = ((totalOps / elapsedSec) / 1e9).toFixed(2);

            console.log('\n\n====================================================');
            console.log('🎉 Matrix Multiplication Completed & Assembled!');
            console.log(`⏱️  Elapsed Time:       ${elapsedSec.toFixed(2)} seconds`);
            console.log(`⚡ Effective Throughput: ${achievedGflops} GFLOPS`);
            console.log(`📐 Output Matrix C:    [${result.matrixC.slice(0, 4).map(v => v.toFixed(3)).join(', ')}...]`);
            console.log('====================================================\n');
        }

    } catch (err) {
        console.error('\n❌ Execution Error:', err.message);
        process.exit(1);
    }
}

runDemo();
