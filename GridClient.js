const http = require('http');

/**
 * GridClient SDK for submitting and managing distributed tasks on GridCompute
 */
class GridClient {
    /**
     * @param {string} serverUrl - The base URL of the GridCompute coordinator (default: http://localhost:8080)
     */
    constructor(serverUrl = 'http://localhost:8080') {
        this.serverUrl = serverUrl;
    }

    /**
     * Internal JSON request helper
     * @private
     */
    _request(path, method = 'GET', body = null) {
        return new Promise((resolve, reject) => {
            const url = new URL(this.serverUrl + path);
            const payload = body ? JSON.stringify(body) : null;
            const headers = {};
            if (payload) {
                headers['Content-Type'] = 'application/json';
                headers['Content-Length'] = Buffer.byteLength(payload);
            }

            const req = http.request(url, { method, headers }, (res) => {
                let data = '';
                res.on('data', c => data += c);
                res.on('end', () => {
                    if (res.statusCode < 200 || res.statusCode >= 300) {
                        return reject(new Error(`HTTP ${res.statusCode}: ${data}`));
                    }
                    try {
                        resolve(JSON.parse(data));
                    } catch (e) {
                        resolve(data);
                    }
                });
            });

            req.on('error', reject);
            if (payload) req.write(payload);
            req.end();
        });
    }

    /**
     * Get real-time cluster statistics and aggregate GFLOPS
     */
    getGridStats() {
        return this._request('/api/v1/grid/stats');
    }

    /**
     * Get list of all connected nodes and their hardware telemetry
     */
    getGridNodes() {
        return this._request('/api/v1/grid/nodes');
    }

    /**
     * Submit a distributed Matrix Multiplication (GEMM) job to the grid
     * @param {Object} options
     * @param {Array<number>} options.matrixA - Flat array of size M x K
     * @param {Array<number>} options.matrixB - Flat array of size K x N
     * @param {number} options.M - Rows in A
     * @param {number} options.K - Columns in A / Rows in B
     * @param {number} options.N - Columns in B
     * @param {number} [options.tileRows=32] - Number of rows computed per chunk
     * @param {Function} [onProgress] - Optional callback receiving progress updates
     */
    async submitGEMM({ matrixA, matrixB, M, K, N, tileRows = 32 }, onProgress = null) {
        const payload = { type: 'gemm', matrixA, matrixB, M, K, N, tileRows };
        const res = await this._request('/api/v1/jobs', 'POST', payload);
        return this._pollJob(res.jobId, onProgress);
    }

    /**
     * Submit a distributed Monte Carlo simulation (e.g. estimating Pi)
     * @param {Object} options
     * @param {number} [options.totalSamples=10000000]
     * @param {number} [options.chunkSize=500000]
     * @param {Function} [onProgress]
     */
    async submitMonteCarlo({ totalSamples = 10000000, chunkSize = 500000 }, onProgress = null) {
        const payload = { type: 'monte_carlo', totalSamples, chunkSize };
        const res = await this._request('/api/v1/jobs', 'POST', payload);
        return this._pollJob(res.jobId, onProgress);
    }

    /**
     * Submit a generic distributed computing job
     * @param {Object} options
     * @param {string} [options.type='custom']
     * @param {string} options.script - Async JavaScript function body
     * @param {Array<Object>} options.tasks - Array of tasks [{ taskId, params }]
     * @param {number} [options.redundancy=1] - Fault tolerance / majority consensus (1-5)
     * @param {Function} [onProgress]
     */
    async submitJob({ type = 'custom', script, tasks, redundancy = 1, imports = [] }, onProgress = null) {
        const payload = { type, script, tasks, redundancy, imports };
        const res = await this._request('/api/v1/jobs', 'POST', payload);
        return this._pollJob(res.jobId, onProgress);
    }

    /**
     * Poll job status until completion
     * @private
     */
    _pollJob(jobId, onProgress = null) {
        return new Promise((resolve, reject) => {
            const interval = setInterval(async () => {
                try {
                    const status = await this._request(`/api/v1/jobs/${jobId}/results`);
                    if (onProgress && typeof onProgress === 'function') {
                        onProgress({
                            jobId,
                            status: status.status,
                            completed: status.completedCount,
                            total: status.totalCount,
                            percent: Math.round((status.completedCount / status.totalCount) * 100)
                        });
                    }

                    if (status.status === 'completed') {
                        clearInterval(interval);
                        resolve(status.assembledResult || status.rawResults);
                    } else if (status.status === 'failed' || status.status === 'cancelled') {
                        clearInterval(interval);
                        reject(new Error(`Job ${jobId} ended with status: ${status.status}`));
                    }
                } catch (e) {
                    clearInterval(interval);
                    reject(e);
                }
            }, 800);
        });
    }
}

module.exports = GridClient;
