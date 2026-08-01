const http = require('http');

/**
 * GridClient SDK for submitting and managing distributed tasks on GridTorrent
 */
class GridClient {
    /**
     * @param {string} serverUrl - The base URL of the GridTorrent coordinator (default: http://localhost:8080)
     */
    constructor(serverUrl = 'http://localhost:8080') {
        this.serverUrl = serverUrl;
    }

    /**
     * Submit a distributed computing job to the grid and await the results.
     * @param {Object} options
     * @param {string} options.script - The JavaScript code to be compiled & run by worker nodes.
     * @param {Array<Object>} options.tasks - Array of tasks, each containing taskId and params.
     * @param {number} [options.redundancy=1] - Redundancy target N (1-5) for majority consensus verification.
     * @param {string} [options.type='custom'] - The job type identifier.
     * @returns {Promise<Object>} The aggregated results mapped by taskId.
     */
    submitJob({ type = 'custom', script, tasks, redundancy = 1 }) {
        return new Promise((resolve, reject) => {
            const payload = JSON.stringify({ type, script, tasks, redundancy });
            const url = new URL(this.serverUrl + '/api/submit-job');

            const req = http.request(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': Buffer.byteLength(payload)
                }
            }, (res) => {
                let body = '';
                res.on('data', c => body += c);
                res.on('end', () => {
                    if (res.statusCode !== 202) {
                        return reject(new Error(`Failed to submit job (Status ${res.statusCode}): ${body}`));
                    }
                    try {
                        const { jobId } = JSON.parse(body);
                        this._pollJob(jobId, resolve, reject);
                    } catch (e) {
                        reject(new Error(`Failed to parse job submission response: ${e.message}`));
                    }
                });
            });

            req.on('error', reject);
            req.write(payload);
            req.end();
        });
    }

    /**
     * Internal polling mechanism for jobs.
     * @private
     */
    _pollJob(jobId, resolve, reject) {
        const url = new URL(this.serverUrl + `/api/job-status/${jobId}`);
        const interval = setInterval(() => {
            http.get(url, (res) => {
                let body = '';
                res.on('data', c => body += c);
                res.on('end', () => {
                    try {
                        const status = JSON.parse(body);
                        if (status.status === 'completed') {
                            clearInterval(interval);
                            resolve(status.results);
                        } else if (status.status === 'failed') {
                            clearInterval(interval);
                            reject(new Error(`Job ${jobId} failed`));
                        }
                    } catch (e) {
                        clearInterval(interval);
                        reject(new Error(`Failed to parse status response: ${e.message}`));
                    }
                });
            }).on('error', (err) => {
                clearInterval(interval);
                reject(err);
            });
        }, 1000);
    }
}

module.exports = GridClient;
