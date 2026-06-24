// GridTorrent Web Worker — 8x Complexity Mandelbrot Renderer
// Uses 4x4 Supersampling Anti-Aliasing (16 sub-pixel samples per pixel)
// maxIterations: 1000 for deep fractal detail

self.onmessage = function (e) {
    const task = e.data;

    // BUG 10 FIX: Wrap entire computation in try/catch.
    // If anything goes wrong, notify the main thread instead of dying silently.
    try {
        const { y, width, height, xmin, xmax, ymin, ymax, maxIterations } = task;

        // Validate task parameters
        if (
            typeof y !== 'number' || typeof width !== 'number' || typeof height !== 'number' ||
            typeof xmin !== 'number' || typeof xmax !== 'number' ||
            typeof ymin !== 'number' || typeof ymax !== 'number' ||
            typeof maxIterations !== 'number' || maxIterations < 1
        ) {
            self.postMessage({ error: 'Invalid task parameters', y: task.y ?? -1 });
            return;
        }

        const iterations = new Float32Array(width);
        // 4x4 SSAA: 4 sub-pixel offsets per axis
        const subOffsets = [-0.375, -0.125, 0.125, 0.375];

        for (let x = 0; x < width; x++) {
            let totalIter = 0;

            // 16 sub-pixel samples per pixel
            for (let sx = 0; sx < 4; sx++) {
                for (let sy = 0; sy < 4; sy++) {
                    const px = x + subOffsets[sx];
                    const py = y + subOffsets[sy];

                    // Map pixel to complex plane
                    const cx = xmin + (px / width) * (xmax - xmin);
                    const cy = ymin + (py / height) * (ymax - ymin);

                    // Optimised escape-time loop (pre-squared terms)
                    let zr = 0.0, zi = 0.0;
                    let zr2 = 0.0, zi2 = 0.0;
                    let iter = 0;

                    while (zr2 + zi2 <= 4.0 && iter < maxIterations) {
                        zi = 2.0 * zr * zi + cy;
                        zr = zr2 - zi2 + cx;
                        zr2 = zr * zr;
                        zi2 = zi * zi;
                        iter++;
                    }

                    totalIter += iter;
                }
            }

            // Store averaged SSAA result
            iterations[x] = totalIter / 16.0;
        }

        self.postMessage({
            y: y,
            iterations: Array.from(iterations)
        });

    } catch (err) {
        // Report error back to main thread — node can then re-register or log
        self.postMessage({
            error: err.message || 'Worker computation error',
            y: (e.data && typeof e.data.y === 'number') ? e.data.y : -1
        });
    }
};
