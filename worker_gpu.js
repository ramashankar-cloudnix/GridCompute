// GridTorrent GPU Web Worker — WebGL2 Mandelbulb 3D Raymarcher
// Renders a strip of pixels using an SDF sphere-tracer with full lighting.
// Output: RGBA Uint8Array — pre-coloured, no JS-side mapping needed.

let gl = null, program = null, vao = null, uLoc = {}, texture = null, framebuffer = null;
let lastWidth = 0, lastChunkHeight = 0;

// ─── One-time WebGL2 initialisation ──────────────────────────────────────────
function initWebGL(width, chunkHeight) {
    try {
        const canvas = new OffscreenCanvas(width, chunkHeight);
        gl = canvas.getContext('webgl2', {
            antialias: false, depth: false, stencil: false,
            alpha: false, powerPreference: 'high-performance',
            preserveDrawingBuffer: false
        });
        if (!gl) throw new Error('WebGL2 not available');

        // ── Vertex Shader ─────────────────────────────────────────────────────
        const vsSource = `#version 300 es
        in vec2 a_position;
        void main() { gl_Position = vec4(a_position, 0.0, 1.0); }`;

        // ── Fragment Shader — Mandelbulb Raymarcher ───────────────────────────
        const fsSource = `#version 300 es
        precision highp float;
        precision highp int;

        uniform float u_yStart;
        uniform float u_chunkHeight;
        uniform float u_width;
        uniform float u_height;
        uniform float u_camTheta;     // azimuth
        uniform float u_camPhi;       // elevation
        uniform float u_camDist;      // distance from origin
        uniform int   u_power;        // Mandelbulb exponent
        uniform int   u_maxSteps;     // ray march step budget

        out vec4 outColor;

        // ── Mandelbulb Distance Estimator ─────────────────────────────────────
        float sdf(vec3 p) {
            vec3 z = p;
            float dr = 1.0;
            float r = 0.0;
            for (int i = 0; i < 15; i++) {
                r = length(z);
                if (r > 2.0) break;
                if (r < 0.00001) break; // Prevent divide-by-zero / NaN at origin

                float theta = acos(clamp(z.z / r, -1.0, 1.0));
                float phi = atan(z.y, z.x);
                dr = pow(r, float(u_power) - 1.0) * float(u_power) * dr + 1.0;

                float zr = pow(r, float(u_power));
                theta = theta * float(u_power);
                phi = phi * float(u_power);

                z = zr * vec3(sin(theta)*cos(phi),
                              sin(phi)*sin(theta),
                              cos(theta));
                z += p;
            }
            if (r < 0.00001) return 0.0; // Safe distance at origin
            return 0.5 * log(r) * r / dr;
        }


        // ── Surface normal via central differences ────────────────────────────
        vec3 calcNormal(vec3 p) {
            const float e = 0.0005;
            return normalize(vec3(
                sdf(p + vec3(e,0,0)) - sdf(p - vec3(e,0,0)),
                sdf(p + vec3(0,e,0)) - sdf(p - vec3(0,e,0)),
                sdf(p + vec3(0,0,e)) - sdf(p - vec3(0,0,e))
            ));
        }

        // ── Ambient occlusion (5 samples along normal) ────────────────────────
        float calcAO(vec3 pos, vec3 nor) {
            float occ = 0.0, sca = 1.0;
            for (int i = 0; i < 5; i++) {
                float h = 0.01 + 0.12 * float(i) / 4.0;
                occ += (h - sdf(pos + nor * h)) * sca;
                sca *= 0.95;
            }
            return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
        }

        // ── Soft shadow ───────────────────────────────────────────────────────
        float softShadow(vec3 ro, vec3 rd, float mint, float maxt) {
            float res = 1.0, t = mint;
            for (int i = 0; i < 20; i++) {
                float h = sdf(ro + rd * t);
                res = min(res, 10.0 * h / t);
                t  += clamp(h, 0.005, 0.1);
                if (res < 0.005 || t > maxt) break;
            }
            return clamp(res, 0.0, 1.0);
        }

        // ── Camera setup from spherical coordinates ───────────────────────────
        mat3 buildCamera(vec3 ro, vec3 target) {
            vec3 cw = normalize(target - ro);
            vec3 cu = normalize(cross(cw, vec3(0.0, 1.0, 0.0)));
            vec3 cv = cross(cu, cw);
            return mat3(cu, cv, cw);
        }

        void main() {
            // Map gl_FragCoord (bottom-up GL) to canvas coordinates (top-down)
            // gl.readPixels reads bottom-up, so gl_FragCoord.y=0.5 becomes array index 0.
            // Canvas ImageData expects array index 0 to be the top row of the chunk.
            // Therefore gl_FragCoord.y=0.5 must compute the top row (u_yStart + 0).
            float localRow = gl_FragCoord.y - 0.5;
            float canvasY  = u_yStart + localRow;
            float canvasX  = gl_FragCoord.x - 0.5;

            // Build camera from spherical params
            float cosT = cos(u_camTheta), sinT = sin(u_camTheta);
            float cosP = cos(u_camPhi),   sinP = sin(u_camPhi);
            vec3 ro     = u_camDist * vec3(sinT*cosP, sinP, cosT*cosP);
            mat3 cam    = buildCamera(ro, vec3(0.0));

            float aspect = u_width / u_height;
            vec3  finalColor = vec3(0.0);

            // Single sample per pixel (no SSAA) to prevent TDR timeouts on complex chunks
            float uvx = (canvasX / u_width * 2.0 - 1.0) * aspect;
            float uvy = 1.0 - (canvasY / u_height * 2.0);

            vec3 rd = normalize(cam * vec3(uvx, uvy, 1.8));

            // ── Sphere-trace ──────────────────────────────────────────
            float t    = 0.0;
            bool  hit  = false;
            int   step;
            for (step = 0; step < u_maxSteps; step++) {
                float h = sdf(ro + rd * t);
                if (h < 0.0004) { hit = true; break; }
                if (t > 6.0)    break;
                t += h;
            }

            vec3 col;
            if (hit) {
                vec3 pos = ro + rd * t;
                vec3 nor = calcNormal(pos);
                vec3 lig = normalize(vec3(0.5, 0.8, 0.4));

                float dif = clamp(dot(nor, lig), 0.0, 1.0);
                float sha = softShadow(pos + nor * 0.001, lig, 0.01, 4.0);
                float ao  = calcAO(pos, nor);
                float rim = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 3.0);
                float dep = float(step) / float(u_maxSteps); // iteration depth

                // Neon emerald/teal/purple palette matching UI theme
                vec3 deepCol = vec3(0.04, 0.35, 0.30);  // deep teal
                vec3 highCol = vec3(0.40, 0.08, 0.50);  // purple highlight
                vec3 baseCol = mix(deepCol, highCol, clamp(nor.y * 0.5 + 0.5, 0.0, 1.0));

                col  = baseCol * (0.25 * ao + 0.75 * dif * sha);
                col += 0.35 * rim * vec3(0.25, 0.95, 0.75) * sha * ao; // rim glow
                col += 0.08 * ao * vec3(0.08, 0.18, 0.28);             // sky bounce

                // Subtle iteration-depth tinting for extra detail
                col  = mix(col, vec3(0.0, 0.6, 0.5) * col, dep * 0.4);
                col  = pow(max(col, vec3(0.0)), vec3(0.4545));          // gamma
            } else {
                // Deep space background — dark navy gradient
                float t2 = 0.5 * (rd.y + 1.0);
                col = mix(vec3(0.01, 0.02, 0.06), vec3(0.0, 0.01, 0.03), t2);
                // Subtle star field
                float star = step(0.9998, fract(sin(dot(rd*200.0, vec3(12.9898, 78.233, 45.164))) * 43758.5453));
                col += star * 0.4 * vec3(0.8, 0.9, 1.0);
            }

            finalColor = col;
            outColor = vec4(clamp(finalColor, 0.0, 1.0), 1.0);
        }`;

        // ── Compile & link ────────────────────────────────────────────────────
        function compileShader(type, src) {
            const sh = gl.createShader(type);
            gl.shaderSource(sh, src);
            gl.compileShader(sh);
            if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS))
                throw new Error('Shader compile error: ' + gl.getShaderInfoLog(sh));
            return sh;
        }

        const vs = compileShader(gl.VERTEX_SHADER,   vsSource);
        const fs = compileShader(gl.FRAGMENT_SHADER, fsSource);
        program  = gl.createProgram();
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS))
            throw new Error('Link error: ' + gl.getProgramInfoLog(program));

        // ── Cache uniform locations ───────────────────────────────────────────
        uLoc = {
            yStart:      gl.getUniformLocation(program, 'u_yStart'),
            chunkHeight: gl.getUniformLocation(program, 'u_chunkHeight'),
            width:       gl.getUniformLocation(program, 'u_width'),
            height:      gl.getUniformLocation(program, 'u_height'),
            camTheta:    gl.getUniformLocation(program, 'u_camTheta'),
            camPhi:      gl.getUniformLocation(program, 'u_camPhi'),
            camDist:     gl.getUniformLocation(program, 'u_camDist'),
            power:       gl.getUniformLocation(program, 'u_power'),
            maxSteps:    gl.getUniformLocation(program, 'u_maxSteps')
        };

        // ── VAO + fullscreen quad ─────────────────────────────────────────────
        vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            -1,-1,  1,-1,  -1,1,
            -1, 1,  1,-1,   1,1
        ]), gl.STATIC_DRAW);
        const aPos = gl.getAttribLocation(program, 'a_position');
        gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
        gl.bindVertexArray(null);

        // ── RGBA framebuffer (UNSIGNED_BYTE — wide device support) ────────────
        texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, chunkHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        framebuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
            throw new Error('Framebuffer incomplete');

        lastWidth       = width;
        lastChunkHeight = chunkHeight;

    } catch (err) {
        console.warn('GPU Worker Init Failed:', err.message);
        gl = null;
    }
}

// ─── Resize texture if chunk size changes (e.g. last partial chunk) ───────────
function ensureTextureSize(width, chunkHeight) {
    if (width === lastWidth && chunkHeight === lastChunkHeight) return;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, chunkHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    lastWidth       = width;
    lastChunkHeight = chunkHeight;
}

// ─── Main message handler ─────────────────────────────────────────────────────
self.onmessage = function (e) {
    const task = e.data;
    try {
        const { yStart, chunkHeight, width, height, camTheta, camPhi, camDist, power, maxSteps } = task;

        if (!gl || gl.isContextLost()) {
            initWebGL(width, chunkHeight);
        }
        
        if (!gl || gl.isContextLost()) {
            // Context is temporarily lost or unsupported. Return an error for this block, 
            // but we will try again on the next task (browser may restore it shortly).
            self.postMessage({ error: 'WebGL context unavailable', yStart });
            return;
        }

        ensureTextureSize(width, chunkHeight);

        const tStart = performance.now();

        gl.useProgram(program);
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.viewport(0, 0, width, chunkHeight);

        gl.uniform1f(uLoc.yStart,      yStart);
        gl.uniform1f(uLoc.chunkHeight, chunkHeight);
        gl.uniform1f(uLoc.width,       width);
        gl.uniform1f(uLoc.height,      height);
        gl.uniform1f(uLoc.camTheta,    camTheta);
        gl.uniform1f(uLoc.camPhi,      camPhi);
        gl.uniform1f(uLoc.camDist,     camDist);
        gl.uniform1i(uLoc.power,       power);
        gl.uniform1i(uLoc.maxSteps,    Math.min(maxSteps, 120)); // GPU TDR safety cap

        gl.bindVertexArray(vao);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        gl.bindVertexArray(null);

        // Readback RGBA pixels (bottom-up GL order — shader already compensates)
        const pixels = new Uint8Array(width * chunkHeight * 4);
        gl.readPixels(0, 0, width, chunkHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        gl.flush();

        const renderMs = performance.now() - tStart;

        // 75% GPU duty cycle: sleep = renderTime × 0.333
        const delay = Math.max(1, Math.floor(renderMs * 0.333));

        setTimeout(() => {
            self.postMessage({ yStart, chunkHeight, pixels: Array.from(pixels) });
        }, delay);

    } catch (err) {
        self.postMessage({
            error:  err.message || 'GPU error',
            yStart: (e.data && typeof e.data.yStart === 'number') ? e.data.yStart : -1
        });
    }
};
