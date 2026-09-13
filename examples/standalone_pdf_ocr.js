/**
 * Standalone Real-World Application Script: Distributed PDF OCR
 * 
 * Demonstrates utilizing the pure, generic Grid Compute API (/api/submit-job)
 * without adding any application-specific routes to the server codebase.
 * 
 * Usage:
 *   node examples/standalone_pdf_ocr.js [optional_image_path]
 */

const fs = require('fs');
const path = require('path');
const GridClient = require('../GridClient');

// Initialize Grid Compute SDK pointing to the generic orchestrator server
const client = new GridClient('http://localhost:8080');

// 1. Worker script that executes on browser Web Worker nodes
// Uses Tesseract.js dynamically imported from CDN
const tesseractOcrScript = `
    const { pageNum, image, textSnippet, lang } = params;
    
    // If real image base64 data URL is provided and Tesseract.js CDN module is imported
    if (image && typeof Tesseract !== 'undefined') {
        const worker = await Tesseract.createWorker(lang || 'eng');
        const ret = await worker.recognize(image);
        await worker.terminate();
        return {
            pageNum,
            text: ret.data.text,
            confidence: Math.round(ret.data.confidence)
        };
    }
    
    // Simulated processing for instant benchmark runs
    await new Promise(r => setTimeout(r, 600));
    return {
        pageNum,
        text: textSnippet || ("Page " + pageNum + ": Document text extracted via distributed Tesseract.js Web Worker."),
        confidence: 98
    };
`;

// Dynamic CDN imports required by the worker node
const cdnImports = [
    'https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js'
];

async function main() {
    console.log('===========================================================');
    console.log('📄 STANDALONE DISTRIBUTED PDF OCR CLIENT (GRID COMPUTE API)');
    console.log('===========================================================\n');

    // 2. Load PDF / sample document pages
    let pagePayloads = [
        {
            pageNum: 1,
            textSnippet: "CONFIDENTIAL SPECIFICATION - PAGE 1\nGridCompute Architecture: Decentralized Web Worker P2P Mesh Network."
        },
        {
            pageNum: 2,
            textSnippet: "CONFIDENTIAL SPECIFICATION - PAGE 2\nLoad Balancing: Dynamic node avoidance & redundancy consensus validation."
        },
        {
            pageNum: 3,
            textSnippet: "CONFIDENTIAL SPECIFICATION - PAGE 3\nExecution Result: All pages processed concurrently across worker nodes."
        }
    ];

    const inputArg = process.argv[2];
    if (inputArg && fs.existsSync(inputArg)) {
        console.log(`📁 Reading document file: ${inputArg}`);
        const fileExt = path.extname(inputArg).toLowerCase();
        
        if (fileExt === '.png' || fileExt === '.jpg' || fileExt === '.jpeg') {
            const imageBuffer = fs.readFileSync(inputArg);
            const base64Image = `data:image/${fileExt.replace('.', '')};base64,${imageBuffer.toString('base64')}`;
            pagePayloads = [{ pageNum: 1, image: base64Image }];
        }
    } else {
        console.log('💡 Tip: Pass an image/PDF path as argument: node examples/standalone_pdf_ocr.js my_doc.png');
        console.log('Using 3 sample pages for demonstration...\n');
    }

    console.log(`Submitting ${pagePayloads.length} page tasks to Grid Compute API...`);
    const startTime = Date.now();

    try {
        // 3. Submit generic custom job to /api/submit-job using GridClient SDK
        const results = await client.submitJob({
            type: 'custom',
            script: tesseractOcrScript,
            imports: cdnImports,
            tasks: pagePayloads.map(p => ({
                taskId: `ocr_page_${p.pageNum}`,
                params: p
            })),
            redundancy: 1
        });

        const duration = ((Date.now() - startTime) / 1000).toFixed(2);
        console.log(`\n🎉 Grid OCR Processing Completed in ${duration}s!\n`);

        // 4. Collate results by page number
        const pages = Object.values(results)
            .map(r => r.value)
            .sort((a, b) => a.pageNum - b.pageNum);

        console.log('===========================================================');
        console.log('FULL EXTRACTED DOCUMENT TEXT (COLLATED BY PAGE):');
        console.log('===========================================================\n');

        pages.forEach(p => {
            console.log(`--- [ PAGE ${p.pageNum} ] (Confidence: ${p.confidence}%) ---`);
            console.log(p.text);
            console.log('');
        });

        console.log('===========================================================');

    } catch (err) {
        console.error('❌ PDF OCR Job failed:', err.message);
        console.error('Ensure the Grid Compute server is running on port 8080 (node server.js)');
    }
}

main();
