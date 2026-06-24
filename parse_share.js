const fs = require('fs');

const htmlPath = '/mnt/c/Users/ramas/.gemini/antigravity-ide/brain/0fd2bdf9-da5b-44c3-81a4-3dafd9659941/.system_generated/steps/4/content.md';
const outputPath = 'share_text.md';
const errorPath = 'error.txt';

try {
    if (!fs.existsSync(htmlPath)) {
        throw new Error(`Input file does not exist at: ${htmlPath}`);
    }
    const content = fs.readFileSync(htmlPath, 'utf8');
    
    // Look for global data
    const match = content.match(/window\.WIZ_global_data\s*=\s*(\{[\s\S]*?\});/);
    let extracted = [];
    
    if (match) {
        try {
            const data = JSON.parse(match[1]);
            function recurse(obj) {
                if (typeof obj === 'string') {
                    if (obj.length > 20) {
                        extracted.push(obj);
                    }
                } else if (Array.isArray(obj)) {
                    obj.forEach(recurse);
                } else if (obj && typeof obj === 'object') {
                    Object.values(obj).forEach(recurse);
                }
            }
            recurse(data);
        } catch (e) {
            extracted.push(`JSON Parse error: ${e.message}`);
        }
    }
    
    // Also pull out divs with class message-content or any HTML-like paragraphs
    const messages = content.match(/<div[^>]*class="[^"]*message-content[^"]*"[^>]*>([\s\S]*?)<\/div>/g);
    if (messages) {
        messages.forEach(msg => {
            const clean = msg.replace(/<[^>]+>/g, ' ').trim();
            if (clean.length > 20) {
                extracted.push(clean);
            }
        });
    }
    
    // Deduplicate and filter strings
    const unique = [...new Set(extracted)];
    
    fs.writeFileSync(outputPath, '# Parsed Shared Content\n\n' + unique.map((txt, i) => `### Item ${i+1}\n${txt}\n\n`).join(''));
    fs.writeFileSync(errorPath, 'Success!');
} catch (err) {
    fs.writeFileSync(errorPath, `Error: ${err.message}\nStack: ${err.stack}`);
}
