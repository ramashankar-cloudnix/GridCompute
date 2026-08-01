const { execSync } = require('child_process');

const commitMessage = `feat: implement distributed compute grid API, WebGL failover, and task redundancy consensus

- Add REST API endpoints (/api/submit-job, /api/job-status, /api/crack) for submitting and tracking general-purpose compute jobs.
- Implement WebGL context loss detection and automatic failover to CPU workers in index.html.
- Add task redundancy (N=1..5) with node-avoidance scheduling and majority consensus validation in server.js.
- Add dynamic JavaScript script execution engine using async function runner in worker.js.
- Add GridClient Node.js SDK for programmatically submitting and polling jobs.
- Add real-time visual progress monitoring for SHA-256 brute force jobs on dashboard.`;

try {
    console.log('Staging files...');
    execSync('git add .');
    
    console.log('Committing changes...');
    const result = execSync(`git commit -m "${commitMessage.replace(/"/g, '\\"')}"`).toString();
    console.log('\nGit Commit Output:\n', result);
} catch (e) {
    console.error('\nGit Error:', e.message);
    if (e.stdout) console.log('stdout:', e.stdout.toString());
    if (e.stderr) console.error('stderr:', e.stderr.toString());
}

