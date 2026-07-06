const fs = require('fs');
const path = require('path');
const { DefaultArtifactClient } = require('@actions/artifact');

async function main() {
  const artifactName = process.argv[2];
  const distDir = process.argv[3] || 'dist';
  const retentionDays = Number(process.argv[4] || '14');

  if (!artifactName) {
    throw new Error('artifact name is required');
  }

  const absDist = path.resolve(distDir);
  if (!fs.existsSync(absDist)) {
    throw new Error(`dist directory not found: ${absDist}`);
  }

  const wheelFiles = fs
    .readdirSync(absDist)
    .filter((f) => f.endsWith('.whl'))
    .map((f) => path.join(absDist, f));

  if (wheelFiles.length === 0) {
    throw new Error(`no wheel files found in ${absDist}`);
  }

  const artifactClient = new DefaultArtifactClient();
  const result = await artifactClient.uploadArtifact(
    artifactName,
    wheelFiles,
    absDist,
    { retentionDays }
  );

  console.log(`uploaded artifact: ${artifactName}`);
  console.log(`artifact id: ${result.id}`);
  console.log(`artifact size: ${result.size}`);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});
