import fs from 'fs';
import path from 'path';

const basePath = path.join('static', 'data', 'text-comparison');
const modelDirs = fs.readdirSync(basePath, { withFileTypes: true })
  .filter((dirent) => dirent.isDirectory())
  .map((dirent) => dirent.name);

const modelsJsonPath = path.join(basePath, 'models.json');
fs.writeFileSync(modelsJsonPath, JSON.stringify(modelDirs, null, 2));
console.log(`✅ models.json generated with ${modelDirs.length} models.`);

for (const model of modelDirs) {
  const modelPath = path.join(basePath, model);
  const files = fs.readdirSync(modelPath)
    .filter(name => name.endsWith('.txt'));
  const outPath = path.join(modelPath, 'files.json');
  fs.writeFileSync(outPath, JSON.stringify(files, null, 2));
  console.log(`✅ ${model}/files.json generated with ${files.length} files.`);
}
