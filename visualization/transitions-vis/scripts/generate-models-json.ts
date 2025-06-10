/// <reference types="node" />
import fs from 'fs';
import path from 'path';

const dataDir = path.join('static', 'data', 'iteration-label-transition');
const outputFile = path.join('static', 'data', 'iteration-label-transition', 'models.json');

const files = fs.readdirSync(dataDir);
const models = files
  .filter((f) => f.endsWith('.csv'))
  .map((f) => path.basename(f, '.csv'));

fs.writeFileSync(outputFile, JSON.stringify({ models }, null, 2));

console.log(`✅ models.json generated with ${models.length} models.`);
