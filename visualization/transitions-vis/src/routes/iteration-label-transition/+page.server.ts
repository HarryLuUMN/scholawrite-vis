import fs from 'fs';
import path from 'path';

export async function load() {
  const dataDir = 'static/data/iteration-label-transition';
  const files: string[] = fs.readdirSync(dataDir);

  const models = files
    .filter((f: string) => f.endsWith('.csv'))
    .map((f: string) => path.basename(f, '.csv'));

  return { models };
}
