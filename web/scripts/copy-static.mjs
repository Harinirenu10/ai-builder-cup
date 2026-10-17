// Copies the static storefront (web/store) and the embeddable widget (web/widget) into dist/ after `vite build`.
import { cpSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dist = resolve(process.env.WEB_OUT_DIR || join(root, 'dist'));
const KEEP = new Set(['.html', '.css', '.js', '.svg', '.json']);

function copyDir(from, to, skip = []) {
  mkdirSync(to, { recursive: true });
  for (const name of readdirSync(from)) {
    const src = join(from, name);
    if (statSync(src).isDirectory()) copyDir(src, join(to, name), skip);
    else if (KEEP.has(extname(name)) && !skip.includes(name)) cpSync(src, join(to, name));
  }
}

copyDir(join(root, 'store'), join(dist, 'store'), ['truth.json', 'products.json']);
copyDir(join(root, 'widget'), join(dist, 'widget'));
console.log(`Copied store and widget into ${dist}`);
