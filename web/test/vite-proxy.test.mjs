import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

// Regression coverage for: GET /v1/me/twin (and other /v1/* endpoints) returning Vite's own
// index.html with HTTP 200 instead of reaching the backend. In the dev server, ApiClient builds
// a relative URL (getApiBaseUrl() returns '' on port 5173, see src/lib/api.ts), trusting that
// Vite's dev proxy forwards every path prefix the app actually calls to the real API on
// :8000. vite.config.ts only proxied '/api', not '/v1' -- so any '/v1/...' request fell through
// to Vite's SPA fallback, which answers unmatched GETs with index.html (200, text/html). Because
// that is a 200 with a text/* content-type, ApiClient.request() treats it as a *successful*
// response (see api.test.mjs's sibling repro) and returns the HTML string as the "twin", instead
// of throwing -- so the frontend's error-handling path (ApiError 404/409 => null, or the
// postJobTwinFetchFailed retry UI) never engages either. The job itself completes fine on the
// backend (hence worker logs show privacy cleanup finishing), but the Twin Viewer falls through
// to the generic "No 3D Twin Yet" state because twinData.glb_url is undefined.
//
// Fix: proxy '/v1' alongside '/api' in vite.config.ts. This test scans the actual
// apiClient.get/post/put/patch/delete(...) call sites for their literal path prefixes and
// asserts every one of them is covered by vite.config.ts's dev proxy, so a future endpoint added
// under a new, unproxied prefix fails this test instead of silently breaking the dev UI.

const WEB_ROOT = new URL('..', import.meta.url).pathname.replace(/^\/([A-Za-z]):/, '$1:');
const SRC_DIR = join(WEB_ROOT, 'src');

function listFilesRecursive(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) {
      out.push(...listFilesRecursive(full));
    } else if (/\.(ts|tsx)$/.test(entry)) {
      out.push(full);
    }
  }
  return out;
}

function extractApiClientPathPrefixes() {
  const prefixes = new Set();
  const callRe = /apiClient\.(get|post|put|patch|delete)(?:<[^>]*>)?\(\s*[`'"]([^`'"]+)[`'"]/g;
  for (const file of listFilesRecursive(SRC_DIR)) {
    const text = readFileSync(file, 'utf-8');
    let match;
    while ((match = callRe.exec(text)) !== null) {
      const path = match[2];
      if (!path.startsWith('/')) continue; // skip absolute http(s) URLs
      const prefix = '/' + path.slice(1).split('/')[0];
      prefixes.add(prefix);
    }
  }
  return prefixes;
}

test('every path prefix called through apiClient is covered by the Vite dev proxy', async () => {
  const { default: viteConfig } = await import('../vite.config.ts');
  const proxyKeys = Object.keys(viteConfig.server.proxy);
  assert.ok(proxyKeys.length > 0, 'expected vite.config.ts to declare at least one dev proxy rule');

  const usedPrefixes = extractApiClientPathPrefixes();
  assert.ok(usedPrefixes.size > 0, 'expected to find at least one apiClient call site to check');

  const uncovered = [...usedPrefixes].filter(
    (prefix) => !proxyKeys.some((key) => prefix === key || prefix.startsWith(key + '/') || prefix === key)
  );
  assert.deepEqual(
    uncovered,
    [],
    `these apiClient path prefixes are called by the app but not proxied in vite.config.ts, so ` +
      `they would hit Vite's own SPA fallback (200 text/html) instead of the backend in dev: ${uncovered.join(', ')}`
  );
});

test('vite dev proxy forwards both /api and /v1 to the backend on :8000', async () => {
  const { default: viteConfig } = await import('../vite.config.ts');
  const proxy = viteConfig.server.proxy;
  for (const prefix of ['/api', '/v1']) {
    assert.ok(proxy[prefix], `expected a proxy rule for "${prefix}"`);
    assert.equal(proxy[prefix].target, 'http://localhost:8000');
  }
});
