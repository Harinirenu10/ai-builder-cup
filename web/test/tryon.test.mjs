import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { buildUrl } from '../src/lib/api.ts';

const WEB_DIR = join(dirname(fileURLToPath(import.meta.url)), '..');
const REPO_ROOT = join(WEB_DIR, '..');

const customerApp = readFileSync(join(WEB_DIR, 'src/pages/CustomerApp.tsx'), 'utf8');
const geometryTryOn = readFileSync(join(REPO_ROOT, 'services/geometry/tryon.py'), 'utf8');
const apiTryOn = readFileSync(join(REPO_ROOT, 'services/api/apilib/tryon.py'), 'utf8');

const appGarmentIds = [...customerApp.matchAll(/const DEMO_TRYON_\w*_?ID = '([^']+)'/g)].map((m) => m[1]);
const specGarmentIds = [...geometryTryOn.matchAll(/garment_id="([^"]+)"/g)].map((m) => m[1]);

// Every demo garment id is written in three places (the Python spec, the API's shopper-facing
// copy, and the PWA's picker). A silent drift between them would show up only as a 404 in the
// demo, so pin them together here.
test('every demo try-on garment id matches across the PWA, the API and the geometry specs', () => {
  assert.equal(appGarmentIds.length, 2, 'the PWA should know the tee and the trousers');
  assert.deepEqual([...appGarmentIds].sort(), [...specGarmentIds].sort());
  for (const id of specGarmentIds) {
    assert.ok(apiTryOn.includes(`"${id}"`), `apilib/tryon.py should carry copy for ${id}`);
  }
});

test('the PWA requests each try-on GLB from the owner-only twin path', () => {
  // The component builds the path from whichever garment is selected.
  assert.ok(customerApp.includes('/tryon/${tryOnGarmentId}/glb'));
  assert.ok(apiTryOn.includes('/twin/{twin_id}/tryon/{garment_id}/glb'));

  const twinId = 'twin_abc123';
  for (const id of appGarmentIds) {
    const url = buildUrl(`/api/twin/${encodeURIComponent(twinId)}/tryon/${id}/glb`);
    assert.ok(url.endsWith(`/api/twin/${twinId}/tryon/${id}/glb`), url);
  }
});

test('the picker offers the body twin plus both garments, and body twin is always reachable', () => {
  const block = customerApp.match(/const TRY_ON_CHOICES[\s\S]*?\n\];/);
  assert.ok(block, 'CustomerApp.tsx should define TRY_ON_CHOICES');
  const choices = block[0];

  // Three options: the plain body (id null) and one per garment.
  assert.ok(choices.includes('id: null'), 'the plain body twin must be an option');
  for (const id of appGarmentIds) {
    assert.ok(choices.includes(id) || choices.includes('DEMO_TRYON'), `missing choice for ${id}`);
  }
  assert.equal((choices.match(/id:/g) || []).length, 3);

  // Switching is a plain set, not a toggle, so any option reaches any other -- including
  // back to the body twin.
  assert.ok(customerApp.includes('onClick={() => setTryOnGarmentId(choice.id)}'));
  // And the picker renders outside the loaded-model branch, so it stays usable while a
  // garment mesh is still being built.
  assert.ok(customerApp.indexOf('TRY_ON_CHOICES.map') < customerApp.indexOf('src={glbObjectUrl}'));
});

test('a new twin resets the viewer back to the plain body', () => {
  assert.ok(customerApp.includes('setTryOnGarmentId(null);'));
  assert.ok(customerApp.includes('}, [twinIdForTryOn]);'));
});

test('a twin id with URL-unsafe characters is escaped, not injected into the path', () => {
  assert.ok(
    customerApp.includes('encodeURIComponent(twinIdForTryOn)'),
    'the twin id should be encodeURIComponent-escaped before interpolation',
  );
  assert.equal(encodeURIComponent('twin/../other'), 'twin%2F..%2Fother');
});
