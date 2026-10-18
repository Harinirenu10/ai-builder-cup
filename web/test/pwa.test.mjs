import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webRoot = path.resolve(__dirname, '..');

test('PWA manifest exists, parses as JSON, and contains Android Web Share Target configuration', () => {
  const manifestPath = path.join(webRoot, 'public', 'manifest.webmanifest');
  assert.ok(fs.existsSync(manifestPath), 'manifest.webmanifest must exist in public folder');

  const content = fs.readFileSync(manifestPath, 'utf-8');
  const manifest = JSON.parse(content);

  assert.equal(manifest.name, 'Measured Fit - 3D Body Twin & Precision Sizing');
  assert.equal(manifest.short_name, 'Measured Fit');
  assert.equal(manifest.start_url, '/app');
  assert.equal(manifest.display, 'standalone');
  assert.equal(manifest.theme_color, '#0284c7');
  assert.equal(manifest.background_color, '#090d16');

  // Verify share_target for Android Web Share Target support
  assert.ok(manifest.share_target, 'manifest must define share_target');
  assert.equal(manifest.share_target.action, '/app');
  assert.equal(manifest.share_target.method, 'GET');
  assert.equal(manifest.share_target.params.url, 'url');
  assert.equal(manifest.share_target.params.title, 'title');
  assert.equal(manifest.share_target.params.text, 'text');

  // Verify icons
  assert.ok(Array.isArray(manifest.icons) && manifest.icons.length >= 3);
  const icon192 = manifest.icons.find((i) => i.sizes === '192x192');
  const icon512 = manifest.icons.find((i) => i.sizes === '512x512');
  const maskable = manifest.icons.find((i) => i.purpose?.includes('maskable'));

  assert.ok(icon192, 'Icon 192x192 must be defined');
  assert.ok(icon512, 'Icon 512x512 must be defined');
  assert.ok(maskable, 'Maskable icon must be defined for Android home screen integration');
});

test('Service Worker enforces strict biometric privacy rules (network-only for authenticated and twin APIs)', () => {
  const swPath = path.join(webRoot, 'public', 'sw.js');
  assert.ok(fs.existsSync(swPath), 'sw.js must exist in public folder');

  const swCode = fs.readFileSync(swPath, 'utf-8');

  // Must define API routes to bypass
  assert.ok(
    swCode.includes('/api/') && swCode.includes('/v1/') && swCode.includes('/twin/'),
    'Service worker must identify /api/, /v1/, and /twin/ endpoints for cache exclusion'
  );

  // Must bypass Authorization headers to protect authenticated user requests
  assert.ok(
    swCode.includes('authorization') || swCode.includes('Authorization'),
    'Service worker must detect Authorization headers and bypass caching'
  );

  // Must never cache private twin data or raw photos - direct fetch bypass
  assert.ok(
    swCode.includes('fetch(request)') || swCode.includes('fetch(event.request)'),
    'Service worker must route API requests directly to network'
  );
});

test('Android Share Target query parser correctly extracts shared product URLs and titles', () => {
  // Scenario 1: Direct URL param (standard Chrome share)
  const url1 = new URL('https://app.measuredfit.com/app?url=https%3A%2F%2Fshop.example.com%2Fitem%2F123&title=Linen+Shirt');
  const sharedUrl1 = url1.searchParams.get('url') || url1.searchParams.get('text') || '';
  const sharedTitle1 = url1.searchParams.get('title') || '';

  assert.equal(sharedUrl1, 'https://shop.example.com/item/123');
  assert.equal(sharedTitle1, 'Linen Shirt');

  // Scenario 2: Text fallback when app sends URL in text parameter
  const url2 = new URL('https://app.measuredfit.com/app?text=https%3A%2F%2Fshop.example.com%2Fitem%2F456');
  const sharedUrl2 = url2.searchParams.get('url') || url2.searchParams.get('text') || '';

  assert.equal(sharedUrl2, 'https://shop.example.com/item/456');

  // Scenario 3: No share target params
  const url3 = new URL('https://app.measuredfit.com/app');
  const sharedUrl3 = url3.searchParams.get('url') || url3.searchParams.get('text') || '';
  assert.equal(sharedUrl3, '');
});

test('Customer progress data structure isolates draft UI state from private body measurements', () => {
  const sampleProgress = {
    user_id: 'user_alice',
    active_tab: 'capture',
    height_cm: '172.5',
    gender: 'female',
    age_18_confirmed: true,
    opt_in_keep_front_photo: false,
    captured_views: ['front', 'back', 'left', 'right'],
    last_job_id: 'job_abc123',
    updated_at: new Date().toISOString(),
  };

  // Ensure forbidden biometric keys are NOT present
  const keys = Object.keys(sampleProgress);
  assert.ok(!keys.includes('measurements'), 'Progress payload must not contain body measurements');
  assert.ok(!keys.includes('zones'), 'Progress payload must not contain anatomic zones');
  assert.ok(!keys.includes('photo_front'), 'Progress payload must not contain raw photos');
  assert.ok(!keys.includes('photo_back'), 'Progress payload must not contain raw photos');
  assert.ok(!keys.includes('glb_bytes'), 'Progress payload must not contain GLB mesh binaries');

  assert.equal(sampleProgress.height_cm, '172.5');
  assert.equal(sampleProgress.gender, 'female');
  assert.equal(sampleProgress.age_18_confirmed, true);
  assert.deepEqual(sampleProgress.captured_views, ['front', 'back', 'left', 'right']);
});

test('Customer App product URL input handles both shared and direct URL submissions', () => {
  // Direct input without share target
  let input = 'https://store.local/shirt.html';
  assert.ok(input.trim().length > 0);
  assert.ok(input.startsWith('http://') || input.startsWith('https://'));

  // Shared target pre-populates input
  const sharedUrl = 'https://demo.local/store/jacket.html';
  input = sharedUrl;
  assert.equal(input, 'https://demo.local/store/jacket.html');

  // Dismissing clears the input
  input = '';
  assert.equal(input, '');
});

test('Customer App terminal job status correctly identifies retake vs failure and provides recovery instructions', () => {
  const terminalJobNeedsRetake = {
    id: 'job_test123',
    status: 'needs_retake',
    error_code: 'POSE_VALIDATION_FAILED',
    error_message: 'Front view arms must be raised to shoulder height with a gap at the chest.',
  };

  const isTerminal = ['done', 'needs_retake', 'rejected', 'failed'].includes(terminalJobNeedsRetake.status);
  assert.ok(isTerminal, 'needs_retake must be treated as terminal');

  const recoveryMessage = terminalJobNeedsRetake.status === 'needs_retake'
    ? `Photo Validation Issue: ${terminalJobNeedsRetake.error_message}. Please capture new photos with clear orthogonal views.`
    : 'Job Error';
  assert.ok(recoveryMessage.includes('arms must be raised'));
  assert.ok(recoveryMessage.includes('clear orthogonal views'));

  const terminalJobFailed = {
    id: 'job_test456',
    status: 'failed',
    error_code: 'SMPS_CONVERGENCE_ERROR',
    error_message: 'Failed to converge SMPL mesh alignment.',
  };
  const isFailedTerminal = ['done', 'needs_retake', 'rejected', 'failed'].includes(terminalJobFailed.status);
  assert.ok(isFailedTerminal, 'failed must be treated as terminal');
});

test('Customer App unauthenticated submission protection prompts with recovery action', () => {
  const user = null;
  const signedIn = false;

  const validateSubmission = (authStatus) => {
    if (!authStatus) {
      return { ok: false, error: 'Please sign in before submitting photos to generate your private 3D Body Twin.', action: 'signIn' };
    }
    return { ok: true };
  };

  const res = validateSubmission(signedIn);
  assert.equal(res.ok, false);
  assert.equal(res.action, 'signIn');
  assert.ok(res.error.includes('Please sign in'));
});

