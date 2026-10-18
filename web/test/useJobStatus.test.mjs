import test from 'node:test';
import assert from 'node:assert/strict';

// Pure logic from useJobStatus.ts, tested directly in Node (no React renderer needed -- these
// are plain functions, not hooks). Covers the polling/failure-handling behavior the Body Twin
// capture flow depends on: POST /api/twin/jobs now runs the pipeline in the background (see
// services/api/main.py), so the frontend must poll GET /api/twin/jobs/{job_id} through every
// real intermediate status (checking -> measuring -> fitting -> refining) until a terminal one.
import { isPollingStatus, isTerminalStatus, describeJobStage } from '../src/lib/jobStatus.ts';

test('isPollingStatus is true for every non-terminal backend status (checking/measuring/fitting/refining)', () => {
  for (const status of ['checking', 'measuring', 'fitting', 'refining']) {
    assert.equal(isPollingStatus(status), true, `expected polling to continue for status "${status}"`);
  }
});

test('isPollingStatus is false for every terminal status and for undefined', () => {
  for (const status of ['done', 'needs_retake', 'rejected', 'failed', undefined]) {
    assert.equal(isPollingStatus(status), false, `expected polling to stop for status "${status}"`);
  }
});

test('isTerminalStatus agrees with isPollingStatus: exactly one is true for every known status', () => {
  const allStatuses = ['checking', 'measuring', 'fitting', 'refining', 'done', 'needs_retake', 'rejected', 'failed'];
  for (const status of allStatuses) {
    assert.notEqual(isPollingStatus(status), isTerminalStatus(status), `status "${status}" must be exactly one of polling/terminal`);
  }
});

test('describeJobStage returns a distinct, non-default stage for every real backend status', () => {
  const seen = new Set();
  for (const status of ['checking', 'measuring', 'fitting', 'refining', 'done']) {
    const { stage, progressPercent } = describeJobStage(status);
    assert.notEqual(stage, 'Preparing submission...', `status "${status}" fell through to the default stage text`);
    assert.ok(!seen.has(stage), `stage text for "${status}" duplicates an earlier status`);
    seen.add(stage);
  }
  assert.equal(describeJobStage('done').progressPercent, 100);
});

test('describeJobStage progress increases monotonically through the real pipeline order', () => {
  const order = ['checking', 'measuring', 'fitting', 'refining', 'done'];
  const percents = order.map((s) => describeJobStage(s).progressPercent);
  for (let i = 1; i < percents.length; i++) {
    assert.ok(percents[i] > percents[i - 1], `progress must increase from ${order[i - 1]} (${percents[i - 1]}%) to ${order[i]} (${percents[i]}%)`);
  }
});

test('describeJobStage surfaces the real error_message for failure/retake statuses', () => {
  const retake = describeJobStage('needs_retake', 'Front photo blurry.');
  assert.ok(retake.stage.includes('Front photo blurry.'));
  assert.equal(retake.progressPercent, 0);

  const failed = describeJobStage('failed', 'Unhandled error in background twin job.');
  assert.ok(failed.stage.includes('Unhandled error in background twin job.'));

  const rejected = describeJobStage('rejected', undefined);
  assert.ok(rejected.stage.length > 0); // falls back to a generic message, never blank
});

test('describeJobStage falls back to a generic "preparing" stage for an unknown/undefined status', () => {
  assert.equal(describeJobStage(undefined).stage, 'Preparing submission...');
});
