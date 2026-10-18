import test from 'node:test';
import assert from 'node:assert/strict';

// Test ApiClient and error parsing logic in a clean Node environment
import { ApiClient, ApiError, buildUrl, parseErrorResponse, loadObjectUrl } from '../src/lib/api.ts';

test('buildUrl constructs correct URLs with base and params', () => {
  const url1 = buildUrl('/api/twin/jobs');
  assert.ok(url1.includes('/api/twin/jobs'));

  const url2 = buildUrl('https://api.example.com/v1/fit', { explain: true, lang: 'en' });
  assert.equal(url2, 'https://api.example.com/v1/fit?explain=true&lang=en');

  const url3 = buildUrl('/test', { page: 1, filter: null, active: undefined });
  assert.equal(url3.split('?')[1], 'page=1');
});

test('parseErrorResponse handles structured FastAPI error responses', async () => {
  const mockRes1 = new Response(JSON.stringify({ detail: { code: 'no_twin', message: 'No body twin found' } }), {
    status: 409,
    headers: { 'Content-Type': 'application/json' },
  });
  const err1 = await parseErrorResponse(mockRes1);
  assert.equal(err1.status, 409);
  assert.equal(err1.code, 'no_twin');
  assert.equal(err1.message, 'No body twin found');

  const mockRes2 = new Response(JSON.stringify({ detail: 'Invalid input' }), {
    status: 422,
    headers: { 'Content-Type': 'application/json' },
  });
  const err2 = await parseErrorResponse(mockRes2);
  assert.equal(err2.status, 422);
  assert.equal(err2.message, 'Invalid input');

  const mockRes401 = new Response(JSON.stringify({ detail: 'Sign in first' }), {
    status: 401,
    headers: { 'Content-Type': 'application/json' },
  });
  const err401 = await parseErrorResponse(mockRes401);
  assert.equal(err401.status, 401);
  assert.equal(err401.code, 'unauthorized');
});

test('ApiClient performs successful request and parses JSON', async () => {
  const client = new ApiClient();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    return new Response(JSON.stringify({ status: 'ok', twin_id: 'twin_123' }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const res = await client.get('/api/twin/twin_123');
    assert.deepEqual(res, { status: 'ok', twin_id: 'twin_123' });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('ApiClient injects authorization token', async () => {
  const client = new ApiClient();
  let capturedAuthHeader = null;

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    capturedAuthHeader = options.headers.get('Authorization');
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    // 1. Explicit token
    await client.get('/test', { token: 'dev:demo-user' });
    assert.equal(capturedAuthHeader, 'Bearer dev:demo-user');

    // 2. Token getter
    client.setTokenGetter(async () => 'dev:getter-user');
    await client.get('/test');
    assert.equal(capturedAuthHeader, 'Bearer dev:getter-user');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('ApiClient handles timeouts by aborting and throwing ApiError 408', async () => {
  const client = new ApiClient();
  client.setDefaultTimeout(50); // 50ms timeout

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    return new Promise((resolve, reject) => {
      options.signal?.addEventListener('abort', () => {
        const abortErr = new Error('The operation was aborted');
        abortErr.name = 'AbortError';
        reject(abortErr);
      });
    });
  };

  try {
    await assert.rejects(
      async () => {
        await client.get('/slow-endpoint', { timeoutMs: 50 });
      },
      (err) => {
        assert.ok(err instanceof ApiError);
        assert.equal(err.status, 408);
        assert.equal(err.code, 'timeout');
        return true;
      }
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('ApiClient handles 401 unauthorized errors and dispatches event', async () => {
  const client = new ApiClient();
  let eventDispatched = false;

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    return new Response(JSON.stringify({ detail: 'Auth required' }), {
      status: 401,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  if (typeof globalThis.window === 'undefined') {
    globalThis.window = {
      dispatchEvent: (evt) => {
        if (evt.type === 'omni:unauthorized') eventDispatched = true;
      },
    };
  }

  try {
    await assert.rejects(
      async () => {
        await client.get('/protected');
      },
      (err) => {
        assert.ok(err instanceof ApiError);
        assert.equal(err.status, 401);
        assert.equal(err.code, 'unauthorized');
        return true;
      }
    );
    assert.equal(eventDispatched, true);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

// ----------------------------------------------------------------------------- Body Twin flow:
// authenticated job creation, authenticated GLB loading, object-URL cleanup, unauthorized errors.
// These cover the frontend side of the backend ownership fix: GET /api/twin/{id}/glb and
// POST /api/twin/jobs now require a signed-in caller, and model-viewer's own `src` fetch cannot
// carry an Authorization header, so the GLB must be loaded as an authenticated Blob (loadObjectUrl)
// instead of being handed to model-viewer as a plain URL.

test('POST /api/twin/jobs sends the Authorization header (authenticated job creation)', async () => {
  const client = new ApiClient();
  let capturedAuthHeader = null;
  let capturedBody = null;

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    capturedAuthHeader = options.headers.get('Authorization');
    capturedBody = JSON.parse(options.body);
    return new Response(JSON.stringify({ status: 'done', job_id: 'job_1', twin_id: 'twin_1' }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };
  client.setTokenGetter(() => 'dev:alice');

  try {
    const res = await client.post('/api/twin/jobs', { user_id: 'alice', height_cm: 170 });
    assert.equal(capturedAuthHeader, 'Bearer dev:alice');
    assert.equal(capturedBody.height_cm, 170);
    assert.equal(res.job_id, 'job_1');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('loadObjectUrl fetches the GLB as an authenticated Blob and returns a usable object URL', async () => {
  const client = new ApiClient();
  let capturedAuthHeader = null;
  const glbBytes = new Uint8Array([0x67, 0x6c, 0x54, 0x46]); // "glTF" magic

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    capturedAuthHeader = options.headers.get('Authorization');
    assert.ok(String(url).includes('/api/twin/twin_1/glb'));
    return new Response(glbBytes, { status: 200, headers: { 'Content-Type': 'model/gltf-binary' } });
  };
  client.setTokenGetter(() => 'dev:alice');

  try {
    const { url, revoke } = await loadObjectUrl('/api/twin/twin_1/glb', client);
    assert.equal(capturedAuthHeader, 'Bearer dev:alice');
    assert.ok(url.startsWith('blob:'));
    revoke(); // cleanup; also exercised precisely in the next test
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('loadObjectUrl: revoke() releases the object URL exactly once, even if called twice', async () => {
  const client = new ApiClient();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response(new Uint8Array([1, 2, 3]), { status: 200, headers: { 'Content-Type': 'model/gltf-binary' } });

  const originalRevoke = URL.revokeObjectURL;
  let revokeCalls = [];
  URL.revokeObjectURL = (u) => revokeCalls.push(u);

  try {
    const { url, revoke } = await loadObjectUrl('/api/twin/twin_1/glb', client);
    revoke();
    revoke(); // calling it again must not revoke a second time
    assert.deepEqual(revokeCalls, [url]);
  } finally {
    globalThis.fetch = originalFetch;
    URL.revokeObjectURL = originalRevoke;
  }
});

test('loadObjectUrl propagates a 401 ApiError (signed out) without creating an object URL', async () => {
  const client = new ApiClient();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response(JSON.stringify({ detail: { code: 'auth_required', message: 'Sign in first.' } }), {
      status: 401,
      headers: { 'Content-Type': 'application/json' },
    });

  const originalCreate = URL.createObjectURL;
  let createCalled = false;
  URL.createObjectURL = (b) => {
    createCalled = true;
    return originalCreate(b);
  };

  try {
    await assert.rejects(
      async () => loadObjectUrl('/api/twin/twin_1/glb', client),
      (err) => {
        assert.ok(err instanceof ApiError);
        assert.equal(err.status, 401);
        assert.equal(err.code, 'unauthorized');
        return true;
      }
    );
    assert.equal(createCalled, false);
  } finally {
    globalThis.fetch = originalFetch;
    URL.createObjectURL = originalCreate;
  }
});

test('loadObjectUrl propagates a 403 ApiError (cross-user access) without creating an object URL', async () => {
  const client = new ApiClient();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response(JSON.stringify({ detail: { code: 'forbidden', message: 'This twin belongs to another user.' } }), {
      status: 403,
      headers: { 'Content-Type': 'application/json' },
    });

  try {
    await assert.rejects(
      async () => loadObjectUrl('/api/twin/twin_1/glb', client),
      (err) => {
        assert.ok(err instanceof ApiError);
        assert.equal(err.status, 403);
        assert.equal(err.code, 'forbidden');
        return true;
      }
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

// Regression for the "No 3D Twin Yet despite worker logs showing privacy cleanup completing" bug:
// an un-proxied dev-server path (e.g. a '/v1/...' call when only '/api' was in vite.config.ts's
// proxy table) falls through to Vite's own SPA fallback, which answers unmatched GETs with
// index.html at HTTP 200, Content-Type text/html. ApiClient.request() sees response.ok === true
// and a text/* content-type, so it resolves successfully with the raw HTML string instead of
// throwing -- meaning callers' ApiError-based error handling (404/409 => null, or a retry-UI
// catch block) never engages, and the caller is left holding HTML in place of twin JSON. The
// real fix is proxying every prefix the app uses (see vite-proxy.test.mjs); this test locks down
// the ApiClient behavior itself so the failure mode stays understood and detectable.
test('ApiClient resolves (does not throw) when a 200 response has non-JSON text content, returning the raw text', async () => {
  const client = new ApiClient();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response('<!doctype html><html><body>Vite SPA fallback</body></html>', {
      status: 200,
      headers: { 'Content-Type': 'text/html' },
    });

  try {
    const result = await client.get('/v1/me/twin');
    assert.equal(typeof result, 'string');
    assert.ok(result.includes('<html>'));
    assert.equal(result?.glb_url, undefined);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
