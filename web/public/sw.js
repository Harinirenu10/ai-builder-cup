/**
 * Measured Fit - Service Worker with Strict Privacy Safeguards
 * 
 * CRITICAL PRIVACY RULE:
 * Never expose body measurements, photos, or private twin data through browser caches.
 * All API routes (/api/, /v1/, /twin/, /twins/) are strictly NETWORK-ONLY and NEVER cached.
 */

const CACHE_NAME = 'measured-fit-shell-v1';

// Static assets for offline App Shell (markup, styles, icons, fonts)
const APP_SHELL_ASSETS = [
  '/',
  '/app',
  '/index.html',
  '/manifest.webmanifest',
  '/favicon.svg',
  '/icons/icon-192.svg',
  '/icons/icon-512.svg',
  '/icons/icon-maskable.svg',
];

// Sensitive endpoints containing biometric data or authenticated state that MUST NEVER be cached
const SENSITIVE_URL_PATTERNS = [
  /\/api\//i,
  /\/v1\//i,
  /\/twin\//i,
  /\/twins\//i,
  /\/mesh\//i,
  /\/glb/i,
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(async (cache) => {
      // Best-effort pre-caching of app shell assets
      for (const asset of APP_SHELL_ASSETS) {
        try {
          await cache.add(asset);
        } catch {
          // Non-blocking if asset not yet available in dev mode
        }
      }
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  const url = new URL(request.url);

  // 1. Strict Privacy Check: never cache private/biometric APIs, POST/PUT mutations, or authenticated calls
  const isSensitive =
    request.method !== 'GET' ||
    request.headers.has('Authorization') ||
    SENSITIVE_URL_PATTERNS.some((pattern) => pattern.test(url.pathname));

  if (isSensitive) {
    // Strictly bypass cache - direct to network
    event.respondWith(fetch(request));
    return;
  }

  // 2. Navigation Requests (HTML pages): Network-first with offline fallback to /index.html
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request).catch(async () => {
        const cache = await caches.open(CACHE_NAME);
        const cachedIndex = await cache.match('/index.html');
        return cachedIndex || new Response('Offline - Measured Fit', {
          status: 503,
          headers: { 'Content-Type': 'text/html' },
        });
      })
    );
    return;
  }

  // 3. Static Assets (JS, CSS, SVGs, Fonts): Stale-While-Revalidate
  event.respondWith(
    caches.match(request).then((cachedResponse) => {
      const fetchPromise = fetch(request)
        .then((networkResponse) => {
          if (networkResponse && networkResponse.status === 200) {
            const responseToCache = networkResponse.clone();
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(request, responseToCache);
            });
          }
          return networkResponse;
        })
        .catch(() => cachedResponse);

      return cachedResponse || fetchPromise;
    })
  );
});

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
