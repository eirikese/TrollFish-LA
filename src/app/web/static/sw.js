const CACHE_NAME = 'trollfish-static-v20260526-cont4';
const PRECACHE_URLS = [
  './index.html',
  './manifest.webmanifest',
  './app.js?v=20260526cont4',
  './icons/icon-192.svg',
  './icons/icon-512.svg',
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE_NAME);
    await Promise.all(PRECACHE_URLS.map(url => (
      cache.add(url).catch(err => console.warn('[SW] precache skipped', url, err))
    )));
    self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map((key) => {
      if (key !== CACHE_NAME) return caches.delete(key);
      return Promise.resolve();
    }));
    await self.clients.claim();
  })());
});

function shouldHandle(requestUrl) {
  if (requestUrl.pathname.includes('/api/')) return false;
  if (requestUrl.origin === self.location.origin) return true;
  return requestUrl.origin.includes('unpkg.com');
}

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (!shouldHandle(url)) return;

  const isNavigation = request.mode === 'navigate';
  const isSameOrigin = url.origin === self.location.origin;

  if (isNavigation) {
    event.respondWith((async () => {
      const appRootUrl = new URL('./', self.registration.scope).toString();
      const appIndexUrl = new URL('./index.html', self.registration.scope).toString();
      try {
        const fresh = await fetch(request);
        const cache = await caches.open(CACHE_NAME);
        cache.put(request, fresh.clone());
        return fresh;
      } catch {
        const cache = await caches.open(CACHE_NAME);
        return (await cache.match(request))
          || (await cache.match(appIndexUrl))
          || (await cache.match(appRootUrl))
          || Response.error();
      }
    })());
    return;
  }

  if (isSameOrigin) {
    const path = url.pathname || '';
    const isAppCode =
      path.endsWith('/')
      || path.endsWith('/index.html')
      || path.endsWith('.js')
      || path.endsWith('.mjs')
      || path.endsWith('.css')
      || path.endsWith('.webmanifest');

    if (isAppCode) {
      event.respondWith((async () => {
        const cache = await caches.open(CACHE_NAME);
        try {
          const fresh = await fetch(request, { cache: 'no-cache' });
          if (fresh && fresh.ok) cache.put(request, fresh.clone());
          return fresh;
        } catch {
          return (await cache.match(request)) || Response.error();
        }
      })());
      return;
    }

    event.respondWith((async () => {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(request);
      if (cached) return cached;
      const fresh = await fetch(request);
      if (fresh && fresh.ok) cache.put(request, fresh.clone());
      return fresh;
    })());
    return;
  }

  // CDN assets: stale-while-revalidate
  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(request);
    const networkPromise = fetch(request).then((fresh) => {
      if (fresh && fresh.ok) cache.put(request, fresh.clone());
      return fresh;
    }).catch(() => null);
    return cached || (await networkPromise) || Response.error();
  })());
});
