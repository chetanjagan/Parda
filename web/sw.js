/* Parda's isolation helper. It only touches Parda's own pages: it adds the two headers that let the browser give
   the page all CPU cores (cross-origin isolation), which GitHub Pages cannot send itself. Nothing is uploaded. */
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (new URL(req.url).origin !== self.location.origin) return;           // other sites: untouched
  if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
  e.respondWith(fetch(req).then((res) => {
    if (res.status === 0) return res;
    const h = new Headers(res.headers);
    h.set("Cross-Origin-Embedder-Policy", "credentialless");
    h.set("Cross-Origin-Opener-Policy", "same-origin");
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
  }));
});
