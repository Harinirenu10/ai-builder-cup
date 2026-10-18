/*! Measured Fit size widget. Vanilla JS, no dependencies.
 *
 *   <script src="fit-widget.js" data-merchant="pk_..." data-api="https://api.example/api" data-app="https://app.example" defer></script>
 *   <div data-fit-product='{"size_chart":{...},"category":"shirt","fit_type":"regular","fabric":[...]}'></div>
 *
 *   Or, when the merchant has not embedded a size chart, the same attribute may hold a bare
 *   product page URL instead of JSON; the Reader service reads its size chart live:
 *   <div data-fit-product="https://shop.example/products/kurta"></div>
 *
 * Flow: "Find my size" -> consent popup on the Measured Fit app -> popup posts a signed consent token back
 * (only to data-app's origin) -> POST {data-api}/v1/fit-query (embedded chart) or
 * POST {data-api}/v1/fit-query/from-url (bare URL) -> "L fits you · high confidence · sleeves may need 2 cm shortening".
 * The widget never sees body measurements; the answer holds a size, a confidence and verdict words only.
 */
(function () {
  "use strict";
  if (window.MeasuredFit) return;

  var script = document.currentScript || document.querySelector("script[data-merchant]");
  if (!script) return;
  var merchant = script.getAttribute("data-merchant");
  var api = (script.getAttribute("data-api") || "/api").replace(/\/+$/, "");
  var appBase = (script.getAttribute("data-app") || window.location.origin).replace(/\/+$/, "");
  var appOrigin = new URL(appBase, window.location.href).origin;
  var STORE_KEY = "mfw:consent:" + merchant;
  var MSG_OK = "measured-fit:consent";
  var MSG_NO = "measured-fit:denied";
  var HIGH = 0.75, MEDIUM = 0.5;                         // confidence words
  var ERRORS = {
    origin_not_allowed: "This store is not set up for Measured Fit yet.",
    unknown_merchant: "This store is not set up for Measured Fit yet.",
    rate_limited: "Lots of people are asking right now. Please try again in a minute.",
    unsupported_chart: "We could not compare this size chart.",
    // from reading the product page live (POST .../fit-query/from-url); see read_product_from_url
    // in services/api/apilib/fitquery.py for the exact codes this maps.
    reader_timeout: "That product page took too long to read. Please try again.",
    reader_unavailable: "We could not read that product page right now. Please try again shortly.",
    reader_error: "We could not read that product page. Please try again.",
    page_blocked: "This store's page could not be read right now.",
    unsupported_category: "This product isn't a supported garment category yet.",
    no_size_chart: "This product page has no size chart, so we can't recommend a size.",
    no_twin: null                                          // handled with a link
  };
  var RECONNECT = { consent_revoked: 1, invalid_consent_token: 1, consent_expired: 1, wrong_merchant: 1, consent_not_found: 1, wrong_scope: 1 };
  var listeners = [];

  // ---------------------------------------------------------------- storage (may be blocked in private mode)
  function readConsent() {
    try {
      var c = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
      if (c && c.token && (!c.expires_at || Date.parse(c.expires_at) > Date.now())) return c;
    } catch (e) { /* ignore */ }
    return null;
  }
  var memory = null;
  function getConsent() { return memory || readConsent(); }
  function setConsent(c) {
    memory = c;
    try { if (c) localStorage.setItem(STORE_KEY, JSON.stringify(c)); else localStorage.removeItem(STORE_KEY); } catch (e) { /* keep in memory */ }
  }

  // ---------------------------------------------------------------- view
  function injectStyle() {
    if (document.getElementById("mfw-style")) return;
    var s = document.createElement("style");
    s.id = "mfw-style";
    s.textContent =
      ".mfw{font:14px/1.4 system-ui,-apple-system,Segoe UI,Arial,sans-serif;border:1px solid #cbd5e1;border-radius:8px;padding:12px;margin:12px 0;background:#fff;color:#0f172a;max-width:520px}" +
      ".mfw button{font:inherit;padding:8px 14px;border-radius:6px;border:1px solid #0f172a;background:#0f172a;color:#fff;cursor:pointer}" +
      ".mfw button:disabled{opacity:.6;cursor:default}.mfw .mfw-link{background:none;border:0;color:#0369a1;text-decoration:underline;padding:0 0 0 6px}" +
      ".mfw-result{font-weight:600}.mfw-sub{color:#475569;font-size:12px;margin-top:4px}.mfw-badge{display:inline-block;background:#dcfce7;color:#166534;border-radius:99px;padding:1px 8px;font-size:12px;margin-left:6px}" +
      ".mfw-err{color:#991b1b}";
    document.head.appendChild(s);
  }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text) n.textContent = text;
    return n;
  }

  function Widget(host) {
    var self = this;
    this.host = host;
    this.product = null;
    this.productUrl = null;
    var raw = (host.getAttribute("data-fit-product") || "").trim();
    if (/^https?:\/\//i.test(raw)) {
      this.productUrl = raw;                                // a bare product page URL: read live by the Reader
    } else {
      try { this.product = JSON.parse(raw); } catch (e) { console.warn("[fit-widget] data-fit-product is not valid JSON or a URL"); return; }
      if (!this.product || typeof this.product !== "object") {
        console.warn("[fit-widget] data-fit-product is empty");
        return;
      }
    }
    this.box = el("div", "mfw");
    this.box.setAttribute("role", "region");
    this.box.setAttribute("aria-label", "Size finder");
    this.live = el("div");
    this.live.setAttribute("aria-live", "polite");
    this.box.appendChild(this.live);
    host.appendChild(this.box);
    this.popup = null;
    this.idle();
    listeners.push(function (e) { self.onMessage(e); });
  }

  Widget.prototype.show = function () {
    this.live.innerHTML = "";
    for (var i = 0; i < arguments.length; i++) this.live.appendChild(arguments[i]);
  };

  Widget.prototype.button = function (label, fn, disabled) {
    var b = el("button", "", label);
    b.type = "button";
    b.disabled = !!disabled;
    b.addEventListener("click", fn);
    return b;
  };

  Widget.prototype.idle = function (message) {
    var self = this, nodes = [];
    if (message) nodes.push(el("div", "mfw-err", message));
    nodes.push(this.button("Find my size", function () { self.start(); }));
    nodes.push(el("div", "mfw-sub", "Uses your saved profile. The store sees only a size, never your measurements."));
    this.show.apply(this, nodes);
  };

  Widget.prototype.busy = function (text) {
    this.show(el("div", "", text), this.button("Please wait", function () {}, true));
  };

  // A click is the only moment a pop-up is allowed, so open it here before any async work.
  Widget.prototype.start = function () {
    if (getConsent()) return this.query();
    this.connect();
  };

  Widget.prototype.connect = function () {
    var url = appBase + "/app/connect?merchant=" + encodeURIComponent(merchant) + "&origin=" + encodeURIComponent(window.location.origin);
    if (this.popup && !this.popup.closed) { this.popup.focus(); return; }               // one pop-up at a time
    var win = window.open(url, "measured-fit-connect-" + Date.now(), "width=460,height=700");   // fresh name: never reuse an old window
    if (!win) return this.idle("Your browser blocked the pop-up. Allow pop-ups for this site and try again.");
    this.popup = win;
    this.busy("Waiting for you to approve in the pop-up…");
    var self = this, t0 = Date.now();
    var timer = setInterval(function () {
      if (!self.popup) return clearInterval(timer);
      if (self.popup.closed || Date.now() - t0 > 600000) {
        clearInterval(timer);
        if (self.popup) { self.popup = null; if (!getConsent()) self.idle("No problem. Nothing was shared."); }
      }
    }, 500);
  };

  Widget.prototype.onMessage = function (e) {
    if (e.origin !== appOrigin || !this.popup || e.source !== this.popup) return;      // only our own popup, only from the app
    var d = e.data;
    if (!d || typeof d !== "object") return;
    if (d.type === MSG_NO) { this.popup = null; return this.idle("No problem. Nothing was shared."); }
    if (d.type !== MSG_OK || typeof d.token !== "string" || d.token.length < 20) return;
    this.popup = null;
    setConsent({ token: d.token, expires_at: d.expires_at || null });
    this.query();
  };

  // One fit query with a consent token; resolves {ok, status, body} whatever the status.
  // `product` is either an embedded product object (POST /v1/fit-query) or, when `isUrl` is
  // true, a bare product page URL for the Reader to read live (POST /v1/fit-query/from-url).
  function request(product, token, isUrl) {
    var path = isUrl ? "/v1/fit-query/from-url" : "/v1/fit-query";
    var body = isUrl ? { product_url: product, consent_token: token } : { product: product, consent_token: token };
    return fetch(api + path, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Merchant-Key": merchant },
      body: JSON.stringify(body)
    }).then(function (r) {
      return r.json().then(function (j) { return { ok: r.ok, status: r.status, body: j }; }, function () { return { ok: r.ok, status: r.status, body: {} }; });
    });
  }
  function codeOf(res) { return ((res.body && res.body.detail) || {}).code || ""; }

  Widget.prototype.query = function () {
    var self = this, c = getConsent();
    if (!c) return this.connect();
    this.busy(this.productUrl ? "Reading this product's page…" : "Finding your size…");
    request(this.productUrl || this.product, c.token, !!this.productUrl).then(function (res) {
      if (res.ok) return self.result(res.body);
      var code = codeOf(res);
      if (RECONNECT[code]) { setConsent(null); return self.reconnect(); }
      if (code === "no_twin") return self.noTwin();
      self.idle(ERRORS[code] || "We could not find your size right now. Please try again.");
    }, function () {
      self.idle(self.productUrl ? "We could not reach the product reader. Please try again." : "We could not reach the size service. Please try again.");
    });
  };

  Widget.prototype.reconnect = function () {
    var self = this;
    this.show(el("div", "", "Your connection to this store has ended. Connect again to see your size."), this.button("Connect again", function () { self.connect(); }));
  };

  Widget.prototype.noTwin = function () {
    var a = el("a", "mfw-link", "Create your size profile");
    a.href = appBase + "/app";
    a.target = "_blank";
    a.rel = "noopener";
    var d = el("div", "", "You have not created a size profile yet.");
    d.appendChild(a);
    this.show(d);
  };

  Widget.prototype.result = function (r) {
    var self = this;
    if (!r.recommended_size) {
      this.show(el("div", "mfw-result", "None of the sizes in this chart is a good match for this garment."), this.button("Check again", function () { self.query(); }));
      return;
    }
    var conf = r.confidence >= HIGH ? "high" : r.confidence >= MEDIUM ? "medium" : "low";
    var parts = [r.recommended_size + " fits you", conf + " confidence"];
    if (r.alteration_hint) parts.push(r.alteration_hint);
    var line = el("div", "mfw-result", parts.join(" · "));
    if (r.guarantee_eligible) line.appendChild(el("span", "mfw-badge", "Fit guarantee eligible"));
    var verdicts = (r.zone_verdicts || []).filter(function (z) { return z.verdict !== "ideal"; }).map(function (z) { return z.zone.replace(/_/g, " ") + ": " + z.verdict; });
    this.show(line, el("div", "mfw-sub", verdicts.length ? "Garment notes: " + verdicts.join(", ") : "Every checked area is in the garment's intended range."));
    try { this.host.dispatchEvent(new CustomEvent("measuredfit:result", { bubbles: true, detail: r })); } catch (e) { /* old browser */ }
  };

  // ---------------------------------------------------------------- boot
  window.addEventListener("message", function (e) { for (var i = 0; i < listeners.length; i++) listeners[i](e); });

  function mountAll() {
    injectStyle();
    var nodes = document.querySelectorAll("[data-fit-product]");
    for (var i = 0; i < nodes.length; i++) if (!nodes[i].__mfw) { nodes[i].__mfw = true; new Widget(nodes[i]); }
  }

  // For the store's own scripts (cart nudge, checkout). recommend() never opens a pop-up: with no consent it rejects.
  window.MeasuredFit = {
    mount: mountAll,
    version: "1.2.0",
    config: { merchant: merchant, api: api },
    // Re-read every data-fit-product (a store page that swapped in a new size chart calls this).
    remount: function () {
      var boxes = document.querySelectorAll(".mfw");
      for (var i = 0; i < boxes.length; i++) boxes[i].parentNode.removeChild(boxes[i]);
      var hosts = document.querySelectorAll("[data-fit-product]");
      for (var j = 0; j < hosts.length; j++) hosts[j].__mfw = false;
      listeners.length = 0;
      mountAll();
    },
    hasConsent: function () { return !!getConsent(); },
    token: function () { var c = getConsent(); return c ? c.token : null; },
    recommend: function (product) {
      var c = getConsent();
      if (!c) return Promise.reject({ code: "no_consent" });
      return request(product, c.token).then(function (res) {
        if (res.ok) return res.body;
        var code = codeOf(res) || "error";
        if (RECONNECT[code]) setConsent(null);
        return Promise.reject({ code: code });
      });
    }
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mountAll); else mountAll();
})();
