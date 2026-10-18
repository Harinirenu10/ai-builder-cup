/* Demo Threads Co. mock storefront: the bracketing nudge, a simulated checkout and a returns form.
 * Talks to the Measured Fit API through the fit widget's config (data-merchant, data-api). Nothing here sees body data. */
(function () {
  var BAG = "dtc_bag_v1", ORDERS = "dtc_orders_v1";

  function load(k) { try { return JSON.parse(localStorage.getItem(k)) || []; } catch (e) { return []; } }
  function save(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private mode */ } }
  function cfg() { return window.MeasuredFit && window.MeasuredFit.config; }
  function uniq(a) { return a.filter(function (x, i) { return a.indexOf(x) === i; }); }
  function el(tag, cls, text) { var n = document.createElement(tag); if (cls) n.className = cls; if (text) n.textContent = text; return n; }
  function bagChanged() { window.dispatchEvent(new CustomEvent("dtc:bag")); }

  function post(path, body) {
    var c = cfg();
    if (!c) return Promise.reject(new Error("widget missing"));
    return fetch(c.api + path, {
      method: "POST", keepalive: true,
      headers: { "Content-Type": "application/json", "X-Merchant-Key": c.merchant },
      body: JSON.stringify(body)
    }).then(function (r) {
      return r.text().then(function (t) {
        var j = {}; try { j = t ? JSON.parse(t) : {}; } catch (e) { /* empty */ }
        return { ok: r.ok, status: r.status, body: j };
      });
    });
  }
  function logEvent(type, ref) { post("/v1/store-events", { type: type, product_ref: ref }).catch(function () { /* analytics only */ }); }

  // ------------------------------------------------------------------ bracketing nudge (cart)
  function initNudge() {
    var box = document.getElementById("nudge");
    if (!box) return;
    var shown = {}, dismissed = {};

    function check() {
      box.innerHTML = "";
      if (!window.MeasuredFit || !window.MeasuredFit.hasConsent()) return;
      var groups = {};
      load(BAG).forEach(function (l) { (groups[l.slug] = groups[l.slug] || []).push(l); });
      Object.keys(groups).forEach(function (slug) {
        var group = groups[slug], sizes = uniq(group.map(function (l) { return l.size; }));
        if (sizes.length < 2 || !group[0].fit || dismissed[slug]) return;
        window.MeasuredFit.recommend(group[0].fit).then(function (r) {
          var rec = r.recommended_size;
          if (!rec || sizes.indexOf(rec) < 0 || document.getElementById("nudge-" + slug)) return;
          var others = sizes.filter(function (s) { return s !== rec; });
          var row = el("div", "nudge");
          row.id = "nudge-" + slug;
          row.setAttribute("role", "status");
          row.appendChild(el("span", "", "Your twin says " + rec + " fits. Remove the " + others.join(" and ") + "?"));
          var yes = el("button", "primary", "Remove the " + others.join(" and "));
          yes.type = "button";
          yes.addEventListener("click", function () {
            save(BAG, load(BAG).filter(function (l) { return !(l.slug === slug && others.indexOf(l.size) >= 0); }));
            logEvent("nudge_accepted", group[0].fit.ref);
            bagChanged();
          });
          var no = el("button", "", "Keep both");
          no.type = "button";
          no.addEventListener("click", function () { dismissed[slug] = true; logEvent("nudge_dismissed", group[0].fit.ref); row.remove(); });
          row.appendChild(yes); row.appendChild(no);
          box.appendChild(row);
          if (!shown[slug]) { shown[slug] = true; logEvent("nudge_shown", group[0].fit.ref); }
        }, function () { /* no consent or no answer: no nudge */ });
      });
    }
    window.addEventListener("dtc:bag", check);
    check();
  }

  // ------------------------------------------------------------------ checkout (simulated, no payment)
  function initCheckout() {
    var root = document.getElementById("checkout-root");
    if (!root) return;
    var lines = load(BAG);
    if (!lines.length) { root.textContent = "Your bag is empty."; return; }
    var list = el("div");
    lines.forEach(function (l) { list.appendChild(el("div", "line", l.title + " · size " + l.size + " · qty " + l.qty)); });
    root.appendChild(list);
    var linked = window.MeasuredFit && window.MeasuredFit.hasConsent();
    root.appendChild(el("p", "", linked
      ? "Your size profile is connected, so we can check whether the fit guarantee applies."
      : "Connect your size profile with Find my size on a product page to get the fit guarantee."));
    var out = el("div"); out.id = "order-result"; out.setAttribute("role", "status");
    var btn = el("button", "primary", "Place order (simulated, no payment)");
    btn.type = "button"; btn.id = "place-order";
    btn.addEventListener("click", function () {
      btn.disabled = true;
      var items = lines.map(function (l) { return { sku: (l.fit && l.fit.ref) || l.slug, title: l.title, size: l.size, qty: l.qty, product: l.fit }; });
      post("/v1/orders", { items: items, consent_token: window.MeasuredFit ? window.MeasuredFit.token() : null }).then(function (res) {
        if (!res.ok) { out.textContent = "We could not place the order. Please try again."; btn.disabled = false; return; }
        var o = res.body;
        out.innerHTML = "";
        out.appendChild(el("p", "", "Order placed (simulated). Order ID: " + o.order_id));
        o.items.forEach(function (it) {
          var t = it.guarantee ? "Fit guarantee: yes. Size " + it.size + " is the size your twin recommends."
            : "Fit guarantee: not for this item" + (it.recommended_size ? " (your twin recommends " + it.recommended_size + ")." : ".");
          out.appendChild(el("div", "line", it.sku + " · size " + it.size + " · " + t));
        });
        if (o.consent === "revoked" || o.consent === "invalid") out.appendChild(el("p", "", "Your size profile connection has ended, so no guarantee was applied."));
        var orders = load(ORDERS);
        orders.push({ id: o.order_id, items: o.items.map(function (it, i) { return { index: i, sku: it.sku, size: it.size, title: items[i].title, guarantee: it.guarantee }; }) });
        save(ORDERS, orders);
        save(BAG, []);
        bagChanged();
        var a = el("a", "", "Need to return something?"); a.href = "returns.html"; out.appendChild(a);
        btn.remove();
      }, function () { out.textContent = "We could not reach the store service. Please try again."; btn.disabled = false; });
    });
    root.appendChild(btn); root.appendChild(out);
  }

  // ------------------------------------------------------------------ returns form
  var REASONS = [["too_small", "Too small"], ["too_big", "Too big"], ["wrong_size_received", "I received a different size than I ordered"],
    ["quality", "Quality"], ["changed_mind", "Changed my mind"], ["other", "Other"]];
  var ERRORS = { already_filed: "A return for this item has already been filed.", order_not_found: "We could not find that order.", bad_unboxing: "That unboxing check does not belong to this item." };

  function initReturns() {
    var root = document.getElementById("returns-root");
    if (!root) return;
    var orders = load(ORDERS), form = el("form"), pick = el("select"), idIn = el("input"), idxIn = el("input"), why = el("select"), note = el("textarea"), ub = el("input"), out = el("div");
    pick.id = "return-item"; idIn.id = "return-order-id"; idxIn.id = "return-item-number"; why.id = "return-reason"; note.id = "return-note"; ub.id = "return-unboxing"; out.id = "return-result";
    out.setAttribute("role", "status");
    orders.forEach(function (o) { o.items.forEach(function (it) {
      var op = el("option", "", "Order " + o.id.slice(0, 8) + ": " + it.title + ", size " + it.size); op.value = o.id + "|" + it.index; pick.appendChild(op);
    }); });
    if (!orders.length) { var none = el("option", "", "No orders on this device"); none.value = ""; pick.appendChild(none); }
    idIn.placeholder = "Order ID (if it is not listed)"; idxIn.type = "number"; idxIn.min = "1"; idxIn.value = "1"; idxIn.title = "Item number in the order";
    REASONS.forEach(function (r) { var op = el("option", "", r[1]); op.value = r[0]; why.appendChild(op); });
    note.placeholder = "Anything you would like us to know (optional)"; note.maxLength = 500; note.rows = 3;
    ub.placeholder = "Unboxing check ID (optional)";
    var submit = el("button", "primary", "Submit return"); submit.type = "submit";
    [["Item", pick], ["or order ID", idIn], ["Item number", idxIn], ["Reason", why], ["Note", note], ["Unboxing check", ub]].forEach(function (p) {
      var row = el("div", "row"); var l = el("label", "", p[0]); l.htmlFor = p[1].id; row.appendChild(l); row.appendChild(p[1]); form.appendChild(row);
    });
    form.appendChild(submit); root.appendChild(form); root.appendChild(out);
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var orderId = idIn.value.trim(), index = Math.max(0, (parseInt(idxIn.value, 10) || 1) - 1);
      if (!orderId && pick.value) { var parts = pick.value.split("|"); orderId = parts[0]; index = parseInt(parts[1], 10); }
      if (!orderId) { out.textContent = "Choose an item or enter an order ID."; return; }
      submit.disabled = true;
      post("/v1/returns", { order_id: orderId, item_index: index, reason: why.value, note: note.value.trim() || null, unboxing_measurement_id: ub.value.trim() || null })
        .then(function (res) {
          submit.disabled = false;
          if (res.ok) { out.textContent = res.body.message; return; }
          var code = res.body && res.body.detail && res.body.detail.code;
          out.textContent = ERRORS[code] || "We could not file the return. Please try again.";
        }, function () { submit.disabled = false; out.textContent = "We could not reach the store service. Please try again."; });
    });
  }

  initNudge();
  initCheckout();
  initReturns();
})();
