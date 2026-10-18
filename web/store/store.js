/* Demo Threads Co. mock storefront: size guide modal and a bag that allows two sizes of one item. */
(function () {
  var KEY = "dtc_bag_v1";

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch (e) { return []; }
  }
  function save(lines) {
    try { localStorage.setItem(KEY, JSON.stringify(lines)); } catch (e) { /* private mode: the bag lasts for this page only */ }
  }
  function count(lines) { return lines.reduce(function (n, l) { return n + l.qty; }, 0); }
  function paint() {
    var el = document.getElementById("cart-count");
    if (el) el.textContent = String(count(load()));
  }

  function sizesOf(lines, slug) {
    return lines.filter(function (l) { return l.slug === slug; }).map(function (l) { return l.size; });
  }

  function initProduct() {
    var body = document.body, slug = body.getAttribute("data-product");
    var modal = document.getElementById("chart-modal");
    var open = document.getElementById("size-guide-btn");
    if (open && modal) {
      open.addEventListener("click", function () { modal.hidden = false; });
      document.getElementById("chart-close").addEventListener("click", function () { modal.hidden = true; });
      document.addEventListener("keydown", function (e) { if (e.key === "Escape") modal.hidden = true; });
    }
    applyLiveChart();
    // The fit widget announces its answer; preselect that size so the shopper can add it in one click.
    document.addEventListener("measuredfit:result", function (e) {
      var d = e.detail, sel = document.getElementById("size");
      if (!d || !d.recommended_size || !sel) return;
      for (var i = 0; i < sel.options.length; i++) if (sel.options[i].value === d.recommended_size) sel.selectedIndex = i;
    });
    var add = document.getElementById("add-to-bag");
    if (!add) return;
    add.addEventListener("click", function () {
      var size = document.getElementById("size").value;
      var qty = Math.max(1, Math.min(5, parseInt(document.getElementById("qty").value, 10) || 1));
      var lines = load();
      var line = lines.filter(function (l) { return l.slug === slug && l.size === size; })[0];
      if (line) line.qty = Math.min(5, line.qty + qty);
      else {
        var slot = document.querySelector("[data-fit-product]"), fit = null;
        try { fit = slot ? JSON.parse(slot.getAttribute("data-fit-product")) : null; } catch (e) { fit = null; }
        lines.push({ slug: slug, title: body.getAttribute("data-title"), price: parseFloat(body.getAttribute("data-price")), size: size, qty: qty, fit: fit });
      }
      save(lines);
      paint();
      var sizes = sizesOf(lines, slug), note = document.getElementById("bag-note");
      note.textContent = sizes.length > 1
        ? "Added size " + size + ". You now have sizes " + sizes.join(" and ") + " of this item in your bag."
        : "Added size " + size + " to your bag.";
    });
  }

  var ZONE_TITLES = { chest: "Chest", waist: "Waist", hip: "Hip", shoulder_width: "Shoulder", sleeve_length: "Sleeve length", garment_length: "Length",
    thigh: "Thigh", inseam: "Inseam", neck: "Neck", back_rise: "Back rise", upper_back: "Upper back" };
  var ZONE_ORDER = ["chest", "waist", "hip", "shoulder_width", "sleeve_length", "garment_length", "thigh", "inseam", "neck", "back_rise", "upper_back"];

  // If the seller has put a chart on the store for this product (size chart studio), show it and let the size widget use it.
  function applyLiveChart() {
    var cfg = window.MeasuredFit && window.MeasuredFit.config, slot = document.querySelector("[data-fit-product]");
    if (!cfg || !slot) return;
    var fit; try { fit = JSON.parse(slot.getAttribute("data-fit-product")); } catch (e) { return; }
    fetch(cfg.api + "/v1/store/charts/" + encodeURIComponent(fit.ref), { headers: { "X-Merchant-Key": cfg.merchant } })
      .then(function (r) { return r.status === 200 ? r.json() : null; })   // 204: the seller has not put a chart on the store
      .then(function (c) {
        if (!c || !c.sizes) return;
        var labels = Object.keys(c.sizes);
        var zones = ZONE_ORDER.filter(function (z) { return labels.some(function (l) { return c.sizes[l][z] !== undefined; }); });
        var modal = document.getElementById("chart-modal");
        if (modal) {
          var table = modal.querySelector("table");
          var html = "<tr><th>Size</th>" + zones.map(function (z) { return "<th>" + ZONE_TITLES[z] + " (cm)</th>"; }).join("") + "</tr>";
          labels.forEach(function (l) { html += "<tr><td>" + l + "</td>" + zones.map(function (z) { return "<td>" + (c.sizes[l][z] === undefined ? "" : c.sizes[l][z]) + "</td>"; }).join("") + "</tr>"; });
          table.innerHTML = html;
          var title = document.getElementById("chart-title"); if (title) title.textContent = "Size guide: garment measurements";
          var note = document.getElementById("chart-live-note") || document.createElement("p");
          note.id = "chart-live-note"; note.textContent = "Garment laid flat, full circumference, in cm. Measured by the seller (version " + c.version + ").";
          if (!note.parentNode) table.parentNode.insertBefore(note, table);
        }
        var sel = document.getElementById("size");
        if (sel) { var keep = sel.value; sel.innerHTML = ""; labels.forEach(function (l) { var o = document.createElement("option"); o.value = l; o.textContent = l; sel.appendChild(o); }); if (labels.indexOf(keep) >= 0) sel.value = keep; }
        fit.size_chart = { sizes: c.sizes, chart_type: "garment", unit: "cm", size_system: (fit.size_chart && fit.size_chart.size_system) || "ALPHA" };
        slot.setAttribute("data-fit-product", JSON.stringify(fit));
        if (window.MeasuredFit.remount) window.MeasuredFit.remount();
        document.body.setAttribute("data-chart-version", String(c.version));
      }, function () { /* the printed chart stays */ });
  }

  function initCart() {
    var box = document.getElementById("cart-lines");
    if (!box) return;
    function render() {
      var lines = load(), total = 0;
      box.innerHTML = "";
      if (!lines.length) box.textContent = "Your bag is empty.";
      lines.forEach(function (l, i) {
        total += l.price * l.qty;
        var row = document.createElement("div");
        row.className = "line";
        var two = sizesOf(lines, l.slug).length > 1;
        var label = document.createElement("span");
        label.textContent = l.title + " · size " + l.size + " · qty " + l.qty;
        if (two) { var t = document.createElement("span"); t.className = "tag"; t.textContent = "2 sizes of this item"; label.appendChild(t); }
        var rm = document.createElement("button");
        rm.type = "button"; rm.textContent = "Remove";
        rm.addEventListener("click", function () { var ls = load(); ls.splice(i, 1); save(ls); paint(); render(); });
        row.appendChild(label); row.appendChild(rm); box.appendChild(row);
      });
      document.getElementById("cart-total").textContent = lines.length ? "Subtotal: $" + total.toFixed(2) : "";
      var slugs = {};
      lines.forEach(function (l) { slugs[l.slug] = (slugs[l.slug] || 0) + (sizesOf(lines, l.slug).length > 1 ? 1 : 0); });
      var multi = Object.keys(slugs).filter(function (s) { return slugs[s] > 0; });
      document.getElementById("cart-two-sizes").textContent = multi.length
        ? "You have more than one size of the same item. At checkout we will help you choose which to keep." : "";
    }
    render();
    window.addEventListener("dtc:bag", function () { paint(); render(); });
  }

  paint();
  initProduct();
  initCart();
})();
