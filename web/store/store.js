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
      else lines.push({ slug: slug, title: body.getAttribute("data-title"), price: parseFloat(body.getAttribute("data-price")), size: size, qty: qty });
      save(lines);
      paint();
      var sizes = sizesOf(lines, slug), note = document.getElementById("bag-note");
      note.textContent = sizes.length > 1
        ? "Added size " + size + ". You now have sizes " + sizes.join(" and ") + " of this item in your bag."
        : "Added size " + size + " to your bag.";
    });
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
  }

  paint();
  initProduct();
  initCart();
})();
