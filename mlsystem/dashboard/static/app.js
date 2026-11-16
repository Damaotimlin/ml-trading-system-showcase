/* Demo dashboard: renders the snapshot embedded in the page and polls /api/snapshot for the replay. */
(function () {
    "use strict";
    var $ = function (sel) { return document.querySelector(sel); };
    var snapshot = JSON.parse($("#snapshot").textContent);
    var sort = { key: "predict_time", dir: -1 };
    var auto = true;

    function el(tag, text, cls) {
        var node = document.createElement(tag);
        if (text !== undefined && text !== null) node.textContent = text;
        if (cls) node.className = cls;
        return node;
    }
    function fmt(value, digits) { return value === null || value === undefined ? "–" : Number(value).toFixed(digits); }
    function signed(value, digits) { return (value > 0 ? "+" : "") + fmt(value, digits); }
    function field(name) { return document.querySelector('[data-field="' + name + '"]'); }

    /* ---- theme ---- */
    var themeBtn = $("#themeToggle");
    function paintTheme() { themeBtn.textContent = document.documentElement.dataset.theme === "dark" ? "Light mode" : "Dark mode"; }
    themeBtn.addEventListener("click", function () {
        var next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
        document.documentElement.dataset.theme = next;
        try { localStorage.setItem("theme", next); } catch (e) {}
        paintTheme();
        render();
    });
    paintTheme();

    /* ---- cards ---- */
    function renderCards() {
        if (!field("signal-direction")) return;
        var c = snapshot.cards, s = c.signal;
        var dir = field("signal-direction");
        dir.textContent = s.direction;
        dir.className = "big dir-" + s.direction;
        field("signal-detail").textContent = s.members
            ? "confidence " + fmt(s.confidence, 2) + " · forecast " + signed(s.forecast_pct, 3) + "% · " + s.members + " combos"
            : "no combo has earned a vote yet";
        var calls = c.recent_calls;
        field("calls-rate").textContent = calls.n ? Math.round(100 * calls.hits / calls.n) + "%" : "–";
        field("calls-detail").textContent = calls.hits + " of " + calls.n + " non-flat calls right, last " + calls.window + " bars";
        field("combos-admitted").textContent = c.combos.admitted + " of " + c.combos.total;
        field("combos-detail").textContent = c.combos.gated_out + " gated out, " +
            (c.combos.total - c.combos.admitted - c.combos.gated_out) + " not trusted yet";
        var outcome = field("decision-outcome");
        outcome.textContent = c.decision.outcome;
        var trail = field("decision-trail");
        trail.replaceChildren.apply(trail, c.decision.trail.map(function (step) {
            var li = el("li");
            li.append(el("b", step.decision + " → " + step.outcome));
            if (step.detail) li.append(" · " + step.detail);
            return li;
        }));
    }

    /* ---- chart (inline SVG, no library) ---- */
    var NS = "http://www.w3.org/2000/svg";
    function svg(tag, attrs) {
        var node = document.createElementNS(NS, tag);
        Object.keys(attrs).forEach(function (k) { node.setAttribute(k, attrs[k]); });
        return node;
    }
    function renderChart() {
        var box = $("#chart");
        if (!box) return;
        var data = snapshot.series, n = data.time.length;
        var W = Math.max(320, box.clientWidth), H = box.clientHeight || 300;
        var pad = { l: 52, r: 12, t: 10, b: 24 };
        var values = data.actual.concat(data.predicted.filter(function (v) { return v !== null; }));
        var lo = Math.min.apply(null, values), hi = Math.max.apply(null, values);
        var margin = (hi - lo) * 0.08 || 1; lo -= margin; hi += margin;
        var x = function (i) { return pad.l + (W - pad.l - pad.r) * i / (n - 1); };
        var y = function (v) { return pad.t + (H - pad.t - pad.b) * (1 - (v - lo) / (hi - lo)); };
        var css = getComputedStyle(document.documentElement);
        var color = function (name) { return css.getPropertyValue(name).trim(); };

        var root = svg("svg", { viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none" });
        for (var g = 0; g <= 4; g++) {
            var v = lo + (hi - lo) * g / 4;
            root.append(svg("line", { x1: pad.l, x2: W - pad.r, y1: y(v), y2: y(v), stroke: color("--grid"), "stroke-width": 1 }));
            var label = svg("text", { x: pad.l - 6, y: y(v) + 4, "text-anchor": "end" });
            label.textContent = v.toFixed(2);
            root.append(label);
        }
        var ticks = Math.max(2, Math.floor((W - pad.l - pad.r) / 140));
        for (var t = 0; t <= ticks; t++) {
            var i = Math.round((n - 1) * t / ticks);
            var tick = svg("text", { x: x(i), y: H - 6, "text-anchor": t === 0 ? "start" : t === ticks ? "end" : "middle" });
            tick.textContent = data.time[i].slice(5);
            root.append(tick);
        }
        function path(series) {
            var d = "", pen = false;
            series.forEach(function (v, i) {
                if (v === null) { pen = false; return; }
                d += (pen ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1);
                pen = true;
            });
            return d;
        }
        root.append(svg("path", { d: path(data.predicted), fill: "none", stroke: color("--predicted"), "stroke-width": 1.5, "stroke-dasharray": "4 3" }));
        root.append(svg("path", { d: path(data.actual), fill: "none", stroke: color("--accent"), "stroke-width": 2 }));
        var cross = svg("line", { y1: pad.t, y2: H - pad.b, stroke: color("--muted"), "stroke-width": 1, visibility: "hidden" });
        root.append(cross);
        box.replaceChildren(root);

        var tip = $("#chart-tip");
        root.addEventListener("mousemove", function (ev) {
            var rect = root.getBoundingClientRect();
            var px = (ev.clientX - rect.left) * W / rect.width;
            var i = Math.max(0, Math.min(n - 1, Math.round((px - pad.l) / (W - pad.l - pad.r) * (n - 1))));
            cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
            tip.replaceChildren(el("div", data.time[i]), el("div", "Actual " + fmt(data.actual[i], 3)),
                el("div", "Predicted " + fmt(data.predicted[i], 3) + " (" + data.direction[i] + ")"));
            tip.hidden = false;
            var left = ev.clientX - rect.left + 14;
            if (left + tip.offsetWidth > rect.width) left = ev.clientX - rect.left - tip.offsetWidth - 14;
            tip.style.left = left + "px";
            tip.style.top = (ev.clientY - rect.top + 30) + "px";
        });
        root.addEventListener("mouseleave", function () { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
    }

    /* ---- predictions table ---- */
    function visibleRows() {
        var model = $("#filter-model").value, result = $("#filter-result").value;
        var minScore = parseFloat($("#filter-score").value) || 0;
        var rows = snapshot.rows.filter(function (r) {
            return (!model || r.model === model) && r.combo_score >= minScore &&
                (!result || (result === "hit") === r.hit);
        });
        rows.sort(function (a, b) {
            var av = a[sort.key], bv = b[sort.key];
            return (av < bv ? -1 : av > bv ? 1 : 0) * sort.dir;
        });
        return rows;
    }
    function renderTable() {
        var table = $("#predictions");
        if (!table) return;
        var select = $("#filter-model");
        if (select.options.length === 1) {
            Array.from(new Set(snapshot.rows.map(function (r) { return r.model; }))).forEach(function (name) {
                select.append(new Option(name, name));
            });
        }
        var rows = visibleRows();
        table.tBodies[0].replaceChildren.apply(table.tBodies[0], rows.map(function (r) {
            var tr = el("tr");
            tr.append(el("td", r.live_time), el("td", fmt(r.live_value, 3), "num"), el("td", r.predict_time),
                el("td", fmt(r.predicted_value, 3), "num"), el("td", signed(r.diff_pct, 3), "num"), el("td", r.interval),
                el("td", r.model), el("td", fmt(r.combo_score, 3), "num"), el("td", signed(r.error_pct, 3), "num"),
                el("td", r.hit ? "right" : "wrong", r.hit ? "hit" : "miss"));
            return tr;
        }));
        $("#row-count").textContent = rows.length + " of " + snapshot.rows.length + " predictions";
        table.querySelectorAll("th[data-key]").forEach(function (th) {
            if (th.dataset.key === sort.key) th.setAttribute("aria-sort", sort.dir > 0 ? "ascending" : "descending");
            else th.removeAttribute("aria-sort");
        });
    }
    function exportCsv() {
        var keys = ["live_time", "live_value", "predict_time", "predicted_value", "diff_pct", "interval", "model", "combo_score", "error_pct", "hit"];
        var lines = [keys.join(",")].concat(visibleRows().map(function (r) { return keys.map(function (k) { return r[k]; }).join(","); }));
        var link = el("a");
        link.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
        link.download = "demo_predictions.csv";
        link.click();
        URL.revokeObjectURL(link.href);
    }

    /* ---- model health table ---- */
    function renderHealth() {
        var table = $("#health");
        if (!table) return;
        var body = table.tBodies[0], out = [];
        snapshot.combos.forEach(function (c) {
            var tr = el("tr");
            var gate = el("td"), status = el("td");
            gate.append(el("span", c.gate === "pass" ? "pass" : "rejected", "badge " + (c.gate === "pass" ? "good" : "bad")));
            status.append(el("span", c.status, "badge " + (c.status === "admitted" ? "good" : c.status === "gated out" ? "bad" : "wait")));
            tr.append(el("td", c.name), el("td", c.features, "num"),
                el("td", c.trials === null ? "–" : c.trials + " of 80", "num"),
                el("td", c.in_sample === null ? "–" : signed(c.in_sample, 3), "num"),
                el("td", c.out_of_sample === null ? "–" : signed(c.out_of_sample, 3), "num"), gate,
                el("td", c.graded ? c.hits + "/" + c.graded : "–", "num"),
                el("td", c.score === null ? "–" : signed(c.score, 3), "num"),
                el("td", c.hit_floor === null ? "–" : (100 * c.hit_floor).toFixed(1) + "%", "num"), status);
            out.push(tr);
            if (c.status !== "admitted") {
                var why = el("tr", null, "reason-row"), cell = el("td", "Why: " + c.status_reason);
                cell.colSpan = 10;
                why.append(cell);
                tr.className = "has-reason";
                out.push(why);
            }
        });
        body.replaceChildren.apply(body, out);
    }

    function render() {
        $("#as-of").textContent = snapshot.meta.as_of;
        renderCards(); renderChart(); renderTable(); renderHealth();
    }

    /* ---- wiring ---- */
    ["#filter-model", "#filter-score", "#filter-result"].forEach(function (sel) {
        if ($(sel)) $(sel).addEventListener("input", renderTable);
    });
    document.querySelectorAll("#predictions th[data-key]").forEach(function (th) {
        th.addEventListener("click", function () {
            sort = { key: th.dataset.key, dir: sort.key === th.dataset.key ? -sort.dir : 1 };
            renderTable();
        });
    });
    if ($("#export-csv")) $("#export-csv").addEventListener("click", exportCsv);
    var autoBtn = $("#autoToggle");
    if (autoBtn) autoBtn.addEventListener("click", function () {
        auto = !auto;
        autoBtn.textContent = "Auto update: " + (auto ? "on" : "off");
        autoBtn.setAttribute("aria-pressed", String(auto));
    });
    window.addEventListener("resize", renderChart);

    var pinned = new URLSearchParams(location.search).has("cursor");
    setInterval(function () {
        if (!auto || pinned || document.hidden) return;
        fetch("api/snapshot").then(function (r) { return r.json(); }).then(function (next) { snapshot = next; render(); }).catch(function () {});
    }, snapshot.meta.replay_seconds_per_bar * 1000);

    render();
})();
