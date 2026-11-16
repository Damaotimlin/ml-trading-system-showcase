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
        var minutes = function (bars) { var m = bars * snapshot.series.bar_minutes; return m < 60 ? m + " min" : (m / 60) + " h"; };
        field("signal-horizon").textContent = minutes(s.bars_ahead);
        var dir = field("signal-direction");
        dir.textContent = s.direction;
        dir.className = "big dir-" + s.direction;
        field("signal-detail").textContent = s.members
            ? "expected move " + signed(s.forecast_pct, 2) + "% by " + s.until.slice(5) + " · confidence " + fmt(s.confidence, 2) + " · " + s.members + (s.members === 1 ? " combo" : " combos")
            : "no combo has earned a vote yet";
        var badge = $("#trend-badge");
        badge.textContent = s.direction;
        badge.className = "trend-badge dir-" + s.direction;
        $("#trend-detail").textContent = s.members
            ? "next " + minutes(s.bars_ahead) + " " + signed(s.forecast_pct, 2) + "% · confidence " + fmt(s.confidence, 2)
            : "no combo may vote yet";
        var calls = c.recent_calls;
        field("calls-rate").textContent = calls.n ? Math.round(100 * calls.hits / calls.n) + "%" : "–";
        field("calls-detail").textContent = calls.hits + " of " + calls.n + " calls made " + minutes(calls.bars_ahead) +
            " ahead were right (last " + calls.window + " bars, overlapping)";
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
    var hidden = { pred: true };          /* legend state: series switched off (individual forecasts start hidden) */
    var overlayOpacity = 1;

    function renderChart() {
        var box = $("#chart");
        if (!box) return;
        var data = snapshot.series, n = data.x.length, now = data.now_index;
        var W = Math.max(320, box.clientWidth), H = box.clientHeight || 420;
        var pad = { l: 64, r: 14, t: 10, b: 28 };
        var css = getComputedStyle(document.documentElement);
        var color = function (name) { return css.getPropertyValue(name).trim(); };
        var show = function (key) { return !hidden[key]; };

        var values = data.actual.concat(data.avg_pred).filter(function (v) { return v !== null; });
        data.locks.forEach(function (lock) { if (show(lock.kind)) values = values.concat(lock.y); });
        var lo = Math.min.apply(null, values), hi = Math.max.apply(null, values);
        var margin = (hi - lo) * 0.1 || 1; lo -= margin * 1.6; hi += margin;
        var x0 = data.x[0], x1 = data.x[n - 1];
        var x = function (v) { return pad.l + (W - pad.l - pad.r) * (v - x0) / (x1 - x0); };
        var y = function (v) { return pad.t + (H - pad.t - pad.b) * (1 - (v - lo) / (hi - lo)); };
        var clip = function (v) { return Math.max(x0, Math.min(x1, v)); };

        var root = svg("svg", { viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none" });
        function add(tag, attrs, text) {
            var node = svg(tag, attrs);
            if (text !== undefined) node.textContent = text;
            root.append(node);
            return node;
        }
        function linePath(xs, ys) {
            var d = "", pen = false;
            xs.forEach(function (xv, i) {
                if (ys[i] === null || ys[i] === undefined) { pen = false; return; }
                d += (pen ? "L" : "M") + x(xv).toFixed(1) + " " + y(ys[i]).toFixed(1);
                pen = true;
            });
            return d;
        }
        function diamond(cx, cy, fill) {
            add("path", { d: "M" + cx + " " + (cy - 4.5) + "l4.5 4.5l-4.5 4.5l-4.5 -4.5z", fill: fill, opacity: overlayOpacity });
        }

        /* session bands: pre-market, regular, post-market */
        data.days.forEach(function (day) {
            [["premarket", "--band-pre"], ["regular", "--band-regular"], ["postmarket", "--band-post"]].forEach(function (band) {
                var a = clip(day[band[0]][0]), b = clip(day[band[0]][1]);
                if (b > a) add("rect", { x: x(a), y: pad.t, width: x(b) - x(a), height: H - pad.t - pad.b, fill: color(band[1]) });
            });
            if (day.midpoint > x0 && day.midpoint < x1) {
                add("line", { x1: x(day.midpoint), x2: x(day.midpoint), y1: pad.t, y2: H - pad.b, stroke: color("--down"),
                    "stroke-width": 1, "stroke-dasharray": "6 5", opacity: 0.7 });
            }
            if (day.midnight >= x0 && day.midnight <= x1) {
                add("text", { x: x(day.midnight), y: H - 8, "text-anchor": "middle" }, day.label + " 00:00");
            }
        });
        for (var g = 0; g <= 5; g++) {
            var v = lo + (hi - lo) * g / 5;
            add("line", { x1: pad.l, x2: W - pad.r, y1: y(v), y2: y(v), stroke: color("--grid"), "stroke-width": 1, opacity: 0.6 });
            add("text", { x: pad.l - 8, y: y(v) + 4, "text-anchor": "end" }, "$" + v.toFixed(2));
        }
        var axis = add("text", { x: 14, y: (H - pad.b) / 2, "text-anchor": "middle", transform: "rotate(-90 14 " + (H - pad.b) / 2 + ")" }, "Price ($)");
        axis.setAttribute("class", "axis-title");

        /* flat zones and locked forecast paths */
        var lockColor = { premarket: "--lock-pre", regular: "--lock-regular", midpoint: "--lock-mid" };
        data.locks.forEach(function (lock) {
            if (!show(lock.kind)) return;
            var a = lock.x[0], b = lock.x[lock.x.length - 1];
            if (b < x0) return;
            if (show("flat")) {
                add("rect", { x: x(clip(a)), y: y(lock.flat[1]), width: x(b) - x(clip(a)), height: Math.max(1, y(lock.flat[0]) - y(lock.flat[1])),
                    fill: color("--flat-zone"), "fill-opacity": 0.12, stroke: color("--flat-zone"), "stroke-width": 1,
                    "stroke-dasharray": "2 3", opacity: overlayOpacity });
            }
            var stroke = color(lockColor[lock.kind]);
            add("path", { d: linePath(lock.x.map(clip), lock.y), fill: "none", stroke: stroke, "stroke-width": 2,
                "stroke-dasharray": "7 5", opacity: overlayOpacity });
            if (a >= x0) diamond(x(a), y(lock.y[0]), stroke);
            diamond(x(b), y(lock.y[lock.y.length - 1]), stroke);
        });

        /* AvgPred runs past the live line; live stops at "now" */
        if (show("avg")) add("path", { d: linePath(data.x, data.avg_pred), fill: "none", stroke: color("--avg-pred"), "stroke-width": 1.75, opacity: overlayOpacity });
        if (show("live")) {
            add("path", { d: linePath(data.x, data.actual), fill: "none", stroke: color("--live"), "stroke-width": 2 });
            data.actual.forEach(function (v, i) {
                if (v !== null) add("circle", { cx: x(data.x[i]), cy: y(v), r: 1.8, fill: color("--live") });
            });
            add("circle", { cx: x(data.x[now]), cy: y(data.actual[now]), r: 4.5, fill: color("--down") });
        }
        if (show("pred")) {
            var predColor = ["--lock-pre", "--lock-regular", "--up", "--lock-mid", "--down"];
            data.open_predictions.forEach(function (p) {
                var stroke = color(predColor[data.horizons.indexOf(p.bars_ahead) % predColor.length]);
                add("line", { x1: x(p.x_from), y1: y(p.y_from), x2: x(p.x), y2: y(p.y), stroke: stroke, "stroke-width": 1,
                    "stroke-dasharray": "1 3", opacity: 0.6 * overlayOpacity });
                var cx = x(p.x), cy = y(p.y);
                add("path", { d: "M" + (cx - 3.5) + " " + (cy - 3.5) + "l7 7M" + (cx - 3.5) + " " + (cy + 3.5) + "l7 -7", stroke: stroke,
                    "stroke-width": 1.75, fill: "none", opacity: overlayOpacity });
            });
        }
        add("line", { x1: x(data.x[now]), x2: x(data.x[now]), y1: pad.t, y2: H - pad.b, stroke: color("--muted"), "stroke-width": 1, opacity: 0.6 });
        add("text", { x: x(data.x[now]) + 5, y: pad.t + 12 }, "ahead");

        /* live price box, bottom left */
        var label = "Live $" + data.actual[now].toFixed(2);
        add("rect", { x: pad.l + 10, y: H - pad.b - 34, width: label.length * 7.4 + 14, height: 24, fill: color("--panel"), stroke: color("--lock-pre"), "stroke-width": 1 });
        var liveText = add("text", { x: pad.l + 17, y: H - pad.b - 17 }, label);
        liveText.setAttribute("class", "live-box");

        var cross = add("line", { y1: pad.t, y2: H - pad.b, stroke: color("--muted"), "stroke-width": 1, visibility: "hidden" });
        box.replaceChildren(root);

        var tip = $("#chart-tip");
        root.addEventListener("mousemove", function (ev) {
            var rect = root.getBoundingClientRect();
            var xv = x0 + ((ev.clientX - rect.left) * W / rect.width - pad.l) / (W - pad.l - pad.r) * (x1 - x0);
            var i = 0;
            for (var k = 1; k < n; k++) if (Math.abs(data.x[k] - xv) < Math.abs(data.x[i] - xv)) i = k;
            cross.setAttribute("x1", x(data.x[i])); cross.setAttribute("x2", x(data.x[i])); cross.setAttribute("visibility", "visible");
            var lines = [el("div", "ET: " + data.time[i] + (i > now ? " · " + (i - now) * data.bar_minutes + " min ahead" : ""))];
            if (data.actual[i] !== null) lines.push(el("div", "Live: $" + fmt(data.actual[i], 2)));
            if (data.avg_pred[i] !== null) lines.push(el("div", "Avg prediction: $" + fmt(data.avg_pred[i], 2)));
            tip.replaceChildren.apply(tip, lines);
            tip.hidden = false;
            var left = ev.clientX - rect.left + 14;
            if (left + tip.offsetWidth > rect.width) left = ev.clientX - rect.left - tip.offsetWidth - 14;
            tip.style.left = left + "px";
            tip.style.top = (ev.clientY - rect.top + 40) + "px";
        });
        root.addEventListener("mouseleave", function () { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
    }

    document.querySelectorAll(".legend [data-series]").forEach(function (button) {
        var key = button.dataset.series;
        button.setAttribute("aria-pressed", String(!hidden[key]));
        button.addEventListener("click", function () {
            hidden[key] = !hidden[key];
            button.setAttribute("aria-pressed", String(!hidden[key]));
            renderChart();
        });
    });
    if ($("#opacity")) $("#opacity").addEventListener("input", function (ev) {
        overlayOpacity = ev.target.value / 100;
        $("#opacity-value").textContent = ev.target.value + "%";
        renderChart();
    });

    /* ---- predictions table ---- */
    function visibleRows() {
        var model = $("#filter-model").value, result = $("#filter-result").value;
        var horizon = $("#filter-horizon").value;
        var minScore = parseFloat($("#filter-score").value);
        if (isNaN(minScore)) minScore = -Infinity;
        var rows = snapshot.rows.filter(function (r) {
            return (!model || r.model === model) && (!horizon || String(r.bars_ahead) === horizon) && r.combo_score >= minScore &&
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
        var horizonSelect = $("#filter-horizon");
        if (horizonSelect.options.length === 1) {
            Array.from(new Set(snapshot.rows.map(function (r) { return r.bars_ahead; }))).sort(function (p, q) { return p - q; })
                .forEach(function (h) { horizonSelect.append(new Option(h, h)); });
        }
        var rows = visibleRows();
        table.tBodies[0].replaceChildren.apply(table.tBodies[0], rows.map(function (r) {
            var tr = el("tr");
            tr.append(el("td", r.live_time), el("td", "$" + fmt(r.live_value, 2), "num"), el("td", r.predict_time),
                el("td", r.bars_ahead, "num"),
                el("td", "$" + fmt(r.predicted_value, 2), "num"), el("td", signed(r.diff_pct, 3), "num"), el("td", r.interval),
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
        var keys = ["live_time", "live_value", "predict_time", "bars_ahead", "predicted_value", "diff_pct", "interval", "model", "combo_score", "error_pct", "hit"];
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

        var skill = $("#horizon-skill").tBodies[0];
        skill.replaceChildren.apply(skill, snapshot.horizon_skill.map(function (h) {
            var tr = el("tr"), rate = h.n ? h.hits / h.n : null, bar = el("td");
            var meter = el("span", null, "meter"), fill = el("span");
            fill.style.width = (rate === null ? 0 : Math.round(rate * 100)) + "%";
            meter.append(fill);
            bar.append(meter, el("span", rate === null ? "–" : (rate * 100).toFixed(1) + "%"));
            tr.append(el("td", h.bars_ahead, "num"), el("td", (h.bars_ahead * 30 < 60 ? h.bars_ahead * 30 + " min" : h.bars_ahead * 30 / 60 + " h"), "num"), el("td", h.n, "num"),
                el("td", h.hits, "num"), bar);
            return tr;
        }));
    }

    function render() {
        $("#as-of").textContent = snapshot.meta.as_of + " ET";
        renderCards(); renderChart(); renderTable(); renderHealth();
    }

    /* ---- wiring ---- */
    ["#filter-model", "#filter-horizon", "#filter-score", "#filter-result"].forEach(function (sel) {
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
