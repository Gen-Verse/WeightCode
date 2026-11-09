// WeightCode project page: interactions only. Every number shown here is copied from the paper (assets/WeightCode.pdf).
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const NS = "http://www.w3.org/2000/svg";
  const REDUCE = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(tag, attrs, parent, text) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
    if (text !== undefined) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }
  const signed = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2);
  function onVisible(node, fn, threshold = 0.35) {
    if (!("IntersectionObserver" in window)) { fn(); return; }
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { io.disconnect(); fn(); }
    }, { threshold });
    io.observe(node);
  }

  /* =====================================================================
     1. The runner: install, rerun, stack
     ===================================================================== */
  const svg = $("#stage");
  const readout = $("#readout");
  const codeLines = $$("#code .ln[data-g]");
  const btn = { install: $("#b-install"), rerun: $("#b-rerun"), stack: $("#b-stack"), reset: $("#b-reset") };
  const timers = [];
  let raf = 0;

  const later = (ms, f) => timers.push(setTimeout(f, REDUCE ? 0 : ms));
  function stopAll() {
    timers.splice(0).forEach(clearTimeout);
    cancelAnimationFrame(raf);
    raf = 0;
  }
  function highlight(group) {
    codeLines.forEach((l) => l.classList.toggle("hi", group !== null && l.dataset.g === group));
  }
  function shell(which) {
    $("#shell-rerun").classList.toggle("on", which === "rerun");
    $("#shell-stack").classList.toggle("on", which === "stack");
  }
  function say(big, sub, tag, cls) {
    readout.innerHTML =
      `<div class="big ${cls || ""}">${big}</div>` +
      (sub ? `<div class="sub">${sub}</div>` : "") +
      (tag ? `<div class="tag">${tag}</div>` : "");
  }
  function clearStage() { while (svg.firstChild) svg.removeChild(svg.firstChild); }
  function wire(d, cls) {
    const p = el("path", { d, class: "wire " + cls }, svg);
    if (!REDUCE) {
      p.style.setProperty("--len", String(p.getTotalLength()));
      p.classList.add("draw");
      later(900, () => p.classList.remove("draw"));            // the finished wire never depends on the animation
    }
    return p;
  }
  function growBar(rect, width, delay) {
    rect.style.width = "0px";
    later(delay, () => { rect.style.width = width + "px"; });
  }

  // --- the block-2 MLP: 512 units on a 32 x 16 grid; the program uses units 360 and 503 ---
  const GX = 165, GY = 52, STEP = 10;
  const unitXY = (i) => [GX + (i % 32) * STEP, GY + Math.floor(i / 32) * STEP];
  const GATE = [360, 503];
  const IN = { x: 12, y: 104, w: 112, h: 50 };
  const OUT = { x: 516, y: 104, w: 114, h: 50 };
  let units = [];
  let gateNode = null;

  function drawNetwork() {
    clearStage();
    el("text", { x: 320, y: 30, "text-anchor": "middle", class: "lbl dim" }, svg, "block-2 MLP of the ViT-Lite · 512 units");
    el("rect", { x: IN.x, y: IN.y, width: IN.w, height: IN.h, rx: 10, class: "node" }, svg);
    el("text", { x: IN.x + IN.w / 2, y: IN.y + 30, "text-anchor": "middle", class: "lbl mono" }, svg, "patch token z");
    el("rect", { x: OUT.x, y: OUT.y, width: OUT.w, height: OUT.h, rx: 10, class: "node" }, svg);
    el("text", { x: OUT.x + OUT.w / 2, y: OUT.y + 30, "text-anchor": "middle", class: "lbl" }, svg, "residual stream");
    units = [];
    for (let i = 0; i < 512; i++) {
      const [x, y] = unitXY(i);
      units.push(el("circle", { cx: x, cy: y, r: 3, class: "u" }, svg));
    }
  }

  function stageIdle() {
    stopAll(); highlight(null); shell(null);
    drawNetwork();
    say('<span class="from">A trained ViT-Lite under Gaussian noise.</span>',
      "Its block-2 MLP has 512 units. Press <b>▶ Install</b> to run the program on it.");
    btn.rerun.disabled = true; btn.stack.disabled = true;
  }

  function stageInstall() {
    stopAll(); shell(null);
    drawNetwork();
    btn.rerun.disabled = true; btn.stack.disabled = true;
    // 1 free
    highlight("free");
    say('<span class="from">1 · free</span>',
      "The program measures the network on its own images and picks the two units used least on clean images: 360 and 503.");
    units.forEach((u, i) => { if (!GATE.includes(i)) u.setAttribute("opacity", "0.45"); });
    GATE.forEach((i) => {
      const [x, y] = unitXY(i);
      units[i].setAttribute("class", "u on"); units[i].setAttribute("r", "6");
      el("text", { x, y: y - 11, "text-anchor": "middle", class: "lbl mono halo fade-in" }, svg, String(i));
    });
    // 2 detect
    later(1700, () => {
      highlight("detect");
      say('<span class="from">2 · detect</span>',
        "Their input weights get a noise direction <i>d</i> and a threshold <i>t</i> above the clean range, so they switch on only for noisy tokens.");
      GATE.forEach((i) => {
        const [x, y] = unitXY(i);
        wire(`M${IN.x + IN.w} ${IN.y + IN.h / 2} C ${IN.x + IN.w + 50} ${IN.y + IN.h / 2}, ${x - 50} ${y}, ${x - 7} ${y}`, "in");
      });
      el("text", { x: IN.x + 4, y: IN.y - 12, class: "lbl mono orange halo fade-in" }, svg, "W1, b1: detect");
    });
    // 3 correct
    later(3400, () => {
      highlight("correct");
      say('<span class="from">3 · correct</span>',
        "Their output weights carry the noise shift measured on the data, with the opposite sign: an open gate subtracts it from the token.");
      GATE.forEach((i) => {
        const [x, y] = unitXY(i);
        wire(`M${x + 7} ${y} C ${x + 70} ${y}, ${OUT.x - 50} ${OUT.y + OUT.h / 2}, ${OUT.x} ${OUT.y + OUT.h / 2}`, "");
      });
      el("text", { x: OUT.x + OUT.w - 4, y: OUT.y - 12, "text-anchor": "end", class: "lbl mono navy halo fade-in" }, svg, "W2: correct");
    });
    // done: tokens through the gate, then the result
    later(5100, () => {
      highlight(null);
      tokenStream();
      say("+32.79 pt net gain",
        "Noisy-image accuracy +32.68 pt, clean accuracy +0.11 pt, 3,846 weights changed. Best of 8 fine-tuning methods with the same images, checks and clean-loss bound: +22.78 pt.",
        "ViT-Lite on CIFAR-10, Gaussian noise, held-out test split. The full program writes the same gate form on the class token in blocks 4 and 6.",
        "navy");
      btn.rerun.disabled = false;
    });
  }

  // Noisy tokens (orange) open the gate and come out corrected (navy); clean tokens (grey) pass unchanged.
  function tokenStream() {
    const Y = 268, X0 = 26, X1 = 614, GATE_X = 320;
    el("line", { x1: X0, y1: Y, x2: X1, y2: Y, class: "axis" }, svg);
    gateNode = el("rect", { x: GATE_X - 26, y: Y - 15, width: 52, height: 30, rx: 8, class: "node" }, svg);
    el("text", { x: GATE_X, y: Y + 4, "text-anchor": "middle", class: "lbl mono dim" }, svg, "gate");
    const legend = [["tok", "noisy"], ["tok clean", "clean"], ["tok fixed", "corrected"]];
    legend.forEach(([cls, name], k) => {
      el("circle", { cx: 32 + k * 92, cy: 307, r: 5, class: cls }, svg);
      el("text", { x: 42 + k * 92, y: 311, class: "lbl dim" }, svg, name);
    });
    el("text", { x: X1, y: 311, "text-anchor": "end", class: "lbl dim" }, svg, "schematic");
    if (REDUCE) {
      [[90, false], [180, true], [420, true], [510, false]].forEach(([x, noisy]) =>
        el("circle", { cx: x, cy: Y, r: 6, class: noisy ? (x > GATE_X ? "tok fixed" : "tok") : "tok clean" }, svg));
      return;
    }
    const pattern = [true, false, true, true, false, true, false, false, true, false];
    let k = 0, last = 0, prev = performance.now();
    const toks = [];
    const frame = (now) => {
      const dt = Math.min(0.05, (now - prev) / 1000); prev = now;
      if (now - last > 650) {
        last = now;
        const noisy = pattern[k++ % pattern.length];
        toks.push({ x: X0, noisy, done: false, c: el("circle", { cx: X0, cy: Y, r: 6, class: noisy ? "tok" : "tok clean" }, svg) });
      }
      let open = false;
      for (let i = toks.length - 1; i >= 0; i--) {
        const t = toks[i];
        t.x += 120 * dt;
        t.c.setAttribute("cx", t.x.toFixed(1));
        if (t.noisy && Math.abs(t.x - GATE_X) < 26) open = true;
        if (t.noisy && !t.done && t.x > GATE_X) { t.done = true; t.c.setAttribute("class", "tok fixed"); }
        if (t.x > X1) { t.c.remove(); toks.splice(i, 1); }
      }
      gateNode.setAttribute("class", open ? "node open" : "node");
      GATE.forEach((i) => units[i].setAttribute("r", open ? "7.5" : "6"));
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
  }

  // Rerun: selection-split net gains of the noise program on five independently trained ViT-Lites (paper Fig. 4).
  const RERUN = [30.03, 27.18, 27.50, 28.55, 27.43];
  const RERUN_MEAN = 28.14, COPIED_MEAN = -0.10;
  function stageRerun() {
    stopAll(); highlight(null); shell("rerun");
    clearStage();
    const X0 = 172, S = 11;
    el("text", { x: 20, y: 24, class: "lbl dim" }, svg, "the same program, five independently trained ViT-Lites, no LLM call");
    el("line", { x1: X0, y1: 36, x2: X0, y2: 270, class: "axis" }, svg);
    RERUN.forEach((v, i) => {
      const y = 48 + i * 34;
      el("text", { x: 20, y: y + 13, class: "lbl" }, svg, `network ${i + 1}`);
      const r = el("rect", { x: X0, y, height: 18, rx: 3, class: "bar" }, svg);
      growBar(r, v * S, 150 + i * 260);
      later(150 + i * 260 + 500, () =>
        el("text", { x: X0 + v * S + 8, y: y + 13, class: "lbl mono navy fade-in" }, svg, signed(v)));
    });
    const mx = X0 + RERUN_MEAN * S;
    el("line", { x1: mx, y1: 42, x2: mx, y2: 214, class: "thresh" }, svg);
    el("text", { x: mx, y: 222 + 4, "text-anchor": "middle", class: "lbl mono dim" }, svg, `mean ${signed(RERUN_MEAN)}`);
    const yc = 240;
    el("text", { x: 20, y: yc + 13, class: "lbl red" }, svg, "weight edit copied");
    el("rect", { x: X0 + COPIED_MEAN * S, y: yc, width: Math.abs(COPIED_MEAN * S), height: 18, class: "bar red" }, svg);
    later(1600, () =>
      el("text", { x: X0 + 8, y: yc + 13, class: "lbl mono red fade-in" }, svg, `${signed(COPIED_MEAN)} (mean of the five)`));
    [0, 10, 20, 30].forEach((t) =>
      el("text", { x: X0 + t * S, y: 292, "text-anchor": "middle", class: "lbl mono dim" }, svg, String(t)));
    el("text", { x: X0 + 15 * S, y: 314, "text-anchor": "middle", class: "lbl dim" }, svg, "net gain under Gaussian noise (pt)");
    say(`${signed(RERUN_MEAN)} pt rerun <span class="from">vs</span> <span style="color:var(--terra-ink)">${signed(COPIED_MEAN)}</span> copied`,
      "Rerun with each network's own 64 + 64 labeled images, the program places its gates anew and keeps its gain. The first network's weight edit, copied unchanged, produces none.",
      "Mean over five ViT-Lites, held-out selection split (test split: +28.57). On the test split all ten programs keep 83% to 110% of their gain.",
      "navy");
    btn.stack.disabled = false;
  }

  // Stack: six second programs written for the noise-edited network add contrast robustness (paper Tab. 13, test split).
  const SECOND = [9.12, 12.42, 14.50, 5.23, 12.80, 13.42];
  const BAR = 10;
  function stageStack() {
    stopAll(); highlight(null); shell("stack");
    clearStage();
    const boxes = [[20, "W0  original", ""], [262, "W1  noise gate", "p1"], [504, "W2  + contrast", "p2"]];
    boxes.forEach(([x, name, cls]) => {
      el("rect", { x, y: 18, width: 118, height: 38, rx: 9, class: "node " + cls }, svg);
      el("text", { x: x + 59, y: 41, "text-anchor": "middle", class: "lbl mono" }, svg, name);
    });
    el("path", { d: "M138 37 H262", class: "wire in" }, svg);
    el("text", { x: 200, y: 31, "text-anchor": "middle", class: "lbl mono orange" }, svg, "p1");
    const p2 = wire("M380 37 H504", "purple");
    p2.setAttribute("marker-end", "");
    el("text", { x: 442, y: 31, "text-anchor": "middle", class: "lbl mono purple" }, svg, "p2");

    const X0 = 130, S = 30, top = 92;
    el("line", { x1: X0 + BAR * S, y1: top - 8, x2: X0 + BAR * S, y2: top + 6 * 25, class: "thresh" }, svg);
    el("text", { x: X0 + BAR * S, y: top - 13, "text-anchor": "middle", class: "lbl mono dim" }, svg, "bar fixed in advance: 10");
    SECOND.forEach((v, i) => {
      const y = top + 10 + i * 25;
      el("text", { x: 20, y: y + 4, class: "lbl" }, svg, `program ${i + 1}`);
      el("line", { x1: X0, y1: y, x2: X0 + 16 * S, y2: y, class: "axis" }, svg);
      later(200 + i * 220, () => {
        el("circle", { cx: X0 + v * S, cy: y, r: 6.5, class: (v >= BAR ? "dot" : "dot miss") + " fade-in" }, svg);
        el("text", { x: X0 + v * S + 12, y: y + 4, class: "lbl mono purple fade-in" }, svg, signed(v));
      });
    });
    [0, 5, 10, 15].forEach((t) =>
      el("text", { x: X0 + t * S, y: top + 6 * 25 + 22, "text-anchor": "middle", class: "lbl mono dim" }, svg, String(t)));
    el("text", { x: X0 + 8 * S, y: top + 6 * 25 + 44, "text-anchor": "middle", class: "lbl dim" }, svg,
      "contrast robustness added on W1 (pt) · filled: met the bar");
    say('<span style="color:var(--purple-ink)">+5.23 to +14.50 pt</span> added',
      "Each second program is written for the already edited network W1. The installed noise robustness moves by only −0.17 to +0.23 pt, and clean accuracy stays within 1.47 pt of the original.",
      "Held-out test split. Four of six pass the 10-point bar fixed before the sessions. None of the six writes the units of the noise gate.");
  }

  if (svg) {
    btn.install.addEventListener("click", stageInstall);
    btn.rerun.addEventListener("click", stageRerun);
    btn.stack.addEventListener("click", stageStack);
    btn.reset.addEventListener("click", stageIdle);
    stageIdle();
    let touched = false;
    Object.values(btn).forEach((b) => b.addEventListener("click", () => { touched = true; }));
    onVisible($("#runner"), () => later(900, () => { if (!touched) stageInstall(); }), 0.5);
    document.addEventListener("visibilitychange", () => { if (document.hidden) cancelAnimationFrame(raf); });
  }

  /* =====================================================================
     2. Install tabs
     ===================================================================== */
  const tabs = $$(".tab");
  function selectTab(tab) {
    tabs.forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
    $$(".panel").forEach((p) => p.classList.toggle("on", p.dataset.panel === tab.dataset.tab));
  }
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => selectTab(t));
    t.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      selectTab(next); next.focus();
    });
  });

  /* =====================================================================
     3. Programs against fine-tuning (paper Tab. 3, held-out test)
     p: first program; p2: second program (ImageNet); ft: best of 8 fine-tuning under the clean-loss bound;
     nb, nbc: best fine-tuning without the bound and its clean change (not reported on ImageNet);
     ref: best of 11 fixed references.
     ===================================================================== */
  const ROWS = [
    { n: "ImageNet · ResNet-18", s: "Gaussian noise", p: 10.47, p2: 17.62, ft: 8.67, nb: null, ref: 8.84 },
    { n: "ImageNet · ResNet-18", s: "contrast reduction", p: 8.84, p2: 14.26, ft: 3.38, nb: null, ref: 6.00 },
    { n: "ViT-Lite · CIFAR-10", s: "Gaussian noise", p: 32.79, ft: 22.78, nb: 29.11, nbc: -1.98, ref: 16.48 },
    { n: "ViT-Lite · CIFAR-10", s: "contrast reduction", p: 16.51, ft: 7.89, nb: 7.89, nbc: -0.34, ref: 8.25 },
    { n: "ResNet-18 · CIFAR-10", s: "Gaussian noise", p: 35.23, ft: 9.47, nb: 30.18, nbc: -2.15, ref: 9.22 },
    { n: "ResNet-18 · CIFAR-10", s: "contrast reduction", p: 21.28, ft: 10.01, nb: 26.70, nbc: -2.26, ref: 13.51 },
  ];
  const MAX = 38;
  const rowsBox = $("#rows");
  const pct = (v) => `${Math.max(0, v) / MAX * 100}%`;
  let shown = false;
  function barline(cls, label) {
    const line = document.createElement("div"); line.className = "barline";
    const track = document.createElement("div"); track.className = "track";
    const fill = document.createElement("div"); fill.className = "fill " + cls;
    track.appendChild(fill);
    const val = document.createElement("div"); val.className = "val" + (cls === "prog" ? " prog" : "");
    val.textContent = label;
    line.append(track, val);
    return { line, track, fill, val };
  }
  const rowEls = ROWS.map((r) => {
    const row = document.createElement("div"); row.className = "row";
    row.innerHTML = `<div class="name">${r.n}<small>${r.s}</small></div>`;
    const bars = document.createElement("div"); bars.className = "bars";
    const prog = barline("prog", r.p2 ? `${signed(r.p)} · ${signed(r.p2)}` : signed(r.p));
    let tick = null;
    if (r.p2) { tick = document.createElement("div"); tick.className = "tick"; tick.style.left = "0%"; prog.track.appendChild(tick); }
    const ft = barline("ft", signed(r.ft));
    const ref = barline("ref", signed(r.ref));
    bars.append(prog.line, ft.line, ref.line);
    row.appendChild(bars);
    rowsBox.appendChild(row);
    return { r, prog, tick, ft, ref };
  });
  function renderChart() {
    const nobound = $("#nobound").checked;
    rowEls.forEach(({ r, prog, tick, ft, ref }) => {
      prog.fill.style.width = shown ? pct(r.p) : "0%";
      if (tick) tick.style.left = shown ? `calc(${pct(r.p2)} - 1px)` : "0%";
      ref.fill.style.width = shown ? pct(r.ref) : "0%";
      const v = nobound ? r.nb : r.ft;
      ft.fill.classList.toggle("nobound", nobound);
      ft.fill.style.width = shown && v !== null ? pct(v) : "0%";
      ft.val.textContent = v === null ? "not reported" : nobound ? `${signed(v)} · clean ${signed(r.nbc)}` : signed(v);
    });
    $("#lg-ft").style.background = nobound ? "var(--pink)" : "var(--gb)";
    $("#lg-ft-t").textContent = nobound ? "Fine-tuning, best of 8, no clean-loss bound" : "Fine-tuning, best of 8";
    $("#chart-foot").textContent = nobound
      ? "Without the bound, fine-tuning passes the program only on ResNet-18 contrast (+26.70 against +21.28), and it pays 2.26 pt of clean accuracy for it. The program loses at most 0.5% of clean check images in every row."
      : "Under the same clean-loss bound, the program leads both fine-tuning and the fixed references on all six settings. On ImageNet the bar is the first program and the tick the second.";
  }
  if (rowsBox) {
    $("#nobound").addEventListener("change", renderChart);
    renderChart();
    onVisible(rowsBox, () => { shown = true; renderChart(); }, 0.25);
  }

  /* =====================================================================
     4. The gate (Eq. 2) with illustrative parameters, and the interventions (paper Fig. 3, selection split)
     ===================================================================== */
  const gateSvg = $("#gate");
  if (gateSvg) {
    const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    const T = 4.2, K_GAIN = 1.6, K = 5;                        // illustrative only
    const g = (s) => (gelu(K_GAIN * (s - T)) - gelu(K_GAIN * (s - T) - K)) / K;
    const PX = (s) => 50 + s * 45, PY = (v) => 196 - v * 146;
    const gauss = (s, m, sd) => Math.exp(-0.5 * ((s - m) / sd) ** 2);
    const bump = (m, sd, cls) => {
      let d = `M${PX(0)} ${PY(0)}`;
      for (let s = 0; s <= 10.001; s += 0.1) d += ` L${PX(s).toFixed(1)} ${(PY(0) - 52 * gauss(s, m, sd)).toFixed(1)}`;
      el("path", { d: d + ` L${PX(10)} ${PY(0)} Z`, class: "bump " + cls }, gateSvg);
    };
    bump(2.0, 0.75, "clean");
    bump(7.4, 0.85, "noisy");
    el("text", { x: PX(2.0), y: PY(0) - 60, "text-anchor": "middle", class: "lbl dim" }, gateSvg, "clean tokens");
    el("text", { x: PX(7.4), y: PY(0) - 60, "text-anchor": "middle", class: "lbl orange" }, gateSvg, "noisy tokens");
    el("line", { x1: PX(0), y1: PY(0), x2: PX(10), y2: PY(0), class: "axis" }, gateSvg);
    el("line", { x1: PX(0), y1: PY(0), x2: PX(0), y2: PY(1) - 8, class: "axis" }, gateSvg);
    el("line", { x1: PX(T), y1: PY(1) - 10, x2: PX(T), y2: PY(0), class: "thresh" }, gateSvg);
    el("text", { x: PX(T) + 6, y: PY(1) - 12, class: "lbl mono dim" }, gateSvg, "t");
    el("text", { x: PX(0) - 8, y: PY(0) + 4, "text-anchor": "end", class: "lbl mono dim" }, gateSvg, "0");
    el("text", { x: PX(0) - 8, y: PY(1) + 4, "text-anchor": "end", class: "lbl mono dim" }, gateSvg, "1");
    el("text", { x: PX(0), y: 20, class: "lbl dim" }, gateSvg, "share of the measured noise shift removed");
    el("text", { x: PX(5), y: 236, "text-anchor": "middle", class: "lbl dim" }, gateSvg, "token's projection on the noise direction, dᵀz");
    let d = "";
    for (let s = 0; s <= 10.001; s += 0.05) d += `${d ? " L" : "M"}${PX(s).toFixed(1)} ${PY(g(s)).toFixed(1)}`;
    el("path", { d, class: "curve" }, gateSvg);
    const guide = el("line", { class: "thresh" }, gateSvg);
    const dot = el("circle", { r: 7, class: "marker" }, gateSvg);
    const slider = $("#proj");
    const update = () => {
      const s = Number(slider.value) / 10, v = g(s), share = Math.max(0, Math.min(1, v));
      dot.setAttribute("cx", PX(s)); dot.setAttribute("cy", PY(v));
      guide.setAttribute("x1", PX(s)); guide.setAttribute("x2", PX(s));
      guide.setAttribute("y1", PY(v)); guide.setAttribute("y2", PY(0));
      $("#gate-val").textContent = `${Math.round(share * 100)}%`;
      $("#gate-state").textContent = s < T
        ? "clean range: gate closed, the token passes unchanged"
        : share < 0.9 ? "on the ramp: part of the shift is removed"
        : "noisy range: gate open, the measured noise shift is removed";
    };
    slider.addEventListener("input", update);
    update();
  }

  const AB = [
    { g: "interventions on the program" },
    { t: "full program", v: 31.75, main: true },
    { t: "block-2 gate removed", v: 10.05 },
    { t: "all three gates removed", v: 0.05 },
    { t: "two random units of the block", v: 31.86 },
    { t: "gates held open (clean −5.03)", v: 19.43 },
    { g: "other constructions, same inputs and rule" },
    { t: "gate from the plain mean difference", v: 31.13, other: true },
    { t: "gradient on the same 3,846 entries", v: 3.60, other: true },
    { t: "the LLM without weight values", v: 10.78, other: true },
  ];
  const abBox = $("#ablate");
  if (abBox) {
    const fills = [];
    AB.forEach((a) => {
      if (a.g) { const h = document.createElement("div"); h.className = "ab-group"; h.textContent = a.g; abBox.appendChild(h); return; }
      const row = document.createElement("div"); row.className = "ab";
      row.innerHTML = `<span class="t${a.main ? " main" : ""}">${a.t}</span><span class="tr"><i${a.other ? ' class="other"' : ""}></i></span><span class="v">${a.v.toFixed(2)}</span>`;
      abBox.appendChild(row);
      fills.push([row.querySelector("i"), a.v]);
    });
    onVisible(abBox, () => fills.forEach(([i, v], k) =>
      setTimeout(() => { i.style.width = `${v / 35 * 100}%`; }, REDUCE ? 0 : k * 90)), 0.3);
  }

  /* =====================================================================
     5. Skill terminal: the real prompt and the real commands of skills/weightcode/references/workflow.md
     ===================================================================== */
  const SCRIPT = [
    ["you", "Use $weightcode to improve my ResNet-18 on these new-condition examples while preserving clean recognition. My checkpoint is model.pt, and labels.json gives its class order. I have separate experience, selection and final image folders. Allow at most 1 percentage point of clean-accuracy loss. First inspect compatibility and propose the run budget."],
    ["who", "Loading the weightcode skill. Checking the engine, then binding your files."],
    ["cmd", "uv run weightcode doctor --sandbox-python .venv-sandbox/bin/python"],
    ["cmd", "uv run weightcode import-model --checkpoint model.pt --labels labels.json --out work/model"],
    ["cmd", "uv run weightcode prepare-data --model work/model --new images/experience/new --old images/experience/old --out work/experience.npz"],
    ["cmd", "uv run weightcode prepare-data --model work/model --new images/selection/new --old images/selection/old --out work/selection.npz"],
    ["cmd", "uv run weightcode prepare-data --model work/model --new images/final/new --old images/final/old --out work/final.npz"],
    ["cmd", "uv run weightcode prepare-task --model work/model --experience work/experience.npz --selection work/selection.npz --final work/final.npz --goal 'Recognize objects under the supplied corruption while retaining clean recognition' --retention-drop 0.01 --min-new-gain 0.01 --out work/task"],
    ["who", "Your files are compatible. Proposed budget: 8 checks, 100 Python cells, 30 minutes for the editor. Shall I run it?"],
    ["you", "yes"],
    ["cmd", "uv run weightcode synthesize --task work/task --sandbox-python .venv-sandbox/bin/python --editor-model sonnet --checks 8 --cells 100 --wall-s 1800 --out work/synthesis"],
    ["who", "Done. work/synthesis/final/report.json states one of three outcomes:"],
    ["eng", "  accepted        updated model.pt + program/ for other checkpoints\n  base-kept       no candidate met your thresholds\n  final-rejected  failed the final split; original kept"],
  ];
  const term = $("#term");
  let termRun = 0;
  function playTerm() {
    const run = ++termRun;
    term.textContent = "";
    let li = 0;
    const caret = document.createElement("span"); caret.className = "caret";
    const next = () => {
      if (run !== termRun) return;
      if (li >= SCRIPT.length) { term.appendChild(caret); return; }
      const [kind, text] = SCRIPT[li++];
      const prefix = { you: "you › ", who: "claude › ", cmd: "$ ", eng: "" }[kind];
      const line = document.createElement("div");
      const pre = document.createElement("span"); pre.className = kind === "cmd" ? "who" : kind === "eng" ? "" : "who";
      pre.textContent = prefix;
      const body = document.createElement("span"); body.className = { you: "you", who: "", cmd: "cmd", eng: "eng" }[kind];
      line.append(pre, body);
      term.appendChild(line);
      if (REDUCE || kind === "who" || kind === "eng") {
        body.textContent = text;
        setTimeout(next, REDUCE ? 0 : kind === "who" ? 650 : 400);
        return;
      }
      let c = 0;
      const speed = kind === "you" ? 4 : 7;                   // characters per tick
      const type = () => {
        if (run !== termRun) return;
        c = Math.min(text.length, c + speed);
        body.textContent = text.slice(0, c);
        body.appendChild(caret);
        if (c < text.length) setTimeout(type, 16);
        else { caret.remove(); setTimeout(next, kind === "you" ? 600 : 300); }
      };
      type();
    };
    next();
  }
  if (term) {
    $("#replay").addEventListener("click", playTerm);
    onVisible(term, playTerm, 0.3);
  }

  /* =====================================================================
     6. Highlight the section in view in the top navigation
     ===================================================================== */
  const navLinks = $$(".topnav a");
  if ("IntersectionObserver" in window && navLinks.length) {
    const byId = new Map(navLinks.map((a) => [a.getAttribute("href").slice(1), a]));
    const spy = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        navLinks.forEach((a) => a.classList.toggle("active", a === byId.get(e.target.id)));
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    byId.forEach((_, id) => { const sec = document.getElementById(id); if (sec) spy.observe(sec); });
  }

  /* =====================================================================
     7. Copy buttons
     ===================================================================== */
  $$(".copy").forEach((b) => b.addEventListener("click", async () => {
    const src = $("#" + b.dataset.copy);
    try {
      await navigator.clipboard.writeText(src.textContent);
      b.textContent = "Copied"; b.classList.add("done");
    } catch (err) {
      const range = document.createRange(); range.selectNodeContents(src);
      const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
      b.textContent = "Press Ctrl+C";
    }
    setTimeout(() => { b.textContent = "Copy"; b.classList.remove("done"); }, 1800);
  }));
})();
