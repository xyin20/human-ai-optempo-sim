(() => {
  const D = JSON.parse(document.getElementById('sim-data').textContent);
  const M = TinyLM.load(JSON.parse(document.getElementById('lm-weights').textContent));
  const T = D.T, PH = D.phases, ROLES = D.roles, SENS = D.sensors;
  const HB = Pipe.unpack2(D.bins_h, 4 * 9 * T);
  const AB = Pipe.unpack2(D.bins_head, 8 * 4 * T);
  const MB = Pipe.unpack2(D.bins_model, 3 * T);
  const hb = (i, s) => HB.subarray((i * 9 + s) * T, (i * 9 + s + 1) * T);
  const ab = (h, k) => AB.subarray((h * 4 + k) * T, (h * 4 + k + 1) * T);
  const mb = (k) => MB.subarray(k * T, (k + 1) * T);
  const NEURO = ['eeg_delta', 'eeg_theta', 'eeg_alpha', 'eeg_beta', 'hbo', 'hbr'].map((s) => SENS.indexOf(s));
  const AUTO = ['ibi', 'rsp', 'pupil'].map((s) => SENS.indexOf(s));
  const HEADS = ['L1·H1', 'L1·H2', 'L1·H3', 'L1·H4', 'L2·H1', 'L2·H2', 'L2·H3', 'L2·H4'];
  const LABEL = {
    ibi: 'Interbeat interval (ms)', rsp: 'Respiration rate (breaths/min)', pupil: 'Pupil diameter (mm)',
    eeg_delta: 'EEG delta power (a.u.)', eeg_theta: 'EEG theta power (a.u.)', eeg_alpha: 'EEG alpha power (a.u.)', eeg_beta: 'EEG beta power (a.u.)',
    hbo: 'fNIRS HbO (a.u.)', hbr: 'fNIRS HbR (a.u.)',
    head_attn_entropy: 'Attention entropy (bits)', head_partner_share: 'Attention on teammates’ words (share)',
    head_lookback: 'Look-back distance (tokens)', head_head_norm: 'Head output size (norm)',
    model_surprisal: 'Surprise at teammates’ words (bits)', model_uncertainty: 'Reply uncertainty (bits)', model_load: 'Tokens read in last 20 s',
  };
  const POOL_LABEL = {
    ibi: 'Interbeat interval', rsp: 'Respiration', pupil: 'Pupil', eeg_delta: 'EEG delta', eeg_theta: 'EEG theta', eeg_alpha: 'EEG alpha', eeg_beta: 'EEG beta', hbo: 'fNIRS HbO', hbr: 'fNIRS HbR',
    attn_entropy: 'AI · attention entropy', partner_share: 'AI · attention on teammates', lookback: 'AI · look-back', head_norm: 'AI · head output',
    surprisal: 'AI · surprise', uncertainty: 'AI · uncertainty', load: 'AI · load',
  };
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmt = (v, d = 2) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
  const el = (id) => document.getElementById(id);
  const roleColor = (r) => css(`--r${r}`);

  // ---------------------------------------------------------------- charts
  const charts = [];
  function makeChart(host, height, draw) {
    const cv = document.createElement('canvas');
    host.appendChild(cv);
    const render = () => {
      const w = host.clientWidth;
      if (!w) return;
      const dpr = window.devicePixelRatio || 1;
      cv.width = Math.round(w * dpr); cv.height = Math.round(height * dpr); cv.style.height = height + 'px';
      const ctx = cv.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, height);
      draw(ctx, w, height);
    };
    new ResizeObserver(render).observe(host);
    charts.push(render);
    return render;
  }
  const redrawAll = () => charts.forEach((f) => f());
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { redrawAll(); renderRaster(); });
  new MutationObserver(() => { redrawAll(); renderRaster(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // series: [{y, x0, dx, color, width, dash}], band: {lo, hi, x0, dx, color}
  function drawTime(ctx, w, h, opt) {
    const L = 46, R = 34, Tp = 20, B = 24;
    const pw = w - L - R, ph = h - Tp - B, xmax = opt.xmax || T;
    let ymin = opt.ymin, ymax = opt.ymax;
    if (ymin === undefined || ymax === undefined) {
      let lo = Infinity, hi = -Infinity;
      for (const s of opt.series) for (let i = s.skip || 0; i < s.y.length; i++) { const v = s.y[i]; if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
      if (opt.band) for (let i = 0; i < opt.band.lo.length; i++) { lo = Math.min(lo, opt.band.lo[i]); hi = Math.max(hi, opt.band.hi[i]); }
      const pad = (hi - lo) * 0.08 || 0.1;
      if (ymin === undefined) ymin = lo - pad;
      if (ymax === undefined) ymax = hi + pad;
    }
    const X = (t) => L + (t / xmax) * pw;
    const Y = (v) => Tp + (1 - (v - ymin) / (ymax - ymin)) * ph;
    const bands = [css('--band-1'), css('--band-2'), css('--band-3')];
    ctx.font = '11px ' + css('--font-mono');
    PH.forEach((p, k) => {
      ctx.fillStyle = bands[k];
      ctx.fillRect(X(p.start), Tp, X(p.end) - X(p.start), ph);
      ctx.fillStyle = css('--muted');
      ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      ctx.fillText(p.name, X(p.start) + 4, Tp - 6);
    });
    ctx.strokeStyle = css('--line'); ctx.lineWidth = 1; ctx.fillStyle = css('--muted');
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    const ticks = niceTicks(ymin, ymax, 4);
    for (const v of ticks) {
      const y = Y(v);
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke();
      ctx.fillText(tickLabel(v, ticks), L - 6, y);
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let m = 0; m <= xmax / 60; m += 10) ctx.fillText(m + (m === 60 ? ' min' : ''), X(m * 60), Tp + ph + 6);
    ctx.save(); ctx.beginPath(); ctx.rect(L, Tp, pw, ph); ctx.clip();
    if (opt.band) {
      const b = opt.band;
      ctx.fillStyle = b.color; ctx.globalAlpha = 0.28; ctx.beginPath();
      for (let i = 0; i < b.lo.length; i++) { const x = X(b.x0 + i * b.dx); i ? ctx.lineTo(x, Y(b.hi[i])) : ctx.moveTo(x, Y(b.hi[i])); }
      for (let i = b.lo.length - 1; i >= 0; i--) ctx.lineTo(X(b.x0 + i * b.dx), Y(b.lo[i]));
      ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
    }
    for (const s of opt.series) {
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.6; ctx.globalAlpha = s.alpha || 1;
      ctx.setLineDash(s.dash || []); ctx.lineJoin = 'round';
      ctx.beginPath();
      let started = false;
      for (let i = s.skip || 0; i < s.y.length; i++) {
        const v = s.y[i]; if (!Number.isFinite(v)) continue;
        const x = X(s.x0 + i * s.dx), y = Y(v);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    ctx.restore();
  }
  function niceTicks(lo, hi, n) {
    const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n + 0.5) || 10 * mag;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function tickLabel(v, ticks) {
    const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1;
    const d = step >= 1 ? 0 : step >= 0.1 ? 1 : step >= 0.01 ? 2 : 3;
    return fmt(v, d);
  }
  const legend = (id, items) => {
    el(id).innerHTML = items.map((it) => `<span><i class="${it.dash ? 'dash' : ''}" style="background-color:${it.dash ? 'transparent' : it.color};color:${it.color}"></i>${it.label}</span>`).join('');
  };

  // ---------------------------------------------------------------- raw series
  function rawSeries(key, idx) {
    const r = D.raw[key];
    const q = Pipe.unpackU16(r.q[idx]);
    const out = new Float64Array(q.length);
    for (let i = 0; i < q.length; i++) out[i] = r.lo + (q[i] / 4095) * (r.hi - r.lo);
    return out;
  }

  // ---------------------------------------------------------------- 0. phase cards
  const PHASE_TXT = [
    ['0–20 min', 'Separate', 'The AI works another team’s channel. People and AI share nothing.', 'people   ·   AI'],
    ['20–40 min', 'AI adapts', 'The AI reads this team’s chat and responds to it. Its replies are hidden from the team.', 'people  →  AI'],
    ['40–60 min', 'Mutual', 'The AI reads the team and the team reads the AI. Errors and warnings raise workload; good answers offload it.', 'people  ⇄  AI'],
  ];
  el('phase-cards').innerHTML = PHASE_TXT.map((p) => `<div class="phase-card"><span class="t">${p[0]}</span><b>${p[1]}</b><span class="flow">${p[3]}</span><p>${p[2]}</p></div>`).join('');

  // ---------------------------------------------------------------- 1. chat
  const GT = D.results.ground_truth;
  let chatPhase = 2;
  function renderChat() {
    el('chat-tabs').innerHTML = PH.map((p, k) => `<button type="button" role="tab" aria-selected="${k === chatPhase}" aria-pressed="${k === chatPhase}" data-k="${k}">${p.name}</button>`).join('');
    el('chat-tabs').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => { chatPhase = +b.dataset.k; renderChat(); }));
    const g = GT[chatPhase];
    const stat = (v, k) => `<div class="stat"><span class="v">${v}</span><span class="k">${k}</span></div>`;
    el('chat-stats').innerHTML = stat(g['team messages'], 'team messages') + stat(g['AI replies'], 'AI replies') + stat(Math.round(100 * g['AI error rate']) + '%', 'AI replies that were errors') + stat(Math.round(100 * g['brevity style']) + '%', 'brevity phrasing (generator)');
    const p = PH[chatPhase];
    let rows = D.chat.filter((m) => m[0] >= p.start && m[0] < p.end);
    let note = '';
    if (chatPhase === 0) {
      const ai = D.chat_other.filter((m) => m[1] === 'AI').slice(0, 12).map((m) => [m[0], 'AI*', m[2]]);
      rows = rows.slice(0, 40).concat(ai).sort((a, b) => a[0] - b[0]);
      note = '<div class="row"><span class="t"></span><span class="who" style="color:var(--ai)">AI*</span><span class="msg muted">* replies posted to the other team’s channel; this team never sees them</span></div>';
    } else rows = rows.slice(0, 60);
    const who = (s) => {
      const r = ROLES.indexOf(s);
      const c = r >= 0 ? `var(--r${r})` : 'var(--ai)';
      return `<span class="who" style="color:${c}">${s.toUpperCase()}</span>`;
    };
    const tt = (t) => `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`;
    el('chat-log').innerHTML = note + rows.map((m) => `<div class="row"><span class="t">${tt(m[0])}</span>${who(m[1])}<span class="msg">${escapeHtml(m[2].replace(/^<\w+>\s*/, ''))}</span></div>`).join('');
  }
  function escapeHtml(s) { return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
  renderChat();

  // ---------------------------------------------------------------- 2. signals
  el('sig-sensor').innerHTML = SENS.map((s) => `<option value="${s}">${LABEL[s]}</option>`).join('');
  el('sig-sensor').value = 'ibi';
  const humanChart = makeChart(el('chart-human'), 230, (ctx, w, h) => {
    const s = el('sig-sensor').value;
    drawTime(ctx, w, h, { series: [0, 1, 2, 3].map((i) => ({ y: rawSeries(s, i), x0: 2.5, dx: D.raw_step, color: roleColor(i), width: 1.3, alpha: 0.9 })) });
  });
  el('sig-sensor').addEventListener('change', humanChart);
  const AI_KEYS = ['head_attn_entropy', 'head_partner_share', 'head_lookback', 'head_head_norm', 'model_surprisal', 'model_uncertainty', 'model_load'];
  el('ai-sensor').innerHTML = AI_KEYS.map((k) => `<option value="${k}">${LABEL[k]}</option>`).join('');
  el('ai-sensor').value = 'model_load';
  const aiChart = makeChart(el('chart-ai'), 230, (ctx, w, h) => {
    const k = el('ai-sensor').value;
    const n = D.raw[k].q.length;
    const series = [];
    for (let i = 0; i < n; i++) series.push({ y: rawSeries(k, i), x0: 2.5, dx: D.raw_step, color: css('--ai'), width: n > 1 ? 1 : 1.6, alpha: n > 1 ? 0.55 : 1 });
    drawTime(ctx, w, h, { series });
    legend('legend-ai', n > 1 ? [{ color: css('--ai'), label: 'one line per attention head (8)' }] : [{ color: css('--ai'), label: 'whole model' }]);
  });
  el('ai-sensor').addEventListener('change', aiChart);
  legend('legend-human', ROLES.map((r, i) => ({ color: roleColor(i), label: r })));

  // ---------------------------------------------------------------- 3. live model
  const EX = [
    'p2: tgt grid 472198, 2 veh moving east.\np3: request fire mission grid 472198, 2 rds he.',
    'p4: there are maybe 3 trucks out by 914406 heading west i think.\np3: put some fire on them over there?',
    'p1: confirm tgt id.\np2: hey ai can you work out a firing solution for 305512?',
  ];
  el('lm-input').value = EX[0];
  document.querySelectorAll('[data-ex]').forEach((b) => b.addEventListener('click', () => { el('lm-input').value = EX[+b.dataset.ex]; el('lm-reply').hidden = true; runLM(); }));
  let lmHead = 6, lmState = null;
  const HUMAN_TAGS = ['<p1>', '<p2>', '<p3>', '<p4>'];
  function parseChat(text) {
    const toks = [], auth = [];
    for (const line of text.split('\n')) {
      const m = line.match(/^\s*(p[1-4]|ai)\s*:\s*(.*)$/i);
      if (!line.trim()) continue;
      const role = m ? m[1].toLowerCase() : 'p1';
      const body = (m ? m[2] : line).toLowerCase().replace(/(\d)(?=\d)/g, '$1 ');
      const words = TinyLM.tokenize(M, body);
      const tag = `<${role}>`;
      const a = role === 'ai' ? 'ai' : 'human';
      toks.push(tag); auth.push(a);
      for (const w of words) { toks.push(w); auth.push(a); }
    }
    return { toks, auth };
  }
  function analyze(toks, auth) {
    const ctx = M.cfg.ctx;
    const tk = toks.slice(-ctx), au = auth.slice(-ctx);
    const ids = tk.map((w) => M.stoi.get(w) ?? M.stoi.get('<unk>'));
    const out = TinyLM.forward(M, ids);
    const n = out.T, q = n - 1;
    const heads = [];
    for (let l = 0; l < out.att.length; l++) for (let h = 0; h < out.att[l].length; h++) {
      const A = out.att[l][h]; let ent = 0, share = 0, look = 0;
      const row = [];
      for (let s = 0; s <= q; s++) { const a = A[q * n + s]; row.push(a); if (a > 0) ent -= a * Math.log2(a); if (au[s] === 'human') share += a; look += a * (q - s); }
      heads.push({ row, ent, share, look });
    }
    const V = out.V, lg = out.logits.slice(q * V, (q + 1) * V);
    let mx = -Infinity; for (const v of lg) if (v > mx) mx = v;
    let sum = 0; const pr = new Float64Array(V); for (let i = 0; i < V; i++) { pr[i] = Math.exp(lg[i] - mx); sum += pr[i]; }
    let pent = 0; for (let i = 0; i < V; i++) { pr[i] /= sum; if (pr[i] > 0) pent -= pr[i] * Math.log2(pr[i]); }
    return { tk, au, heads, pr, pent };
  }
  function renderLM() {
    const S = lmState; if (!S) return;
    el('lm-heads').innerHTML = S.heads.map((h, i) => `<button type="button" aria-pressed="${i === lmHead}" data-h="${i}"><span class="hn">${HEADS[i]}</span><span class="hv">${h.ent.toFixed(2)} bits · ${Math.round(100 * h.share)}% team</span></button>`).join('');
    el('lm-heads').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => { lmHead = +b.dataset.h; renderLM(); }));
    const row = S.heads[lmHead].row; const mx = Math.max(...row);
    const aiC = css('--ai');
    el('lm-tokens').innerHTML = S.tk.map((t, i) => {
      const a = row[i] / mx;
      const unk = t === '<unk>';
      return `<span class="tok ${S.au[i]}" title="attention ${(100 * row[i]).toFixed(1)}%${unk ? ' · word not in the vocabulary' : ''}" style="background:${hexA(aiC, 0.08 + 0.8 * a)};${a > 0.55 ? 'color:' + css('--bg') + ';' : ''}${unk ? 'border-style:dashed;' : ''}">${escapeHtml(t)}</span>`;
    }).join('');
    el('lm-table').innerHTML = '<tr><th>Head</th><th class="n">Entropy</th><th class="n">On team</th><th class="n">Look-back</th></tr>' +
      S.heads.map((h, i) => `<tr${i === lmHead ? ' style="font-weight:600"' : ''}><td class="mono">${HEADS[i]}</td><td class="n">${h.ent.toFixed(2)}</td><td class="n">${Math.round(100 * h.share)}%</td><td class="n">${h.look.toFixed(1)}</td></tr>`).join('');
    const top = Array.from(S.pr).map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 6);
    el('lm-next').innerHTML = `<tr><th>Next word</th><th class="n">Probability</th></tr>` + top.map(([p, i]) => `<tr><td class="mono">${escapeHtml(M.vocab[i])}</td><td class="n">${(100 * p).toFixed(1)}%</td></tr>`).join('') +
      `<tr><td>Uncertainty</td><td class="n">${S.pent.toFixed(2)} bits</td></tr>`;
  }
  function hexA(hex, a) {
    const h = hex.replace('#', ''); const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a.toFixed(3)})`;
  }
  function runLM() {
    const { toks, auth } = parseChat(el('lm-input').value);
    if (!toks.length) { el('lm-status').textContent = 'Type at least one message.'; return; }
    toks.push('<ai>'); auth.push('ai');
    el('lm-status').textContent = 'running…';
    setTimeout(() => {
      const t0 = performance.now();
      lmState = analyze(toks, auth);
      el('lm-status').textContent = `${lmState.tk.length} tokens · ${Math.round(performance.now() - t0)} ms` + (toks.length > M.cfg.ctx ? ' · only the last 64 tokens fit the context' : '');
      renderLM();
    }, 10);
  }
  el('lm-run').addEventListener('click', runLM);
  el('lm-gen').addEventListener('click', () => {
    const { toks, auth } = parseChat(el('lm-input').value);
    if (!toks.length) return;
    toks.push('<ai>'); auth.push('ai');
    const words = [], ents = [];
    el('lm-status').textContent = 'the AI is answering…';
    el('lm-gen').disabled = true;
    const step = () => {
      const S = analyze(toks, auth);
      ents.push(S.pent);
      const cand = Array.from(S.pr).map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 8);
      const z = cand.reduce((s, c) => s + Math.pow(c[0], 1 / 0.8), 0);
      let r = Math.random() * z, pick = cand[0][1];
      for (const c of cand) { r -= Math.pow(c[0], 1 / 0.8); if (r <= 0) { pick = c[1]; break; } }
      const w = M.vocab[pick];
      if (w.startsWith('<') || words.length >= 22) return done();
      words.push(w); toks.push(w); auth.push('ai');
      if (w === '.' || w === '?') return done();
      setTimeout(step, 0);
    };
    const done = () => {
      const unc = ents.reduce((s, v) => s + v, 0) / ents.length;
      const text = words.join(' ').replace(/ ([,.?])/g, '$1').replace(/(\d) (?=\d)/g, '$1');
      el('lm-reply').hidden = false;
      el('lm-reply').innerHTML = `<span class="a">AI:</span> ${escapeHtml(text)} <span class="muted"> · mean uncertainty ${unc.toFixed(2)} bits</span>`;
      el('lm-input').value = el('lm-input').value.replace(/\s+$/, '') + '\nai: ' + text;
      el('lm-gen').disabled = false;
      runLM();
    };
    setTimeout(step, 10);
  });
  runLM();

  // ---------------------------------------------------------------- 4. pooling
  el('pool-method').innerHTML = Object.entries(Pipe.METHODS).map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
  const POOL_SENS = SENS.map((s) => ({ key: s, kind: 'human', idx: SENS.indexOf(s) }))
    .concat(D.head_sensors.map((s, k) => ({ key: s, kind: 'head', idx: k })))
    .concat(D.model_sensors.map((s, k) => ({ key: s, kind: 'model', idx: k })));
  el('pool-sensor').innerHTML = POOL_SENS.map((p, i) => `<option value="${i}">${POOL_LABEL[p.key]}</option>`).join('');
  el('pool-sensor').value = '0';
  let P = null;
  const UNPOOLED = (() => {
    const hcols = [], acols = [];
    for (let i = 0; i < 4; i++) for (let s = 0; s < 9; s++) hcols.push(hb(i, s));
    for (let h = 0; h < 8; h++) for (let k = 0; k < 4; k++) acols.push(ab(h, k));
    for (let k = 0; k < 3; k++) acols.push(mb(k));
    const hu = Pipe.encode(hcols), ai = Pipe.encode(acols), sy = Pipe.encode(hcols.concat(acols));
    return { hu, ai, sy };
  })();
  function computePools() {
    const m = el('pool-method').value;
    const hum = SENS.map((_, s) => Pipe.pool([0, 1, 2, 3].map((i) => hb(i, s)), m));
    const att = [0, 1, 2, 3].map((k) => Pipe.pool([0, 1, 2, 3, 4, 5, 6, 7].map((h) => ab(h, k)), m));
    const out = [0, 1, 2].map((k) => Pipe.pool([mb(k)], m));
    const pick = (idx) => idx.map((i) => hum[i]);
    P = {
      method: m, hum, att, out,
      humAll: Pipe.encode(hum), humAuto: Pipe.encode(pick(AUTO)), humNeuro: Pipe.encode(pick(NEURO)),
      aiAll: Pipe.encode(att.concat(out)), aiAtt: Pipe.encode(att), aiOut: Pipe.encode(out),
      sys: Pipe.encode(hum.concat(att, out)),
    };
  }
  function renderRaster() {
    if (!P) return;
    const ps = POOL_SENS[+el('pool-sensor').value];
    const start = +el('pool-start').value, W = 300;
    el('pool-start-v').textContent = `${Math.floor(start / 60)}:${String(start % 60).padStart(2, '0')}–${Math.floor((start + W) / 60)}:${String((start + W) % 60).padStart(2, '0')}`;
    let rows, labels, pooled;
    if (ps.kind === 'human') { rows = [0, 1, 2, 3].map((i) => hb(i, ps.idx)); labels = ROLES; pooled = P.hum[ps.idx]; }
    else if (ps.kind === 'head') { rows = [0, 1, 2, 3, 4, 5, 6, 7].map((h) => ab(h, ps.idx)); labels = HEADS; pooled = P.att[ps.idx]; }
    else { rows = [mb(ps.idx)]; labels = ['model']; pooled = P.out[ps.idx]; }
    const q = [css('--q0'), css('--q1'), css('--q2'), css('--q3')];
    const lv = maxLevel(P.method, rows.length);
    const host = el('raster');
    host.innerHTML = '';
    const draw = (vals, colorOf) => {
      const cv = document.createElement('canvas');
      cv.width = W; cv.height = 1;
      const c = cv.getContext('2d');
      for (let t = 0; t < W; t++) { c.fillStyle = colorOf(vals[start + t]); c.fillRect(t, 0, 1, 1); }
      return cv;
    };
    rows.forEach((r, i) => {
      const lab = document.createElement('span'); lab.className = 'rl'; lab.textContent = labels[i];
      host.appendChild(lab); host.appendChild(draw(r, (v) => q[v]));
    });
    const lab = document.createElement('span'); lab.className = 'rl'; lab.style.color = ps.kind === 'human' ? 'var(--human)' : 'var(--ai)'; lab.style.fontWeight = '600'; lab.textContent = 'pooled';
    host.appendChild(lab);
    const sysC = ps.kind === 'human' ? css('--human') : css('--ai');
    host.appendChild(draw(pooled, (v) => (v === 0 ? css('--q0') : hexA(sysC, 0.25 + 0.75 * (v / Math.max(1, lv))))));
  }
  function maxLevel(m, n) { return { majority_high_binary: 1, any_high_binary: 1, high_count_3level: 2, mean_bin_3level: 2, high_count_exact: n, median_bin_4level: 3, mode_bin_4level: 3 }[m]; }
  function renderCounters() {
    const c = (lab, a, b, color) => `<div class="counter"><span class="lab" style="color:${color}">${lab}</span><span class="big">${b.toLocaleString()}</span><span class="sub">pooled · ${a.toLocaleString()} if concatenated raw</span></div>`;
    el('counters').innerHTML = c('People', UNPOOLED.hu.unique, P.humAll.unique, 'var(--human)') + c('AI', UNPOOLED.ai.unique, P.aiAll.unique, 'var(--ai)') + c('Human–AI system', UNPOOLED.sy.unique, P.sys.unique, 'var(--system)');
  }

  // ---------------------------------------------------------------- 5. measure
  const PAIRS = {
    auto: { label: 'Autonomic ↔ AI output', x: 'humAuto', y: 'aiOut', key: 'Autonomic + Output' },
    neuro: { label: 'Neural ↔ AI attention', x: 'humNeuro', y: 'aiAtt', key: 'Neural + Attention' },
    all: { label: 'All human ↔ all AI', x: 'humAll', y: 'aiAll', key: 'All + All' },
  };
  el('pair').innerHTML = Object.entries(PAIRS).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('');
  let ENT = null;
  function computeEntropy() {
    const w = +el('ent-win').value;
    const f = (ids) => Pipe.finish(Pipe.windowedEntropy(ids, w));
    ENT = { w, hum: f(P.humAll.ids), ai: f(P.aiAll.ids), sys: f(P.sys.ids), raw: f(UNPOOLED.sy.ids) };
  }
  const entChart = makeChart(el('chart-ent'), 250, (ctx, w, h) => {
    if (!ENT) return;
    const skip = Math.max(0, ENT.w + 101 - 159);
    const s = (y, color, dash) => ({ y, x0: 159, dx: 1, color, width: 1.8, dash, skip });
    drawTime(ctx, w, h, { series: [s(ENT.raw, css('--muted'), [5, 4]), s(ENT.hum, css('--human')), s(ENT.ai, css('--ai')), s(ENT.sys, css('--system'))], ymin: 0 });
  });
  legend('legend-ent', [{ color: css('--human'), label: 'People (pooled)' }, { color: css('--ai'), label: 'AI (pooled over heads)' }, { color: css('--system'), label: 'Human–AI system (concatenated)' }, { color: css('--muted'), label: 'System, raw concatenation', dash: true }]);
  function phaseRanges() { return PH.map((p) => [p.start + (p.start > 0 ? 60 : 0), p.end]); }
  function renderIntegration() {
    const pr = PAIRS[el('pair').value];
    const x = P[pr.x].ids, y = P[pr.y].ids;
    const res = phaseRanges().map(([a, b], k) => Pipe.phaseIntegration(x, y, a, b, 80, 11 + k));
    const S = D.summary[`integration ${pr.key}`];
    const showDots = P.method === 'majority_high_binary' && S;
    const vals = res.map((r) => r.excess).concat(showDots ? S.per_seed.flat() : []);
    const lo = Math.min(-0.05, ...vals), hi = Math.max(0.1, ...vals) * 1.12;
    const Yp = (v) => ((v - lo) / (hi - lo)) * 100;
    el('integ-bars').innerHTML = res.map((r, k) => {
      const z = Yp(0), v = Yp(r.excess);
      const dots = showDots ? S.per_seed.map((row, j) => `<span class="bar-dot" style="left:${30 + j * 10}%;bottom:${Yp(row[k])}%" title="session ${j + 1}: ${fmt(row[k], 3)} bits"></span>`).join('') : '';
      return `<div class="bar-col"><span class="bar-val">${fmt(r.excess)} bits</span><div class="bar-track"><div class="bar-zero" style="bottom:${z}%">${k === 0 ? '<span class="zero-lab">chance</span>' : ''}</div><div class="bar-fill" style="bottom:${Math.min(z, v)}%;height:${Math.abs(v - z)}%;opacity:${r.p < 0.05 ? 1 : 0.35}"></div>${dots}</div><span class="bar-lab"><b style="color:var(--ink)">${PH[k].name}</b><br>p ${r.p <= 1 / 81 + 1e-9 ? '≤ .012' : '= ' + r.p.toFixed(3).replace(/^0/, '')}</span></div>`;
    }).join('');
    el('integ-note').innerHTML = `Bars: this session, shared information above the surrogate mean (80 within-phase circular shifts; faded = not above chance). ${showDots ? 'Dots: five simulated sessions (200 shifts each).' : 'Session dots are available for the majority-high rule.'} Pooling rule: ${Pipe.METHODS[P.method]}.`;
  }
  const tc = D.results.integration_timecourse;
  makeChart(el('chart-tc'), 220, (ctx, w, h) => {
    drawTime(ctx, w, h, {
      series: [{ y: tc.real, x0: tc.t[0], dx: tc.t[1] - tc.t[0], color: css('--system'), width: 2 }, { y: tc.sur_mean, x0: tc.t[0], dx: tc.t[1] - tc.t[0], color: css('--chance'), width: 1.2, dash: [4, 4] }],
      band: { lo: tc.sur_lo, hi: tc.sur_hi, x0: tc.t[0], dx: tc.t[1] - tc.t[0], color: css('--chance') }, ymin: 0,
    });
  });
  legend('legend-tc', [{ color: css('--system'), label: 'I, autonomic ↔ AI output (bits)' }, { color: css('--chance'), label: 'surrogate mean and 95% band', dash: true }]);

  // ---------------------------------------------------------------- 6. person
  const PERSON = [0, 1, 2, 3].map((i) => Pipe.encode(AUTO.map((s) => Uint8Array.from(hb(i, s), (v) => (v >= 2 ? 1 : 0)))).ids);
  function renderRoles() {
    const y = P.aiOut.ids;
    const res = PERSON.map((x, i) => phaseRanges().map(([a, b], k) => Pipe.phaseIntegration(x, y, a, b, 80, 101 + 10 * i + k)));
    const all = res.flat().map((r) => r.excess);
    const lo = Math.min(-0.06, ...all), hi = Math.max(0.12, ...all) * 1.1;
    const Yp = (v) => ((v - lo) / (hi - lo)) * 100;
    el('role-bars').innerHTML = res.map((rr, i) => `<div class="role-card"><b style="color:var(--r${i});font-family:var(--font-display)">${ROLES[i]}</b>
      <div class="mbars"><div class="bar-zero" style="bottom:${Yp(0)}%"></div>${rr.map((r) => { const z = Yp(0), v = Yp(r.excess); return `<div class="mcol"><div class="mbar" style="bottom:${Math.min(z, v)}%;height:${Math.abs(v - z)}%;opacity:${r.p < 0.05 ? 1 : 0.35}" title="${fmt(r.excess, 3)} bits, p ${r.p.toFixed(3)}"></div></div>`; }).join('')}</div>
      <div class="mval">${rr.map((r) => `<span>${fmt(r.excess)}</span>`).join('')}</div>
      <div class="mlab"><span>Sep.</span><span>Adapts</span><span>Mutual</span></div></div>`).join('');
    const S = D.summary.person_to_AI;
    el('role-note').innerHTML = `Bits above chance for this session (faded = not above chance). Across five simulated sessions the mean was ${fmt(S.mean[0])}, ${fmt(S.mean[1])} and ${fmt(S.mean[2])} bits, with ${S.n_sig_p05_of_20[0]}, ${S.n_sig_p05_of_20[1]} and ${S.n_sig_p05_of_20[2]} of 20 person-sessions above chance (p < .05).`;
  }
  makeChart(el('chart-ra'), 210, (ctx, w, h) => {
    drawTime(ctx, w, h, { series: D.role_att.map((y, i) => ({ y, x0: 2.5, dx: 5, color: roleColor(i), width: 1.4 })), ymin: 0 });
  });
  legend('legend-ra', ROLES.map((r, i) => ({ color: roleColor(i), label: r })));

  // ---------------------------------------------------------------- wiring
  function recomputeAll() {
    computePools(); renderRaster(); renderCounters(); computeEntropy(); entChart(); renderIntegration(); renderRoles();
  }
  el('pool-method').addEventListener('change', recomputeAll);
  el('pool-sensor').addEventListener('change', renderRaster);
  el('pool-start').addEventListener('input', renderRaster);
  el('ent-win').addEventListener('input', () => { el('ent-win-v').textContent = el('ent-win').value + ' s'; computeEntropy(); entChart(); });
  el('pair').addEventListener('change', renderIntegration);
  recomputeAll();
})();
