// TeamTalk-GPT forward pass in plain JS (matches sim/tinylm.py).
const TinyLM = (() => {
  function f16ToF32(b64) {
    const bin = atob(b64);
    const n = bin.length >> 1;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const h = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
      const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      out[i] = e === 0 ? s * Math.pow(2, -14) * (f / 1024)
             : e === 31 ? (f ? NaN : s * Infinity)
             : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return out;
  }

  function load(json) {
    const W = {};
    for (const [k, v] of Object.entries(json.tensors)) W[k] = { shape: v.shape, a: f16ToF32(v.f16) };
    const cfg = json.cfg, vocab = json.vocab;
    const stoi = new Map(vocab.map((w, i) => [w, i]));
    return { W, cfg, vocab, stoi };
  }

  function layernorm(x, T, d, g, b) {
    const out = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      let m = 0, v = 0;
      for (let i = 0; i < d; i++) m += x[t * d + i];
      m /= d;
      for (let i = 0; i < d; i++) { const z = x[t * d + i] - m; v += z * z; }
      v /= d;
      const inv = 1 / Math.sqrt(v + 1e-5);
      for (let i = 0; i < d; i++) out[t * d + i] = (x[t * d + i] - m) * inv * g[i] + b[i];
    }
    return out;
  }

  function matmul(x, T, n, w, m, bias) {          // x [T,n] @ w [n,m]
    const out = new Float32Array(T * m);
    for (let t = 0; t < T; t++) {
      const o = t * m;
      if (bias) for (let j = 0; j < m; j++) out[o + j] = bias[j];
      for (let i = 0; i < n; i++) {
        const xv = x[t * n + i];
        if (xv === 0) continue;
        const r = i * m;
        for (let j = 0; j < m; j++) out[o + j] += xv * w[r + j];
      }
    }
    return out;
  }

  const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));

  // ids: token ids (<= ctx). Returns logits [T,V], att [L][H] each Float32Array [T*T], head outputs.
  function forward(M, ids) {
    const { W, cfg } = M;
    const d = cfg.d, H = cfg.H, L = cfg.L, dh = d / H, dff = cfg.dff;
    const V = W.wte.shape[0];
    const T = ids.length;
    let x = new Float32Array(T * d);
    for (let t = 0; t < T; t++) for (let i = 0; i < d; i++) x[t * d + i] = W.wte.a[ids[t] * d + i] + W.wpe.a[t * d + i];
    const att = [], hnorm = [];
    for (let l = 0; l < L; l++) {
      const p = (k) => W[`l${l}_${k}`].a;
      const h = layernorm(x, T, d, p('ln1_g'), p('ln1_b'));
      const q = matmul(h, T, d, p('wq'), d), k = matmul(h, T, d, p('wk'), d), v = matmul(h, T, d, p('wv'), d);
      const o = new Float32Array(T * d);
      const attL = [], normL = [];
      for (let hh = 0; hh < H; hh++) {
        const A = new Float32Array(T * T);
        const nrm = new Float32Array(T);
        for (let t = 0; t < T; t++) {
          let mx = -Infinity;
          const row = new Float32Array(t + 1);
          for (let s = 0; s <= t; s++) {
            let dot = 0;
            for (let c = 0; c < dh; c++) dot += q[t * d + hh * dh + c] * k[s * d + hh * dh + c];
            row[s] = dot / Math.sqrt(dh);
            if (row[s] > mx) mx = row[s];
          }
          let sum = 0;
          for (let s = 0; s <= t; s++) { row[s] = Math.exp(row[s] - mx); sum += row[s]; }
          let n2 = 0;
          for (let c = 0; c < dh; c++) {
            let acc = 0;
            for (let s = 0; s <= t; s++) acc += (row[s] / sum) * v[s * d + hh * dh + c];
            o[t * d + hh * dh + c] = acc;
            n2 += acc * acc;
          }
          nrm[t] = Math.sqrt(n2);
          for (let s = 0; s <= t; s++) A[t * T + s] = row[s] / sum;
        }
        attL.push(A); normL.push(nrm);
      }
      att.push(attL); hnorm.push(normL);
      const proj = matmul(o, T, d, p('wo'), d);
      for (let i = 0; i < T * d; i++) x[i] += proj[i];
      const h2 = layernorm(x, T, d, p('ln2_g'), p('ln2_b'));
      const a1 = matmul(h2, T, d, p('w1'), dff, p('b1'));
      for (let i = 0; i < a1.length; i++) a1[i] = gelu(a1[i]);
      const a2 = matmul(a1, T, dff, p('w2'), d, p('b2'));
      for (let i = 0; i < T * d; i++) x[i] += a2[i];
    }
    const xf = layernorm(x, T, d, W.lnf_g.a, W.lnf_b.a);
    const logits = matmul(xf, T, d, W.head.a, V);
    return { logits, att, hnorm, T, V };
  }

  function tokenize(M, text) {
    const toks = text.toLowerCase().replace(/([,.?])/g, ' $1 ').split(/\s+/).filter(Boolean);
    return toks.map((w) => (M.stoi.has(w) ? w : '<unk>'));
  }

  return { load, forward, tokenize };
})();
if (typeof module !== 'undefined') module.exports = TinyLM;
