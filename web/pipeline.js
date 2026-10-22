// Pooled signal analysis in JS (matches the simulation implementation).
const Pipe = (() => {
  function unpack2(b64, n) {
    const bin = atob(b64);
    const out = new Uint8Array(n);
    for (let i = 0; i < n; i++) out[i] = (bin.charCodeAt(i >> 2) >> ((i & 3) * 2)) & 3;
    return out;
  }
  function unpackU16(b64) {
    const bin = atob(b64);
    const out = new Uint16Array(bin.length >> 1);
    for (let i = 0; i < out.length; i++) out[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    return out;
  }

  const METHODS = {
    majority_high_binary: 'Majority-high binary',
    high_count_3level: 'High-count, 3-level',
    high_count_exact: 'Exact high count',
    any_high_binary: 'Any-high binary',
    mean_bin_3level: 'Mean-bin, 3-level',
    median_bin_4level: 'Median-bin, 4-level',
    mode_bin_4level: 'Mode-bin, 4-level',
  };

  function median(vals) {
    const s = vals.slice().sort((a, b) => a - b);
    const n = s.length;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  }

  // rows: array of Uint8Array (members), same length. Returns Uint8Array.
  function pool(rows, method) {
    const n = rows.length, T = rows[0].length;
    const out = new Uint8Array(T);
    const vals = new Array(n);
    for (let t = 0; t < T; t++) {
      let high = 0, sum = 0;
      for (let m = 0; m < n; m++) { const v = rows[m][t]; vals[m] = v; if (v >= 2) high++; sum += v; }
      let r;
      switch (method) {
        case 'high_count_3level': { const q = high / n; r = q <= 1 / 3 ? 0 : q < 2 / 3 ? 1 : 2; break; }
        case 'high_count_exact': r = high; break;
        case 'any_high_binary': r = high > 0 ? 1 : 0; break;
        case 'majority_high_binary': r = high > n / 2 ? 1 : 0; break;
        case 'mean_bin_3level': { const m = sum / n; r = m < 1 ? 0 : m < 2 ? 1 : 2; break; }
        case 'median_bin_4level': r = Math.max(0, Math.min(3, Math.floor(median(vals) + 0.5))); break;
        case 'mode_bin_4level': {
          const c = [0, 0, 0, 0]; for (const v of vals) c[v]++;
          const mx = Math.max(...c); const med = median(vals);
          let best = -1, bd = Infinity;
          for (let v = 0; v < 4; v++) if (c[v] === mx) { const dd = Math.abs(v - med); if (dd < bd) { bd = dd; best = v; } }
          r = best; break;
        }
        default: throw new Error(method);
      }
      out[t] = r;
    }
    return out;
  }

  // Concatenate columns into a state label -> dense ids (Int32Array) + number of unique states.
  function encode(cols) {
    const T = cols[0].length;
    const ids = new Int32Array(T);
    const map = new Map();
    for (let t = 0; t < T; t++) {
      let key = '';
      for (let c = 0; c < cols.length; c++) key += String.fromCharCode(48 + cols[c][t]);
      let id = map.get(key);
      if (id === undefined) { id = map.size; map.set(key, id); }
      ids[t] = id;
    }
    return { ids, unique: map.size };
  }

  const xlogx = (c) => (c > 0 ? c * Math.log2(c) : 0);

  function entropy(seq, a = 0, b = seq.length) {
    const m = new Map();
    for (let i = a; i < b; i++) m.set(seq[i], (m.get(seq[i]) || 0) + 1);
    const n = b - a; let s = 0;
    for (const c of m.values()) s += xlogx(c);
    return Math.log2(n) - s / n;
  }

  // Forward windowed entropy: value at t uses seq[t-w..t-1]; first w values 0.
  function windowedEntropy(seq, w) {
    const T = seq.length, out = new Float64Array(T);
    const m = new Map(); let S = 0;
    const add = (v, d) => { const c = m.get(v) || 0; S -= xlogx(c); const c2 = c + d; S += xlogx(c2); if (c2) m.set(v, c2); else m.delete(v); };
    for (let i = 0; i < w && i < T; i++) add(seq[i], 1);
    for (let t = w; t < T; t++) {
      out[t] = Math.log2(w) - S / w;
      add(seq[t - w], -1); add(seq[t], 1);
    }
    return out;
  }

  function jointIds(x, y) {
    const T = x.length, out = new Int32Array(T);
    let my = 0; for (let i = 0; i < T; i++) if (y[i] > my) my = y[i];
    for (let i = 0; i < T; i++) out[i] = x[i] * (my + 1) + y[i];
    return out;
  }

  function mi(x, y, a = 0, b = x.length) {
    return entropy(x, a, b) + entropy(y, a, b) - entropy(jointIds(x, y), a, b);
  }

  // scipy uniform_filter1d(size, mode='nearest'), then drop the first `drop` samples.
  function finish(series, size = 200, drop = 159) {
    const T = series.length, left = Math.floor(size / 2), right = size - 1 - left;
    const out = new Float64Array(T);
    const at = (i) => series[Math.min(T - 1, Math.max(0, i))];
    let s = 0;
    for (let i = -left; i <= right; i++) s += at(i);
    for (let t = 0; t < T; t++) {
      out[t] = s / size;
      s += at(t + right + 1) - at(t - left);
    }
    return out.slice(drop);
  }

  function rng(seed) { // mulberry32
    let a = seed >>> 0;
    return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }

  // MI in [a,b) and a circular-shift surrogate of y inside [a,b).
  function phaseIntegration(x, y, a, b, nSur = 60, seed = 1) {
    const xs = x.slice(a, b), ys = y.slice(a, b), L = b - a;
    const real = mi(xs, ys);
    const r = rng(seed), sur = [];
    const yy = new Int32Array(L);
    for (let k = 0; k < nSur; k++) {
      const sh = 60 + Math.floor(r() * (L - 120));
      for (let i = 0; i < L; i++) yy[i] = ys[(i + sh) % L];
      sur.push(mi(xs, yy));
    }
    sur.sort((p, q) => p - q);
    const mean = sur.reduce((s, v) => s + v, 0) / sur.length;
    return { real, surMean: mean, surHi: sur[Math.floor(0.975 * (sur.length - 1))], excess: real - mean, p: (sur.filter((v) => v >= real).length + 1) / (sur.length + 1) };
  }

  return { unpack2, unpackU16, METHODS, pool, encode, entropy, windowedEntropy, mi, finish, phaseIntegration, rng };
})();
if (typeof module !== 'undefined') module.exports = Pipe;
