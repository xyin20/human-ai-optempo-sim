"""TeamTalk-GPT: a tiny decoder-only transformer trained from scratch in JAX.

2 layers x 4 heads, d_model 64, context 64 tokens, word-level vocab (~170).
About 0.1 M parameters: small enough to run in a phone browser.
"""
from __future__ import annotations

import base64
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from corpus import build_vocab, conversation_tokens

CFG = dict(ctx=64, d=64, L=2, H=4, dff=256)
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def init_params(key, V, ctx, d, L, H, dff):
    def nrm(k, shape, s=0.02):
        return jax.random.normal(k, shape) * s

    keys = iter(jax.random.split(key, 8 + 8 * L))
    p = {
        "wte": nrm(next(keys), (V, d)),
        "wpe": nrm(next(keys), (ctx, d)),
        "lnf_g": jnp.ones(d), "lnf_b": jnp.zeros(d),
        "head": nrm(next(keys), (d, V)),
        "layers": [],
    }
    for _ in range(L):
        p["layers"].append({
            "ln1_g": jnp.ones(d), "ln1_b": jnp.zeros(d),
            "wq": nrm(next(keys), (d, d)), "wk": nrm(next(keys), (d, d)), "wv": nrm(next(keys), (d, d)),
            "wo": nrm(next(keys), (d, d), 0.02 / math.sqrt(2 * L)),
            "ln2_g": jnp.ones(d), "ln2_b": jnp.zeros(d),
            "w1": nrm(next(keys), (d, dff)), "b1": jnp.zeros(dff),
            "w2": nrm(next(keys), (dff, d), 0.02 / math.sqrt(2 * L)), "b2": jnp.zeros(d),
        })
    return p


def layernorm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * g + b


def forward(p, idx, keymask, H):
    """idx [B,T] int, keymask [B,T] bool (True = real token).

    Returns logits [B,T,V], attention [L,B,H,T,T], head outputs [L,B,H,T,dh], resid [B,T,d].
    """
    B, T = idx.shape
    x = p["wte"][idx] + p["wpe"][:T][None]
    d = x.shape[-1]
    dh = d // H
    causal = jnp.tril(jnp.ones((T, T), bool))
    allowed = causal[None, None] & keymask[:, None, None, :]
    # every query must see at least itself
    allowed = allowed | jnp.eye(T, dtype=bool)[None, None]
    atts, houts = [], []
    for lp in p["layers"]:
        h = layernorm(x, lp["ln1_g"], lp["ln1_b"])
        q = (h @ lp["wq"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        k = (h @ lp["wk"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        v = (h @ lp["wv"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        s = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(dh)
        s = jnp.where(allowed, s, -1e9)
        a = jax.nn.softmax(s, axis=-1)
        o = a @ v                                   # [B,H,T,dh]
        atts.append(a)
        houts.append(o)
        x = x + o.transpose(0, 2, 1, 3).reshape(B, T, d) @ lp["wo"]
        h2 = layernorm(x, lp["ln2_g"], lp["ln2_b"])
        x = x + jax.nn.gelu(h2 @ lp["w1"] + lp["b1"], approximate=True) @ lp["w2"] + lp["b2"]
    xf = layernorm(x, p["lnf_g"], p["lnf_b"])
    logits = xf @ p["head"]
    return logits, jnp.stack(atts), jnp.stack(houts), x


def loss_fn(p, xb, yb, mb, H):
    logits, _, _, _ = forward(p, xb, mb, H)
    logp = jax.nn.log_softmax(logits, -1)
    nll = -jnp.take_along_axis(logp, yb[..., None], -1)[..., 0]
    w = mb.astype(jnp.float32)
    return (nll * w).sum() / w.sum()


def make_stream(vocab, n_conv, seed):
    rng = random.Random(seed)
    stoi = {w: i for i, w in enumerate(vocab)}
    ids = []
    for _ in range(n_conv):
        toks = conversation_tokens(rng, rng.randint(12, 40))
        ids.extend(stoi.get(t, 1) for t in toks)
    return np.array(ids, dtype=np.int32)


def train(steps=3500, batch=48, lr=3e-3, seed=0):
    vocab = build_vocab()
    V = len(vocab)
    ctx, d, L, H, dff = (CFG[k] for k in ("ctx", "d", "L", "H", "dff"))
    stream = make_stream(vocab, 60000, seed=11)
    val = make_stream(vocab, 2000, seed=99)
    print(f"vocab {V}, train tokens {len(stream):,}, val tokens {len(val):,}")
    p = init_params(jax.random.PRNGKey(seed), V, ctx, d, L, H, dff)
    nparams = sum(x.size for x in jax.tree_util.tree_leaves(p))
    print(f"params {nparams:,}")

    grad_fn = jax.jit(jax.value_and_grad(partial(loss_fn, H=H)))
    eval_fn = jax.jit(partial(loss_fn, H=H))
    m = jax.tree_util.tree_map(jnp.zeros_like, p)
    v = jax.tree_util.tree_map(jnp.zeros_like, p)
    b1, b2, eps, wd = 0.9, 0.99, 1e-8, 0.01

    @jax.jit
    def adam(p, g, m, v, t, lr_t):
        m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, m, g)
        v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, v, g)
        mh = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** t), m)
        vh = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** t), v)
        p = jax.tree_util.tree_map(lambda p, mh, vh: p - lr_t * (mh / (jnp.sqrt(vh) + eps) + wd * p * (p.ndim > 1)), p, mh, vh)
        return p, m, v

    rs = np.random.default_rng(seed)
    ones = jnp.ones((batch, ctx), bool)
    t0 = time.time()
    for step in range(1, steps + 1):
        starts = rs.integers(0, len(stream) - ctx - 1, batch)
        xb = np.stack([stream[s:s + ctx] for s in starts])
        yb = np.stack([stream[s + 1:s + ctx + 1] for s in starts])
        warm = min(1.0, step / 200)
        lr_t = lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        loss, g = grad_fn(p, jnp.asarray(xb), jnp.asarray(yb), ones)
        p, m, v = adam(p, g, m, v, step, lr_t)
        if step % 250 == 0 or step == 1:
            vs = rs.integers(0, len(val) - ctx - 1, 256)
            vx = np.stack([val[s:s + ctx] for s in vs]); vy = np.stack([val[s + 1:s + ctx + 1] for s in vs])
            vl = eval_fn(p, jnp.asarray(vx), jnp.asarray(vy), jnp.ones((256, ctx), bool))
            print(f"step {step:5d} loss {float(loss):.3f} val {float(vl):.3f} ({time.time()-t0:.0f}s)", flush=True)
    return p, vocab


def save(p, vocab, path=OUT / "teamtalk_gpt.npz"):
    flat = {"wte": p["wte"], "wpe": p["wpe"], "lnf_g": p["lnf_g"], "lnf_b": p["lnf_b"], "head": p["head"]}
    for i, lp in enumerate(p["layers"]):
        for k, val in lp.items():
            flat[f"l{i}_{k}"] = val
    np.savez(path, **{k: np.asarray(v) for k, v in flat.items()})
    (OUT / "vocab.json").write_text(json.dumps(vocab))


def load(path=OUT / "teamtalk_gpt.npz"):
    z = np.load(path)
    vocab = json.loads((OUT / "vocab.json").read_text())
    p = {k: jnp.asarray(z[k]) for k in ("wte", "wpe", "lnf_g", "lnf_b", "head")}
    p["layers"] = []
    for i in range(CFG["L"]):
        keys = [k for k in z.files if k.startswith(f"l{i}_")]
        p["layers"].append({k.split("_", 1)[1]: jnp.asarray(z[k]) for k in keys})
    return p, vocab


def export_js(p, vocab, path=OUT / "teamtalk_gpt_weights.json"):
    """Float16 base64 blobs for the in-browser forward pass."""
    def b64(a):
        return base64.b64encode(np.asarray(a, dtype=np.float16).tobytes()).decode()

    tensors = {"wte": p["wte"], "wpe": p["wpe"], "lnf_g": p["lnf_g"], "lnf_b": p["lnf_b"], "head": p["head"]}
    for i, lp in enumerate(p["layers"]):
        for k, val in lp.items():
            tensors[f"l{i}_{k}"] = val
    out = {"cfg": CFG, "vocab": vocab,
           "tensors": {k: {"shape": list(np.asarray(v).shape), "f16": b64(v)} for k, v in tensors.items()}}
    path.write_text(json.dumps(out))
    print("exported", path, path.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    p, vocab = train()
    save(p, vocab)
    export_js(p, vocab)
