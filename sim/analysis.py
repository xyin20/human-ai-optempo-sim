"""Run the OP-TEMPO pooled pipeline on the human-AI co-simulation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import optempo as op
from cosim import AUTO, HEAD_SENSORS, MODEL_SENSORS, NEURO, PHASES, ROLES, SENSORS

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def load(path=OUT / "cosim.npz"):
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    return d


def symbolize(d):
    X = d["X"]                                   # [T, 4, 9]
    T = X.shape[0]
    hb = np.zeros((4, len(SENSORS), T), np.int8)
    for i in range(4):
        for s in range(len(SENSORS)):
            hb[i, s] = op.quartile_bins(X[:, i, s])
    ab = np.zeros((8, len(HEAD_SENSORS), T), np.int8)
    for k, name in enumerate(HEAD_SENSORS):
        H = d[f"head_{name}"]                    # [T, 8]
        for h in range(8):
            ab[h, k] = op.quartile_bins(H[:, h])
    mb = np.zeros((len(MODEL_SENSORS), T), np.int8)
    for k, name in enumerate(MODEL_SENSORS):
        mb[k] = op.quartile_bins(d[f"model_{name}"])
    return hb, ab, mb


def pooled_columns(hb, ab, mb, method, people=(0, 1, 2, 3)):
    """Pooled per-sensor columns for each block."""
    people = list(people)
    hum = {s: op.pool(hb[people, j], method) for j, s in enumerate(SENSORS)}
    att = {s: op.pool(ab[:, j], method) for j, s in enumerate(HEAD_SENSORS)}
    out = {s: op.pool(mb[j][None], method) for j, s in enumerate(MODEL_SENSORS)}
    return hum, att, out


def layer_states(hb, ab, mb, method, people=(0, 1, 2, 3)):
    hum, att, out = pooled_columns(hb, ab, mb, method, people)
    L = {}
    L["Human Neuro"] = [hum[s] for s in NEURO]
    L["Human Auto"] = [hum[s] for s in AUTO]
    L["Human All"] = [hum[s] for s in SENSORS]
    L["AI Attention"] = [att[s] for s in HEAD_SENSORS]
    L["AI Output"] = [out[s] for s in MODEL_SENSORS]
    L["AI All"] = L["AI Attention"] + L["AI Output"]
    L["System"] = L["Human All"] + L["AI All"]
    L["System Auto+Output"] = L["Human Auto"] + L["AI Output"]
    L["System Neuro+Attention"] = L["Human Neuro"] + L["AI Attention"]
    return {k: op.encode(v) for k, v in L.items()}, L


def individual_state(hb, i, sensors):
    idx = [SENSORS.index(s) for s in sensors]
    return op.encode([hb[i, j] for j in idx])


def phase_slices(n_after_trim):
    """Phase index ranges in the trimmed (remove_initial) time base."""
    out = []
    for a, b, name in PHASES:
        a2 = max(a - op.REMOVE_INITIAL, 0)
        b2 = b - op.REMOVE_INITIAL
        # skip the first 100 s after a phase switch (smoothing half-width)
        out.append((name, a2 + (100 if a > 0 else 0), b2))
    return out


def run(method="majority_high_binary", n_surr=19, verbose=True):
    d = load()
    hb, ab, mb = symbolize(d)
    T = hb.shape[2]
    states, cols = layer_states(hb, ab, mb, method)
    res = {"method": method, "T": T}

    # 1) state-space size (whole session)
    unpooled_h = op.encode([hb[i, j] for i in range(4) for j in range(len(SENSORS))])
    unpooled_ai = op.encode([ab[h, j] for h in range(8) for j in range(len(HEAD_SENSORS))] + [mb[j] for j in range(len(MODEL_SENSORS))])
    unpooled_sys = op.encode([unpooled_h, unpooled_ai])
    res["unique"] = {
        "Human (unpooled)": int(len(np.unique(unpooled_h))),
        "AI (unpooled)": int(len(np.unique(unpooled_ai))),
        "System (unpooled)": int(len(np.unique(unpooled_sys))),
        **{k: int(len(np.unique(v))) for k, v in states.items()},
    }

    # 2) windowed team entropy
    ent = {k: op.finish(op.windowed_entropy(states[k])) for k in ("Human All", "AI All", "System", "Human Auto", "AI Output", "System Auto+Output")}
    res["entropy"] = ent

    # 3) human-AI integration = windowed MI(Human pooled, AI pooled) + surrogate
    integ = {}
    for name, (a, b) in {"All": ("Human All", "AI All"), "Auto+Output": ("Human Auto", "AI Output"),
                         "Neuro+Attention": ("Human Neuro", "AI Attention")}.items():
        real = op.windowed_mi(states[a], states[b])
        sur, shifts = op.circular_surrogate_mi(states[a], states[b], n_surr=n_surr, seed=3)
        real_f = op.finish(real)
        sur_f = np.stack([op.finish(s) for s in sur])
        integ[name] = {"real": real_f, "sur_mean": sur_f.mean(0), "sur_lo": np.percentile(sur_f, 5, 0),
                       "sur_hi": np.percentile(sur_f, 95, 0), "excess": real_f - sur_f.mean(0)}
    res["integration"] = integ

    # 4) leave-one-out role-to-system AMI, and person-to-AI AMI
    role = {}
    for i, r in enumerate(ROLES):
        others = [j for j in range(4) if j != i]
        st_loo, _ = layer_states(hb, ab, mb, method, people=others)
        x = individual_state(hb, i, SENSORS)
        xa = individual_state(hb, i, AUTO)
        entry = {}
        for tname, target in {"to system (LOO)": st_loo["System"], "to team (LOO)": st_loo["Human All"],
                              "to AI": st_loo["AI All"]}.items():
            src = xa if tname == "to AI" else x
            real = op.finish(op.windowed_mi(src, target))
            sur, _ = op.circular_surrogate_mi(src, target, n_surr=n_surr, seed=10 + i)
            sur_f = np.stack([op.finish(s) for s in sur])
            entry[tname] = {"real": real, "sur_mean": sur_f.mean(0), "excess": real - sur_f.mean(0)}
        role[r] = entry
    res["role"] = role

    # 5) lagged MI per phase (whole phase, not windowed): direction of coupling
    lags = np.arange(-40, 41, 2)
    lag = {}
    for a, b, name in PHASES:
        x = states["Human Auto"][a + 60:b]
        y = states["AI Output"][a + 60:b]
        real = op.lagged_mi(x, y, lags)
        rng = np.random.default_rng(5)
        surs = []
        for _ in range(n_surr):
            k = int(rng.integers(100, len(y) - 100))
            surs.append(op.lagged_mi(x, np.roll(y, k), lags))
        surs = np.array(surs)
        lag[name] = {"real": real, "sur_mean": surs.mean(0), "sur_hi": np.percentile(surs, 95, 0), "excess": real - surs.mean(0)}
    res["lag"] = {"lags": lags, **lag}

    # 6) phase summaries
    n = len(ent["System"])
    ph = phase_slices(n)
    summ = []
    for name, a, b in ph:
        row = {"phase": name}
        for k in ("Human All", "AI All", "System"):
            row[f"H {k}"] = float(ent[k][a:b].mean())
        for k, v in integ.items():
            row[f"I {k} real"] = float(v["real"][a:b].mean())
            row[f"I {k} sur"] = float(v["sur_mean"][a:b].mean())
            row[f"I {k} excess"] = float(v["excess"][a:b].mean())
        for r in ROLES:
            for tname in ("to system (LOO)", "to team (LOO)", "to AI"):
                row[f"{r} {tname} excess"] = float(role[r][tname]["excess"][a:b].mean())
        summ.append(row)
    res["phase_summary"] = summ
    res["phase_slices"] = ph

    # 7) ground-truth checks on the latent variables
    W, S = d["W"], d["S"]
    gt = []
    for a, b, name in PHASES:
        wm = W[a:b].mean(1)
        gt.append({"phase": name,
                   "corr(workload, AI uncertainty)": float(np.corrcoef(wm, d["model_uncertainty"][a:b])[0, 1]),
                   "corr(workload, AI surprisal)": float(np.corrcoef(wm, d["model_surprisal"][a:b])[0, 1]),
                   "mean pairwise corr workload": float(np.mean([np.corrcoef(W[a:b, i], W[a:b, j])[0, 1] for i in range(4) for j in range(i + 1, 4)])),
                   "AI error rate": float(d["err"][a:b].sum() / max(d["err"][a:b].sum() + d["ok"][a:b].sum(), 1)),
                   "brevity style": float(S[a:b].mean()),
                   "workload": float(wm.mean())})
    res["ground_truth"] = gt

    if verbose:
        print("method:", method)
        print("unique states:", json.dumps(res["unique"], indent=1))
        for row in gt:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
        for row in summ:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
        for name in [p[2] for p in PHASES]:
            ex = res["lag"][name]["excess"]
            print(name, "lagged excess MI (Human Auto -> AI Output), lags -40..40 step 2:", np.round(ex, 3).tolist())
    return res, (hb, ab, mb, states, cols, d)


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "majority_high_binary"
    run(m)
