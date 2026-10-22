"""Compute everything the deck and the interactive page show, from out/cosim.npz.

Writes out/results.json (numbers) and out/web_data.json (compact data for the page).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import optempo as op
from analysis import load, symbolize, layer_states
from cosim import AUTO, HEAD_SENSORS, MODEL_SENSORS, NEURO, PHASES, ROLES, SENSORS

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
METHOD = "majority_high_binary"
N_SUR = 200
SKIP = 60          # seconds skipped after each phase switch


def phase_idx():
    return [(a + (SKIP if a > 0 else 0), b, name) for a, b, name in PHASES]


def within_phase_surrogate(x, y, a, b, n=N_SUR, seed=0, min_shift=60):
    """MI after circularly shifting y inside [a, b). Keeps each block's own
    autocorrelation and alphabet; breaks only their simultaneity."""
    rng = np.random.default_rng(seed)
    xs, ys = x[a:b], y[a:b]
    L = len(xs)
    out = np.empty(n)
    for k in range(n):
        s = int(rng.integers(min_shift, L - min_shift))
        out[k] = op.mi(xs, np.roll(ys, s))
    return out


def binary_person_state(hb, i, sensors):
    """A person's own high/low per sensor: the same 'high' rule the pooling uses."""
    idx = [SENSORS.index(s) for s in sensors]
    return op.encode([(hb[i, j] >= 2).astype(np.int8) for j in idx])


def sliding_integration(x, y, w=600, step=10, n=40, seed=1):
    """Long-window MI and a per-window circular-shift surrogate (time course)."""
    rng = np.random.default_rng(seed)
    T = len(x)
    ts, real, sur_m, sur_lo, sur_hi = [], [], [], [], []
    for end in range(w, T + 1, step):
        a = end - w
        xs, ys = x[a:end], y[a:end]
        r = op.mi(xs, ys)
        ss = np.array([op.mi(xs, np.roll(ys, int(rng.integers(60, w - 60)))) for _ in range(n)])
        ts.append(end); real.append(r); sur_m.append(ss.mean())
        sur_lo.append(np.percentile(ss, 2.5)); sur_hi.append(np.percentile(ss, 97.5))
    return {"t": ts, "real": real, "sur_mean": sur_m, "sur_lo": sur_lo, "sur_hi": sur_hi}


def main():
    d = load()
    hb, ab, mb = symbolize(d)
    T = hb.shape[2]
    states, cols = layer_states(hb, ab, mb, METHOD)
    st3, _ = layer_states(hb, ab, mb, "high_count_3level")
    R: dict = {"method": METHOD, "T": T, "phases": [{"start": a, "end": b, "name": n} for a, b, n in PHASES]}

    # ---- A. state-space size: concatenating everything vs pooling first
    unpooled_h = op.encode([hb[i, j] for i in range(4) for j in range(len(SENSORS))])
    unpooled_ai = op.encode([ab[h, j] for h in range(8) for j in range(len(HEAD_SENSORS))] + [mb[j] for j in range(len(MODEL_SENSORS))])
    unpooled_sys = op.encode([unpooled_h, unpooled_ai])
    R["unique_states"] = {
        "unpooled": {"Human": int(len(np.unique(unpooled_h))), "AI": int(len(np.unique(unpooled_ai))), "System": int(len(np.unique(unpooled_sys)))},
        "majority": {"Human": int(len(np.unique(states["Human All"]))), "AI": int(len(np.unique(states["AI All"]))), "System": int(len(np.unique(states["System"])))},
        "three_level": {"Human": int(len(np.unique(st3["Human All"]))), "AI": int(len(np.unique(st3["AI All"]))), "System": int(len(np.unique(st3["System"])))},
    }
    # windowed entropy of the unpooled vs pooled system state (saturation demo)
    ent_unpooled_sys = op.finish(op.windowed_entropy(unpooled_sys))
    ent = {k: op.finish(op.windowed_entropy(states[k])) for k in ("Human All", "AI All", "System")}
    R["entropy_30s_phase_means"] = {}
    off = op.REMOVE_INITIAL
    for a, b, name in phase_idx():
        a2, b2 = max(a - off, 0) + (100 if a > 0 else 0), b - off
        R["entropy_30s_phase_means"][name] = {k: float(v[a2:b2].mean()) for k, v in ent.items()}
        R["entropy_30s_phase_means"][name]["System (unpooled)"] = float(ent_unpooled_sys[a2:b2].mean())

    # ---- B. whole-phase integration I = H(Human) + H(AI) - H(Human+AI)
    pairs = {"Autonomic + Output": ("Human Auto", "AI Output"),
             "Neural + Attention": ("Human Neuro", "AI Attention"),
             "All + All": ("Human All", "AI All")}
    integ = {}
    for pname, (hk, ak) in pairs.items():
        rows = []
        for k, (a, b, name) in enumerate(phase_idx()):
            x, y = states[hk][a:b], states[ak][a:b]
            Hh, Ha = op.entropy(x), op.entropy(y)
            Hs = op.entropy(op.encode([x, y]))
            I = Hh + Ha - Hs
            sur = within_phase_surrogate(states[hk], states[ak], a, b, seed=100 + k)
            rows.append({"phase": name, "H_human": Hh, "H_ai": Ha, "H_system": Hs, "I": I,
                         "sur_mean": float(sur.mean()), "sur_lo": float(np.percentile(sur, 2.5)),
                         "sur_hi": float(np.percentile(sur, 97.5)), "excess": float(I - sur.mean()),
                         "p": float((np.sum(sur >= I) + 1) / (len(sur) + 1))})
        integ[pname] = rows
    R["integration_whole_phase"] = integ

    # ---- C. person-level: each person's own state vs the AI, vs the rest of the team
    person = {}
    for i, r in enumerate(ROLES):
        others = [j for j in range(4) if j != i]
        st_loo, _ = layer_states(hb, ab, mb, METHOD, people=others)
        xa = binary_person_state(hb, i, AUTO)
        rows = []
        for k, (a, b, name) in enumerate(phase_idx()):
            row = {"phase": name}
            for tname, target in (("AI", st_loo["AI Output"]), ("team (others)", st_loo["Human Auto"]),
                                  ("system (others + AI)", op.encode([st_loo["Human Auto"], st_loo["AI Output"]]))):
                real = op.mi(xa[a:b], target[a:b])
                sur = within_phase_surrogate(xa, target, a, b, n=120, seed=300 + 10 * i + k)
                row[tname] = {"I": real, "sur_mean": float(sur.mean()), "sur_hi": float(np.percentile(sur, 97.5)),
                              "excess": float(real - sur.mean()), "p": float((np.sum(sur >= real) + 1) / (len(sur) + 1))}
            rows.append(row)
        person[r] = rows
    R["person_level"] = person

    # ---- D. time course: sliding 10-min integration (Autonomic + Output) and 30-s OP-TEMPO entropy
    tc = sliding_integration(states["Human Auto"], states["AI Output"])
    R["integration_timecourse"] = tc

    # ---- E. ground truth from the generator
    W, S = d["W"], d["S"]
    gt = []
    for a, b, name in PHASES:
        gt.append({"phase": name,
                   "mean workload": float(W[a:b].mean()),
                   "pairwise workload corr": float(np.mean([np.corrcoef(W[a:b, i], W[a:b, j])[0, 1] for i in range(4) for j in range(i + 1, 4)])),
                   "AI error rate": float(d["err"][a:b].sum() / max(d["err"][a:b].sum() + d["ok"][a:b].sum(), 1)),
                   "AI replies": int(d["err"][a:b].sum() + d["ok"][a:b].sum()),
                   "team messages": int(d["n_msgs"][a:b].sum()),
                   "brevity style": float(S[a:b].mean())})
    R["ground_truth"] = gt

    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=float))
    np.savez_compressed(OUT / "series.npz", ent_human=ent["Human All"], ent_ai=ent["AI All"], ent_sys=ent["System"],
                        ent_sys_unpooled=ent_unpooled_sys)

    # ---- print a summary
    print("unique states:", R["unique_states"])
    print("30-s entropy phase means:", json.dumps(R["entropy_30s_phase_means"], indent=0))
    for pname, rows in integ.items():
        print(pname)
        for row in rows:
            print(f"   {row['phase']:10s} Hh={row['H_human']:.2f} Ha={row['H_ai']:.2f} Hs={row['H_system']:.2f} I={row['I']:.3f} sur={row['sur_mean']:.3f} [{row['sur_lo']:.3f},{row['sur_hi']:.3f}] excess={row['excess']:.3f} p={row['p']:.3f}")
    for r, rows in person.items():
        print(r, [(row["phase"], round(row["AI"]["excess"], 3), round(row["AI"]["p"], 3), round(row["team (others)"]["excess"], 3), round(row["system (others + AI)"]["excess"], 3)) for row in rows])
    print("ground truth:", gt)


if __name__ == "__main__":
    main()
