"""Aggregate the five simulated sessions (seeds 7-11) into out/summary_5sessions.json."""
import json
import numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out"
seeds = [7, 8, 9, 10, 11]; phases = ["Separate", "AI adapts", "Mutual"]; roles = ["P1", "P2", "P3", "P4"]
R = [json.load(open(OUT / f"results_seed{s}.json")) for s in seeds]
S = {"seeds": seeds, "phases": phases, "roles": roles}
for pname in ["Autonomic + Output", "Neural + Attention", "All + All"]:
    a = np.array([[r["excess"] for r in x["integration_whole_phase"][pname]] for x in R])
    p = np.array([[r["p"] for r in x["integration_whole_phase"][pname]] for x in R])
    S[f"integration {pname}"] = {"per_seed": a.round(4).tolist(), "mean": a.mean(0).round(4).tolist(), "min": a.min(0).round(4).tolist(), "max": a.max(0).round(4).tolist(), "n_sig_p05": (p < 0.05).sum(0).tolist()}
pa = np.array([[[x["person_level"][r][k]["AI"]["excess"] for k in range(3)] for r in roles] for x in R])
pp = np.array([[[x["person_level"][r][k]["AI"]["p"] for k in range(3)] for r in roles] for x in R])
S["person_to_AI"] = {"per_seed_role": pa.round(4).tolist(), "mean_by_role": pa.mean(0).round(4).tolist(), "mean": pa.mean((0, 1)).round(4).tolist(), "n_sig_p05_of_20": (pp < 0.05).sum((0, 1)).tolist()}
gain = np.array([[[x["person_level"][r][k]["system (others + AI)"]["excess"] - x["person_level"][r][k]["team (others)"]["excess"] for k in range(3)] for r in roles] for x in R]).mean(1)
S["system_minus_team_gain"] = {"per_seed": gain.round(4).tolist(), "mean": gain.mean(0).round(4).tolist(), "n_positive": (gain > 0).sum(0).tolist()}
ent = {k: np.array([[x["entropy_30s_phase_means"][ph][k] for ph in phases] for x in R]) for k in ["Human All", "AI All", "System", "System (unpooled)"]}
S["entropy_30s"] = {"Human All": {"per_seed": ent["Human All"].round(3).tolist(), "mean": ent["Human All"].mean(0).round(3).tolist()},
                    "AI All": {"mean": ent["AI All"].mean(0).round(3).tolist()}, "System": {"mean": ent["System"].mean(0).round(3).tolist()},
                    "System unpooled": {"mean": ent["System (unpooled)"].mean(0).round(3).tolist()},
                    "mutual_lower_than_adapts": int((ent["Human All"][:, 2] < ent["Human All"][:, 1]).sum()),
                    "mutual_minus_adapts_mean": float((ent["Human All"][:, 2] - ent["Human All"][:, 1]).mean().round(3))}
u = np.array([[x["unique_states"]["unpooled"]["Human"], x["unique_states"]["unpooled"]["System"], x["unique_states"]["majority"]["Human"], x["unique_states"]["majority"]["AI"], x["unique_states"]["majority"]["System"], x["unique_states"]["three_level"]["System"]] for x in R])
S["unique_states_mean"] = dict(zip(["unpooled Human", "unpooled System", "majority Human", "majority AI", "majority System", "three-level System"], u.mean(0).round(0).tolist()))
keys = ["pairwise workload corr", "mean workload", "AI error rate", "AI replies", "team messages", "brevity style"]
gt = np.array([[[g[k] for k in keys] for g in x["ground_truth"]] for x in R])
S["ground_truth_mean"] = {ph: dict(zip(keys, gt.mean(0)[i].round(3).tolist())) for i, ph in enumerate(phases)}
json.dump(S, open(OUT / "summary_5sessions.json", "w"), indent=1)
print("integration A+O mean", S["integration Autonomic + Output"]["mean"], "person mean", S["person_to_AI"]["mean"])
