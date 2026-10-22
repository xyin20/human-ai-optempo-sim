"""Pack the showcase session + 5-session summary + model weights for the web page."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from analysis import load, symbolize
from cosim import HEAD_SENSORS, MODEL_SENSORS, PHASES, ROLES, SENSORS

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def pack2(a: np.ndarray) -> str:
    """Pack values 0..3 four per byte, base64."""
    a = np.asarray(a, dtype=np.uint8).reshape(-1)
    pad = (-len(a)) % 4
    a = np.concatenate([a, np.zeros(pad, np.uint8)])
    b = (a[0::4] | (a[1::4] << 2) | (a[2::4] << 4) | (a[3::4] << 6)).astype(np.uint8)
    return base64.b64encode(b.tobytes()).decode()


def ds(x, k=5):
    x = np.asarray(x, float)
    n = len(x) // k
    return x[: n * k].reshape(n, k).mean(1)


def q16(x, lo, hi):
    """Quantize to 0..4095, little-endian uint16, base64 (decoded in JS)."""
    x = np.clip((np.asarray(x) - lo) / (hi - lo), 0, 1)
    return base64.b64encode(np.round(x * 4095).astype("<u2").tobytes()).decode()


def main():
    d = load()
    hb, ab, mb = symbolize(d)
    T = hb.shape[2]
    X = d["X"]
    web = {"T": T, "phases": [{"start": a, "end": b, "name": n} for a, b, n in PHASES],
           "roles": ROLES, "sensors": SENSORS, "head_sensors": HEAD_SENSORS, "model_sensors": MODEL_SENSORS}
    # quartile bins, packed: humans [4][9][T], heads [8][4][T], model [3][T]
    web["bins_h"] = pack2(hb)
    web["bins_head"] = pack2(ab)
    web["bins_model"] = pack2(mb)
    # raw series for display, 5-s means, quantized
    raw = {}
    for j, s in enumerate(SENSORS):
        v = np.stack([ds(X[:, i, j]) for i in range(4)])
        lo, hi = float(np.percentile(v, 0.5)), float(np.percentile(v, 99.5))
        raw[s] = {"lo": lo, "hi": hi, "q": [q16(v[i], lo, hi) for i in range(4)]}
    for s in HEAD_SENSORS:
        H = d[f"head_{s}"]
        v = np.stack([ds(H[:, h]) for h in range(8)])
        lo, hi = float(np.percentile(v, 0.5)), float(np.percentile(v, 99.5))
        raw["head_" + s] = {"lo": lo, "hi": hi, "q": [q16(v[h], lo, hi) for h in range(8)]}
    for s in MODEL_SENSORS:
        v = ds(d[f"model_{s}"])
        lo, hi = float(np.percentile(v, 0.5)), float(np.percentile(v, 99.5))
        raw["model_" + s] = {"lo": lo, "hi": hi, "q": [q16(v, lo, hi)]}
    wv = np.stack([ds(d["W"][:, i]) for i in range(4)])
    raw["workload"] = {"lo": float(wv.min()), "hi": float(wv.max()), "q": [q16(wv[i], wv.min(), wv.max()) for i in range(4)]}
    web["raw"] = raw
    web["raw_step"] = 5
    # attention given to each role (mean over heads), 5-s means
    ra = d["role_att"]
    web["role_att"] = [np.round(ds(ra[:, r]), 3).tolist() for r in range(4)]
    # chat log
    chat = json.loads((OUT / "chat_team.json").read_text())
    other = json.loads((OUT / "chat_other.json").read_text())
    web["chat"] = chat
    web["chat_other"] = [m for m in other if m[0] < 1200]
    # numbers
    web["results"] = json.loads((OUT / "results.json").read_text())
    web["summary"] = json.loads((OUT / "summary_5sessions.json").read_text())
    for k in ("integration_timecourse",):
        tc = web["results"][k]
        web["results"][k] = {kk: [round(float(x), 4) for x in vv] for kk, vv in tc.items()}
    (OUT / "web_data.json").write_text(json.dumps(web, separators=(",", ":"), default=float))
    print("web_data.json", (OUT / "web_data.json").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
