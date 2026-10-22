"""Human-AI team co-simulation at 1 Hz with TeamTalk-GPT in the loop.

Four teammates (P1-P4) share a chat channel with one AI teammate.
Every AI number comes from real forward passes of the tiny transformer on the chat
text; nothing on the AI side is drawn from a formula.

Phases (1,200 s each):
  1 Separate   the AI works another team's channel; no human-AI coupling
  2 AI adapts  the AI reads this team's channel (human -> AI); its replies are not shown
  3 Mutual     the AI reads the team AND its replies change the humans: visible AI
               errors raise workload and push people toward brevity phrasing;
               correct replies offload work (AI -> human)

Human -> AI pathway: workload changes how people write. Under load, messages come
faster, carry urgency markers and slip from brevity codes into vague free-form text.
The model's attention, surprisal and uncertainty respond to that text.

Human physiology (9 OP-TEMPO sensors) is generated from each person's latent
workload and engagement with person-specific baselines, gains and noise.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from corpus import Conversation, human_message
from tinylm import CFG, forward, load

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

T = 3600
PHASES = [(0, 1200, "Separate"), (1200, 2400, "AI adapts"), (2400, 3600, "Mutual")]
ROLES = ["P1", "P2", "P3", "P4"]
ROLE_TAG = ["p1", "p2", "p3", "p4"]
SENSORS = ["ibi", "eeg_delta", "eeg_theta", "eeg_alpha", "eeg_beta", "hbo", "hbr", "rsp", "pupil"]
NEURO = ["eeg_delta", "eeg_theta", "eeg_alpha", "eeg_beta", "hbo", "hbr"]
AUTO = ["ibi", "rsp", "pupil"]
HEAD_SENSORS = ["attn_entropy", "partner_share", "lookback", "head_norm"]
MODEL_SENSORS = ["surprisal", "uncertainty", "load"]
AI_ROLE = 4
OTHER = 5                 # tokens from the other team's channel (phase 1)
WIN_AI = 20               # seconds of tokens that make up the AI's current state


def phase_of(t):
    for i, (a, b, _) in enumerate(PHASES):
        if a <= t < b:
            return i
    return len(PHASES) - 1


class LLM:
    def __init__(self):
        self.p, self.vocab = load()
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.ctx = CFG["ctx"]
        self._f = jax.jit(lambda p, idx, m: forward(p, idx, m, CFG["H"]))

    def ids(self, words):
        return [self.stoi.get(w, 1) for w in words]

    def run(self, ids, authors):
        ids = ids[-self.ctx:]; authors = authors[-self.ctx:]
        pad = self.ctx - len(ids)
        idx = np.array([0] * pad + list(ids), dtype=np.int32)[None]
        mask = np.array([False] * pad + [True] * len(ids))[None]
        auth = np.array([-1] * pad + list(authors))
        logits, att, hout, resid = self._f(self.p, jnp.asarray(idx), jnp.asarray(mask))
        return np.asarray(logits[0]), np.asarray(att[:, 0]), np.asarray(hout[:, 0]), np.asarray(resid[0]), auth, idx[0], pad

    def token_records(self, ids, authors, n_new):
        """Per-token readings for the last n_new tokens of the context."""
        logits, att, hout, resid, auth, idx, pad = self.run(ids, authors)
        n = self.ctx
        q = np.arange(max(pad, n - n_new), n)
        partner = (auth >= 0) & (auth != AI_ROLE)
        L, H = att.shape[0], att.shape[1]
        A = att[:, :, q, :]                                           # [L,H,K,n]
        ent = -(A * np.log2(np.clip(A, 1e-12, 1))).sum(-1)            # [L,H,K]
        share = A[..., partner].sum(-1)
        dist = (A * (q[:, None] - np.arange(n)[None])[None, None]).sum(-1)
        norm = np.linalg.norm(hout[:, :, q, :], axis=-1)
        lsm = logits - logits.max(1, keepdims=True)
        lsm = lsm - np.log(np.exp(lsm).sum(1, keepdims=True))
        pr = np.exp(lsm)
        pent = -(pr * lsm).sum(1) / math.log(2)                      # predictive entropy per position
        role_att = np.zeros((len(q), 4))
        for r in range(4):
            role_att[:, r] = A[..., auth == r].sum(-1).mean((0, 1))
        recs = []
        for k, pos in enumerate(q):
            sur = float(-lsm[pos - 1, idx[pos]] / math.log(2)) if pos - 1 >= pad else np.nan
            recs.append({
                "author": int(auth[pos]),
                "ent": ent[:, :, k].reshape(-1), "share": share[:, :, k].reshape(-1),
                "dist": dist[:, :, k].reshape(-1), "norm": norm[:, :, k].reshape(-1),
                "surprisal": sur, "pent": float(pent[pos]), "role_att": role_att[k],
            })
        return recs

    def generate(self, ids, authors, rng, temp=1.0, topk=12, max_len=22):
        ids = list(ids) + [self.stoi["<ai>"]]
        authors = list(authors) + [AI_ROLE]
        words, ents = [], []
        for _ in range(max_len):
            logits = self.run(ids, authors)[0][-1] / temp
            lp = logits - logits.max()
            pr = np.exp(lp) / np.exp(lp).sum()
            ents.append(float(-(pr * np.log2(np.clip(pr, 1e-12, 1))).sum()))
            top = np.argsort(pr)[::-1][:topk]
            pt = pr[top] / pr[top].sum()
            nxt = int(rng.choice(top, p=pt))
            w = self.vocab[nxt]
            if w.startswith("<"):
                break
            words.append(w); ids.append(nxt); authors.append(AI_ROLE)
            if w in (".", "?"):
                break
        return words, float(np.mean(ents)) if ents else 0.0


class Channel:
    def __init__(self):
        self.ids: list[int] = []
        self.auth: list[int] = []
        self.log: list[tuple[int, str, str]] = []

    def post(self, t, speaker, author, words, llm):
        self.ids += llm.ids(words)
        self.auth += [author] * len(words)
        self.log.append((t, speaker, " ".join(words)))


def grid_in(words):
    digs, best = [], None
    for w in words:
        if w.isdigit() and len(w) == 1:
            digs.append(w)
            if len(digs) >= 6:
                best = " ".join(digs[-6:])
        else:
            digs = []
    return best


class HumanTeam:
    """Latent workload, engagement and phrasing style for four people, plus messages."""

    def __init__(self, rng: np.random.Generator, pyrng: random.Random):
        self.rng, self.pyrng = rng, pyrng
        self.mu = rng.normal(0, 0.2, 4)
        self.sens = np.array([0.8, 1.2, 1.0, 1.1])
        self.w = rng.normal(0, 0.4, 4)
        self.e = rng.normal(0, 0.4, 4)
        self.s_pref = rng.uniform(0.45, 0.6, 4)
        self.s = self.s_pref.copy()
        self.D = 0.0
        self.err_trace = 0.0
        self.ok_trace = 0.0
        self.warn_trace = 0.0
        self.ask_ai = 0.0
        self.conv = Conversation(pyrng)
        self.queue: list[tuple[int, int, str]] = []

    def step(self, t, ai_coupling: float, new_err: int, new_ok: int, new_warn: int = 0):
        r = self.rng
        if r.random() < 1 / 110:
            self.D += r.uniform(0.8, 1.5)
            self.conv.new_target()
            spotter = int(r.choice([1, 3]))
            self.queue += [(t + int(r.integers(0, 5)), spotter, "target"),
                           (t + int(r.integers(6, 16)), 2, "request"),
                           (t + int(r.integers(10, 22)), 0, "confirm"),
                           (t + int(r.integers(30, 60)), 3, "adjust")]
        self.D *= math.exp(-1 / 45)
        self.err_trace = self.err_trace * math.exp(-1 / 25) + new_err
        self.ok_trace = self.ok_trace * math.exp(-1 / 25) + new_ok
        self.warn_trace = self.warn_trace * math.exp(-1 / 25) + new_warn
        th = 1 / 18
        others = (self.w.sum() - self.w) / 3
        target = (self.mu + 1.1 * self.D * self.sens + 0.25 * (others - self.w)
                  + ai_coupling * (0.8 * math.tanh(self.err_trace) + 0.5 * math.tanh(self.warn_trace) - 0.45 * math.tanh(self.ok_trace)))
        self.w = self.w + th * (target - self.w) + 0.45 * math.sqrt(2 * th) * r.normal(0, 1, 4)
        th_e = 1 / 40
        self.e = self.e + th_e * (0 - self.e) + 0.5 * math.sqrt(2 * th_e) * r.normal(0, 1, 4)
        # phrasing: AI errors push people toward brevity codes; otherwise drift back
        self.s = self.s + (self.s_pref - self.s) / 80 + ai_coupling * 0.15 * new_err * (1 - self.s)
        self.s = np.clip(self.s, 0, 1)

    def messages(self, t):
        out = []
        due = [q for q in self.queue if q[0] <= t]
        self.queue = [q for q in self.queue if q[0] > t]
        for _, i, kind in due:
            out.append((i, kind))
        for i in range(4):
            lam = (1 / 34) * math.exp(0.65 * min(max(self.w[i], -2.0), 2.0))
            if self.rng.random() < lam:
                out.append((i, None))
        msgs = []
        for i, kind in out:
            if kind is None and self.rng.random() < self.ask_ai:
                kind = "ask_ai"
            style = float(np.clip(self.s[i] - 0.32 * self.w[i], 0.02, 0.98))   # load erodes comm discipline
            urg = 0.75 / (1 + math.exp(-2.2 * (self.w[i] - 0.7)))
            words = human_message(self.conv, ROLE_TAG[i], style, urg, kind).split()
            self.e[i] += 0.3
            msgs.append((i, words))
        return msgs


def wants_reply(words, rng):
    txt = " ".join(words)
    if "ai" in words or "request" in words or "fire" in words or "rounds" in txt:
        return rng.random() < 0.85
    if "tgt" in words or "contact" in words or "eyes" in words or "maybe" in words:
        return rng.random() < 0.35
    return rng.random() < 0.08


class AIState:
    """Rolling 20-s readings of the model, sampled each second like a sensor."""

    def __init__(self):
        self.recs: list[tuple[int, dict]] = []
        self.last = None
        self.unc_hold = 1.0

    def add(self, t, recs):
        self.recs += [(t, r) for r in recs]

    def read(self, t):
        self.recs = [(tt, r) for tt, r in self.recs if tt > t - 60]
        win = [r for tt, r in self.recs if tt > t - WIN_AI]
        if not win:
            if self.last is None:
                return None
            out = dict(self.last); out["load"] = 0.0
            return out
        out = {
            "attn_entropy": np.mean([r["ent"] for r in win], 0),
            "partner_share": np.mean([r["share"] for r in win], 0),
            "lookback": np.mean([r["dist"] for r in win], 0),
            "head_norm": np.mean([r["norm"] for r in win], 0),
            "role_att": np.mean([r["role_att"] for r in win], 0),
        }
        sur = [r["surprisal"] for r in win if r["author"] != AI_ROLE and not np.isnan(r["surprisal"])]
        out["surprisal"] = float(np.mean(sur)) if sur else (self.last["surprisal"] if self.last else 1.0)
        own = [r["pent"] for tt, r in self.recs if r["author"] == AI_ROLE and tt > t - 30]
        if own:
            self.unc_hold = float(np.mean(own))
        out["uncertainty"] = self.unc_hold
        out["load"] = float(len(win))
        self.last = out
        return out


def run(seed=7):
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    llm = LLM()
    team = HumanTeam(rng, pyrng)
    other = HumanTeam(np.random.default_rng(seed + 100), random.Random(seed + 100))
    ch_team, ch_other = Channel(), Channel()
    ai = AIState()

    W = np.zeros((T, 4)); E = np.zeros((T, 4)); S = np.zeros((T, 4)); DD = np.zeros(T)
    Wo = np.zeros((T, 4))
    head = {k: np.zeros((T, 8)) for k in HEAD_SENSORS}
    model = {k: np.zeros(T) for k in MODEL_SENSORS}
    role_att = np.zeros((T, 4))
    err_t = np.zeros(T); ok_t = np.zeros(T)
    n_msgs = np.zeros(T)
    pending = []
    pending_err = pending_ok = pending_warn = 0

    for t in range(T):
        ph = phase_of(t)
        ai_ch = ch_other if ph == 0 else ch_team
        ai_name = "other" if ph == 0 else "team"
        coupling = 1.0 if ph == 2 else 0.0

        team.ask_ai = 0.2 if ph == 2 else 0.0
        team.step(t, coupling, pending_err, pending_ok, pending_warn)
        other.step(t, 0.0, 0, 0)
        pending_err = pending_ok = pending_warn = 0
        W[t], E[t], S[t], DD[t], Wo[t] = team.w, team.e, team.s, team.D, other.w

        for teamobj, ch, name in ((team, ch_team, "team"), (other, ch_other, "other")):
            for i, words in teamobj.messages(t):
                author = i if name == "team" else OTHER
                ch.post(t, ROLES[i] if name == "team" else f"other-{ROLES[i]}", author, words, llm)
                if name == "team":
                    n_msgs[t] += 1
                if name == ai_name:
                    ai.add(t, llm.token_records(ai_ch.ids, ai_ch.auth, len(words)))
                    if wants_reply(words, rng):
                        g = grid_in(words)
                        rw, unc = llm.generate(ai_ch.ids, ai_ch.auth, rng)
                        rg = grid_in(rw)
                        txt = " ".join(rw)
                        confused = ("say again" in txt) or ("unable" in txt) or ("request grid" in txt)
                        wrong_grid = (g is not None and rg is not None and rg != g)
                        is_err = confused or wrong_grid or len(rw) < 2
                        pending.append((t + int(rng.integers(2, 5)), name, rw, unc, is_err))

        still = []
        for due, name, rw, unc, is_err in pending:
            if due <= t:
                ch = ch_team if name == "team" else ch_other
                ch.post(t, "AI", AI_ROLE, ["<ai>"] + rw, llm)
                if name == ai_name:
                    ai.add(t, llm.token_records(ch.ids, ch.auth, len(rw) + 1))
                    err_t[t] += int(is_err); ok_t[t] += int(not is_err)
                if name == "team" and ph == 2:
                    txt = " ".join(rw)
                    if is_err:
                        pending_err += 1
                    elif "warning" in txt:
                        pending_warn += 1
                    elif ("solution" in txt) or ("copy" in txt):
                        pending_ok += 1
            else:
                still.append((due, name, rw, unc, is_err))
        pending = still

        f = ai.read(t)
        if f is None:
            f = {"attn_entropy": np.ones(8), "partner_share": np.full(8, 0.5), "lookback": np.full(8, 5.0),
                 "head_norm": np.ones(8), "role_att": np.zeros(4), "surprisal": 1.0, "uncertainty": 1.0, "load": 0.0}
        for k in HEAD_SENSORS:
            head[k][t] = f[k]
        for k in MODEL_SENSORS:
            model[k][t] = f[k]
        role_att[t] = f["role_att"] if ph > 0 else 0.0
        if t % 600 == 0:
            print(f"t={t:4d} phase={PHASES[ph][2]:10s} team msgs so far={int(n_msgs.sum())} AI replies={int(err_t.sum()+ok_t.sum())}", flush=True)

    return dict(W=W, E=E, S=S, D=DD, Wo=Wo, head=head, model=model, role_att=role_att,
                err=err_t, ok=ok_t, n_msgs=n_msgs, log_team=ch_team.log, log_other=ch_other.log)


# ------------------------------------------------------------- physiology
def hrf(n=20):
    t = np.arange(n, dtype=float)
    h = (t ** 5) * np.exp(-t) / math.factorial(5) - (t ** 15) * np.exp(-t) / (6 * math.factorial(15))
    return h / h.sum()


def ou(rng, n, tau, sd=1.0):
    th = 1 / tau
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] - th * x[i - 1] + sd * math.sqrt(2 * th) * rng.normal()
    return x


def physiology(W, E, seed=11):
    """[T, 4, 9] sensor values in OP-TEMPO order."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    X = np.zeros((n, 4, len(SENSORS)))
    h = hrf()
    for i in range(4):
        w, e = W[:, i], E[:, i]
        g = rng.uniform(0.8, 1.2, 9)
        drive_hb = np.convolve(0.6 * w + 0.4 * e, h)[:n]
        mayer = np.sin(2 * np.pi * 0.1 * np.arange(n) + np.cumsum(rng.normal(0, 0.05, n)))
        X[:, i, 0] = 860 + rng.normal(0, 40) - 60 * g[0] * w + 22 * ou(rng, n, 8) + 12 * rng.normal(size=n)       # ibi ms
        X[:, i, 1] = 1.0 + 0.04 * w + 0.30 * ou(rng, n, 3) + 0.25 * rng.normal(size=n)                            # delta
        X[:, i, 2] = 0.6 + 0.28 * g[2] * w + 0.10 * e + 0.25 * ou(rng, n, 3) + 0.20 * rng.normal(size=n)         # theta
        X[:, i, 3] = 0.8 - 0.26 * g[3] * w - 0.15 * e + 0.25 * ou(rng, n, 4) + 0.20 * rng.normal(size=n)         # alpha
        X[:, i, 4] = 0.4 + 0.14 * g[4] * w + 0.25 * e + 0.20 * ou(rng, n, 3) + 0.20 * rng.normal(size=n)         # beta
        X[:, i, 5] = 0.9 * g[5] * drive_hb + 0.25 * mayer + 0.3 * ou(rng, n, 120) + 0.12 * rng.normal(size=n)   # hbo
        X[:, i, 6] = -0.35 * g[6] * drive_hb + 0.08 * mayer + 0.12 * ou(rng, n, 120) + 0.08 * rng.normal(size=n)  # hbr
        X[:, i, 7] = 15 + rng.normal(0, 1) + 2.4 * g[7] * w + 1.1 * ou(rng, n, 20) + 0.6 * rng.normal(size=n)    # rsp
        X[:, i, 8] = 3.6 + rng.normal(0, 0.3) + 0.32 * g[8] * np.r_[w[:1], w[:-1]] + 0.12 * e + 0.12 * ou(rng, n, 5) + 0.05 * rng.normal(size=n)  # pupil
    return X


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    res = run(seed)
    X = physiology(res["W"], res["E"], seed=seed + 4)
    np.savez_compressed(OUT / "cosim.npz", X=X, W=res["W"], E=res["E"], S=res["S"], D=res["D"], Wo=res["Wo"],
                        **{f"head_{k}": v for k, v in res["head"].items()},
                        **{f"model_{k}": v for k, v in res["model"].items()},
                        role_att=res["role_att"], err=res["err"], ok=res["ok"], n_msgs=res["n_msgs"])
    (OUT / "chat_team.json").write_text(json.dumps(res["log_team"]))
    (OUT / "chat_other.json").write_text(json.dumps(res["log_other"]))
    for a, b, name in PHASES:
        e = res["err"][a:b].sum(); o = res["ok"][a:b].sum()
        print(f"{name:10s} AI replies {int(e+o):4d}  error rate {e/max(e+o,1):.2f}  mean style {res['S'][a:b].mean():.2f}  mean workload {res['W'][a:b].mean():.2f}")
