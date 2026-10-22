"""Synthetic team radio chat for training and driving the tiny LLM.

Word-level grammar. Four teammates (p1-p4) and one AI teammate.
Each human message has two latent knobs:
  style  : probability of standard "brevity" phrasing (vs free-form phrasing)
  urgency: probability of urgent markers (immediate, danger close, ...)
AI replies echo grids/numbers from the conversation (so a model that learns to
copy from context gets them right), or ask to "say again" when the request was
vague. Free-form requests are vaguer, so they make AI replies less certain.
"""
from __future__ import annotations

import random

ROLES = ["p1", "p2", "p3", "p4"]
DIRS = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
TGT_B = ["veh", "trk", "pax", "mortar", "bunker", "tank"]
TGT_F = ["vehicles", "trucks", "people", "guys", "troops", "something", "a mortar team", "a bunker", "tanks"]
DIGITS = [str(d) for d in range(10)]


def grid(rng):
    return " ".join(rng.choice(DIGITS) for _ in range(6))


def num(rng, lo=1, hi=6):
    return str(rng.randint(lo, hi))


class Conversation:
    """Keeps the shared facts of one conversation so messages reuse them."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.tgt_grid = grid(rng)
        self.friendly_grid = grid(rng)
        self.n = num(rng)
        self.dir = rng.choice(DIRS)
        self.last_grid = None          # last grid a human said
        self.last_was_request = False
        self.last_vague = False

    def new_target(self):
        self.tgt_grid = grid(self.rng)
        self.n = num(self.rng)
        self.dir = self.rng.choice(DIRS)


def human_message(conv: Conversation, role: str, style: float, urgency: float, kind: str | None = None) -> str:
    """One human message as a token string (role tag first)."""
    r = conv.rng
    brev = r.random() < style
    urgent = r.random() < urgency
    if kind is None:
        kind = r.choices(
            ["target", "request", "confirm", "adjust", "friendly", "ask_ai", "status"],
            weights={
                "p1": [0.05, 0.10, 0.40, 0.05, 0.10, 0.20, 0.10],
                "p2": [0.35, 0.25, 0.05, 0.05, 0.05, 0.20, 0.05],
                "p3": [0.05, 0.40, 0.05, 0.05, 0.25, 0.15, 0.05],
                "p4": [0.35, 0.05, 0.05, 0.40, 0.05, 0.05, 0.05],
            }[role],
        )[0]
    g = conv.tgt_grid
    words: list[str]
    vague = False
    if kind == "target":
        if r.random() < 0.35:
            conv.new_target()
            g = conv.tgt_grid
        if brev:
            words = ["tgt", "grid", g, ",", conv.n, r.choice(TGT_B), "moving", conv.dir, "."]
        else:
            opts = [
                ["i", "have", "eyes", "on", conv.n, r.choice(TGT_F), "near", "grid", g, ",", "looks", "like", "they", "are", "moving", conv.dir, "."],
                ["there", "are", "maybe", conv.n, r.choice(TGT_F), "out", "by", g, "heading", conv.dir, "i", "think", "."],
                ["new", "contact", ",", "some", r.choice(TGT_F), "over", "near", "the", "ridge", "heading", conv.dir, "."],
            ]
            words = r.choice(opts)
            vague = "ridge" in words
        conv.last_grid = None if vague else g
    elif kind == "request":
        if brev:
            words = ["request", "fire", "mission", "grid", g, ",", num(r, 1, 4), "rds", "he", "."]
            conv.last_grid = g
        else:
            opts = [
                ["can", "we", "get", "fire", "on", "grid", g, ",", "maybe", num(r, 1, 4), "rounds", "of", "he", "?"],
                ["we", "need", "rounds", "on", "that", "target", "now", "please", "."],
                ["put", "some", "fire", "on", "them", "over", "there", "?"],
            ]
            words = r.choice(opts)
            vague = "grid" not in words
            conv.last_grid = None if vague else g
        conv.last_was_request = True
    elif kind == "confirm":
        if brev:
            words = r.choice([["confirm", "tgt", "id", "."], ["cleared", "hot", "."], ["check", "fire", "."], ["hold", "fire", "."]])
        else:
            words = r.choice([
                ["before", "we", "shoot", "can", "someone", "confirm", "what", "that", "target", "is", "?"],
                ["ok", "you", "are", "cleared", ",", "go", "ahead", "and", "fire", "."],
                ["hold", "on", ",", "stop", "firing", "for", "now", "."],
            ])
    elif kind == "adjust":
        if brev:
            words = r.choice([
                ["splash", ",", "add", num(r, 1, 4), "0", "0", ",", r.choice(["left", "right"]), num(r, 1, 4), "0", "."],
                ["target", "destroyed", "."],
                ["rounds", "complete", "."],
            ])
        else:
            words = r.choice([
                ["rounds", "landed", "a", "bit", "short", ",", "move", "it", "up", "about", num(r, 1, 4), "hundred", "."],
                ["that", "was", "close", ",", "i", "think", "we", "got", "them", "."],
                ["missed", "a", "little", "to", "the", r.choice(["left", "right"]), "."],
            ])
    elif kind == "friendly":
        if brev:
            words = ["friendly", "at", "grid", conv.friendly_grid, "."]
        else:
            words = ["just", "so", "everyone", "knows", "our", "guys", "are", "at", conv.friendly_grid, "."]
    elif kind == "ask_ai":
        if brev:
            words = r.choice([
                ["ai", ",", "compute", "solution", "grid", g, "."],
                ["ai", ",", "status", "?"],
                ["ai", ",", "check", "friendly", "grid", g, "."],
            ])
            conv.last_grid = g if "grid" in words else conv.last_grid
        else:
            words = r.choice([
                ["hey", "ai", "can", "you", "work", "out", "a", "firing", "solution", "for", g, "?"],
                ["ai", "what", "do", "you", "think", "about", "that", "target", "?"],
                ["ai", "are", "we", "safe", "to", "shoot", "there", "?"],
            ])
            vague = g not in " ".join(words)
            conv.last_grid = None if vague else g
        conv.last_was_request = True
    else:  # status
        words = ["status", "green", "."] if brev else ["all", "good", "on", "my", "end", "."]
    if urgent:
        words = r.choice([["immediate", ","], ["priority", ","], ["danger", "close", ","]]) + words
    conv.last_vague = vague
    return " ".join([f"<{role}>"] + words)


def ai_message(conv: Conversation) -> str:
    """Template AI reply used only to build the training corpus."""
    r = conv.rng
    g = conv.last_grid
    if g is None or conv.last_vague:
        p_say_again = 0.55
    else:
        p_say_again = 0.06
    if r.random() < p_say_again:
        words = r.choice([["say", "again", "grid", "."], ["say", "again", ",", "unable", "to", "comply", "."], ["request", "grid", "and", "target", "type", "."]])
    elif g is None:
        words = r.choice([["standing", "by", "."], ["target", "classified", conv.n, "veh", ",", "confidence", "low", "."]])
    else:
        words = r.choice([
            ["copy", "grid", g, ",", num(r, 1, 4), "rds", "he", ",", "time", "of", "flight", num(r, 1, 3), num(r, 0, 9), "sec", "."],
            ["solution", "ready", "grid", g, ",", "charge", num(r, 1, 7), "."],
            ["warning", "friendly", "within", num(r, 1, 9), "0", "0", "m", "of", "grid", g, "."],
            ["target", "classified", conv.n, "veh", ",", "confidence", r.choice(["high", "high", "medium"]), "."],
        ])
    conv.last_was_request = False
    return " ".join(["<ai>"] + words)


def conversation_tokens(rng: random.Random, n_msgs: int) -> list[str]:
    conv = Conversation(rng)
    style = rng.random()
    urgency = rng.random() * 0.6
    toks: list[str] = []
    for _ in range(n_msgs):
        if conv.last_was_request and rng.random() < 0.8:
            msg = ai_message(conv)
        else:
            role = rng.choice(ROLES)
            st = min(1.0, max(0.0, style + rng.gauss(0, 0.15)))
            msg = human_message(conv, role, st, urgency)
        toks.extend(msg.split())
    return toks


def build_vocab() -> list[str]:
    rng = random.Random(0)
    seen = set()
    for _ in range(4000):
        seen.update(conversation_tokens(rng, 30))
    specials = ["<pad>", "<unk>"]
    return specials + sorted(seen)


if __name__ == "__main__":
    rng = random.Random(1)
    print(" ".join(conversation_tokens(rng, 14)))
    v = build_vocab()
    print(len(v), "tokens in vocab")
