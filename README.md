# Human–AI OP-TEMPO simulation

Code and data behind the lightning-talk deck and the interactive walkthrough.
All data are simulated. The coupling strengths were chosen to demonstrate the pipeline, so none of the numbers are empirical findings.

**[Open the interactive demo](https://xyin20.github.io/human-ai-optempo-sim/)**

The demo runs entirely in your browser, including the transformer forward pass. No account, API key, or backend is needed.

## Build and preview the demo

Python 3 is the only build requirement; no Python packages need to be installed for the website.

```bash
python web/build.py
python -m http.server 8000 --directory web/dist
```

Open `http://localhost:8000`. The build produces `web/dist/index.html` and the downloadable `web/dist/optempo-walkthrough.html`. Both include the showcase data and model weights. The supplied `sim/out/web_data.json` preserves the data embedded in the original walkthrough, so building the website does not require rerunning the simulation.

GitHub Actions builds and deploys the demo to GitHub Pages after each push to `main`. The repository's Pages publishing source must be set to **GitHub Actions**.

## What is here

| File | What it does |
|---|---|
| `corpus.py` | Word-level grammar for synthetic team radio chat: four teammates (p1–p4) and the AI. Each message has a brevity-vs-free-form style and an urgency level. |
| `tinylm.py` | **TeamTalk-GPT**: a decoder-only transformer trained from scratch in JAX. 2 layers × 4 heads, d = 64, 64-token context, 166-word vocabulary, 124,928 parameters. Trains in about 4 minutes on 2 CPU cores. |
| `cosim.py` | 1-Hz co-simulation, 3,600 s in three 1,200-s phases (Separate → AI adapts → Mutual). The LLM runs in the loop: it reads the chat, replies, and every AI signal comes from real forward passes. Human physiology (the 9 OP-TEMPO sensors) is generated from each person's latent workload and engagement. |
| `optempo.py` | Port of the OP-TEMPO pooled pipeline: quartile symbols, the seven pooling rules, concatenated state labels, 30-s windowed entropy, 200-s smoothing, drop 159 s, MI = H(X)+H(Y)−H(X,Y). |
| `analysis.py`, `results.py` | Human, AI and human–AI pools; integration I = H(people) + H(AI) − H(people ⊕ AI); within-phase circular-shift surrogates; leave-one-out person-level measures. |
| `export_web.py` | Packs the showcase session for the web page. |
| `multi_seed.sh` | Runs sessions with seeds 8–11 (seed 7 is the showcase). |
| `out/simulated_session_seed7.csv` | The showcase session, one row per second: 36 human sensor columns (P1–P4 × 9), 32 attention-head columns, 3 model columns, latent workload. |
| `out/summary_5sessions.json` | Numbers used on the slides (5 sessions, seeds 7–11). |
| `out/teamtalk_gpt.npz`, `out/vocab.json` | Trained model weights and vocabulary. |
| `web/` | Source of the interactive walkthrough (the JS forward pass matches the Python model to within 1e-5 on the logits). |
| `sim/out/web_data.json` | Packaged showcase data used to build the standalone demo. |

Simulation source files and their `out/` data directory are under `sim/`.

## Re-run

```bash
cd sim
pip install jax numpy scipy pandas
python tinylm.py          # trains TeamTalk-GPT (about 4 min)
python cosim.py 7         # one simulated session -> out/cosim.npz
python results.py         # pooled measures -> out/results.json
bash multi_seed.sh        # four more sessions
```

## The design in one paragraph

People are pooled across the four humans inside each sensor, exactly as in OP-TEMPO. The AI is pooled the same way, with its eight attention heads as the members (attention entropy, attention on teammates' tokens, look-back distance, head output norm), plus three single-member model signals (surprise at teammates' words, reply uncertainty, tokens read in the last 20 s). Pooled sensors are concatenated into a people state, an AI state and a system state. The information they share, I = H(people) + H(AI) − H(people ⊕ AI), is compared with 200 surrogates that circularly shift the AI stream inside the same phase. That keeps each stream's own autocorrelation and alphabet and breaks only their simultaneity.

## Main results (majority-high pooling, 5 sessions)

| | Separate | AI adapts | Mutual |
|---|---|---|---|
| I, autonomic ↔ AI output, bits above chance (mean) | −0.02 | 0.17 | 0.21 |
| Sessions above chance (p < .05) | 0 / 5 | 5 / 5 | 5 / 5 |
| Person's own autonomic state ↔ AI output (mean of 4 teammates) | −0.01 | 0.11 | 0.16 |
| Person-sessions above chance | 0 / 20 | 19 / 20 | 17 / 20 |
| People pooled entropy, 30-s windows (bits) | 3.55 | 3.76 | 3.33 |

Raw concatenation of every symbol makes all 3,600 seconds unique states (30-s entropy pinned at log2 30 = 4.91 bits). Majority pooling gives about 342 people states, 116 AI states and 2,083 system states.

## Things worth knowing

- **30-s windows mostly miss slow coupling.** The whole-phase measure detects it in every session; 30-s windowed MI barely moves. This matches the OP-TEMPO RQA smoke test (team structure, if any, is at slower time scales).
- **MI is symmetric.** It detects coupling but cannot tell one-way from mutual. Directed measures (lagged AMI, transfer entropy) are the next step.
- **Surrogates must stay inside the phase.** Shifting across the whole session pairs segments with different alphabets and biases the null; within-phase shifts fixed that here.
- **Two differences from the production notebook:** AMI windows are aligned with the entropy windows (both end at t), and person-level measures use a leave-one-out pool, as flagged in the review copy (P08, P21).
- **Pooling rule matters.** The walkthrough page lets you switch among all seven rules.

## Numbers in the literature review that disagree with each other

Spotted while building the slides (worth fixing before submission):

- Neural-level adaptation: Table 6 and the later paragraph say 6 (6.9%); the "Levels of Adaptation" text says 4 (4.6%).
- Sensorimotor / behavioral / cognitive: Table 6 says 29 / 47 / 42; the text says 24 / 46 / 40.
- Prediction and alignment: Table 5 says 43 (49.4%) and 52 (59.8%); the text says prediction 42 (48.3%) and alignment 43 (49.4%).
- Negotiation: Table 5 says 10 (11.5%); the text says 7 (8.0%).
- RQ2 says "33 framework families"; Table 3 and its text say 37.

The deck uses only the figures that agree across the paper (locus 95.4% / 20.7%, time scale 73.6% / 10.3%, terminology from Table 2) plus neural 6 of 87 from Table 6.
