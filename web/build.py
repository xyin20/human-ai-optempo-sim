from pathlib import Path
H = Path(__file__).resolve().parent
S = H.parent / "sim" / "out"
page = (H / "page.html").read_text(encoding="utf-8")
data = (S / "web_data.json").read_text(encoding="utf-8").replace("</", "<\\/")
weights = (S / "teamtalk_gpt_weights.json").read_text(encoding="utf-8").replace("</", "<\\/")
out = (page.replace("{{DATA}}", data).replace("{{WEIGHTS}}", weights)
       .replace("{{TINYLM}}", (H / "tinylm.js").read_text(encoding="utf-8"))
       .replace("{{PIPELINE}}", (H / "pipeline.js").read_text(encoding="utf-8"))
       .replace("{{APP}}", (H / "app.js").read_text(encoding="utf-8")))
(H / "dist").mkdir(exist_ok=True)
for filename in ("index.html", "optempo-walkthrough.html"):
    (H / "dist" / filename).write_text(out, encoding="utf-8")
print("built", len(out) // 1024, "KB")
