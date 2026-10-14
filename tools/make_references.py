"""Write docs/REFERENCES.md from docs/references.json:  python tools/make_references.py

Edit the JSON (fix a reference, set "verified": true once it has been checked), then run this to regenerate the readable list.
Only verified references are shown on the About page.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def render():
    data = json.loads((ROOT / "docs" / "references.json").read_text())
    n_ok = sum(1 for r in data["references"] if r.get("verified"))
    L = ["# References to check", "",
         "These references were written from memory when the models were documented and have **not been checked**. For each row: is this the right source for the relation, "
         "and are the journal, volume, page and year right? Correct anything that is wrong in `docs/references.json` (or tell me), set `\"verified\": true`, and run "
         "`python tools/make_references.py`. **Only verified references are shown on the About page**; until then the page shows the equation, the description and the code link, without a reference.", "",
         f"Verified so far: {n_ok} of {len(data['references'])}.", "",
         "| Verified | Relation or software | Code | Draft reference |", "|---|---|---|---|"]
    for r in data["references"]:
        funcs = ", ".join(f"`{f}`" for f in r["functions"]) or "–"
        L.append(f"| {'yes' if r.get('verified') else 'no'} | **{r['name']}** (`{r['id']}`) | {r['file']}: {funcs} | {r['reference']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    (ROOT / "docs" / "REFERENCES.md").write_text(render())
    print("wrote docs/REFERENCES.md")
