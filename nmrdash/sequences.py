"""Which analysis a dataset gets, read from the sequence name in acqu.par (the `experiment` key)."""
import re

# normalised name -> kind. Spelling variants (case, hyphens, spaces) are tolerated: "SSE-T2" == "SSET2".
KINDS = {"cpmg": "cpmg", "t1sat": "t1", "sset2": "t2d"}

ANALYSIS = {
    "cpmg": "exponential decay (T₂)",
    "t1": "saturation recovery (T₁)",
    "t2d": "T₂–D correlation",
}
SUPPORTED = {"cpmg", "t1", "t2d"}


def normalise(name):
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def kind_of(params):
    """'cpmg', 't1', 't2d' or None for an unrecognised sequence."""
    return KINDS.get(normalise(params.get("experiment")))


def describe(params):
    """{kind, supported, analysis, reason} for the page and the runner."""
    kind = kind_of(params)
    name = params.get("experiment")
    if kind is None:
        return {"kind": None, "supported": False, "analysis": None,
                "reason": f'the sequence "{name}" is not one nmrdash can analyse (CPMG, T1Sat)' if name else "acqu.par has no `experiment` entry"}
    if kind not in SUPPORTED:
        return {"kind": kind, "supported": False, "analysis": ANALYSIS[kind],
                "reason": f'{ANALYSIS[kind]} analysis (sequence "{name}") is not available yet'}
    return {"kind": kind, "supported": True, "analysis": ANALYSIS[kind], "reason": ""}
