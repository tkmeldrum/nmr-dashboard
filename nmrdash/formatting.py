"""Number formatting shared by the summary files; the same rules as the web page (plotutil.js).

T2 values and their uncertainties are scientific with ONE base exponent: 1.980e-2 +/- 0.049e-2, uncertainty to
2 significant figures.
"""
import math


def _exp_str(mant, e, dec):
    return f"{mant:.{dec}f}e{'-' if e < 0 else '+'}{abs(e)}"


def fmt(x, sig=3):
    """3 significant figures; exponent form for very large or small values."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return fmt_sci(x, sig)
    text = repr(float(f"{x:.{sig}g}"))          # 3 significant figures, then plain digits (1310, 0.0169), like the page
    return text[:-2] if text.endswith(".0") else text


def fmt_sci(x, sig=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    mant = x / 10 ** e
    if abs(float(f"{mant:.{sig - 1}f}")) >= 10:
        e += 1
        mant = x / 10 ** e
    return _exp_str(mant, e, sig - 1)


def sci_group(v, se=None):
    """A formatter that writes v, its SE and interval limits with one shared base exponent."""
    ref = abs(v) if v else abs(se or 0)
    if not ref:
        return fmt_sci
    dec_for = lambda e: min(6, max(2, 1 - math.floor(math.log10(se / 10 ** e)))) if se and se > 0 else 2
    e = math.floor(math.log10(ref))
    if float(f"{ref / 10 ** e:.{dec_for(e)}f}") >= 10:
        e += 1
    dec = dec_for(e)
    return lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else _exp_str(x / 10 ** e, e, dec)


def is_odd(x):
    return x is not None and x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4)


def ok_unc(v, se):
    return se is not None and not (isinstance(se, float) and (math.isnan(se) or math.isinf(se))) and (v == 0 or se <= 10 * abs(v))


def pair(v, se, sci=False):
    """'value ± SE' in one notation (scientific with a shared exponent if either needs it, or if sci=True)."""
    if not ok_unc(v, se):
        return f"{fmt_sci(v) if sci else fmt(v)} (uncertainty undetermined)"
    if sci or is_odd(v) or is_odd(se):
        g = sci_group(v, se)
        return f"{g(v)} ± {g(se)}"
    return f"{fmt(v)} ± {fmt(se)}"


def interval(v, lo, hi, se=None, sci=False):
    """'value [lo, hi]' in one notation per call."""
    if sci or is_odd(v) or is_odd(se):
        g = sci_group(v, se if ok_unc(v, se) else None)
        return f"{g(v)} [{g(lo)}, {g(hi)}]"
    return f"{fmt(v)} [{fmt(lo)}, {fmt(hi)}]"


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"
