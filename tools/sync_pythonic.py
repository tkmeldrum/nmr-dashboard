"""Copy the standalone Tecmag reader files into MeldrumLabCode/pythonic (like kea_io.py there: drop them next to a notebook).

    python tools/sync_pythonic.py [path/to/MeldrumLabCode/pythonic]

nmrdash/ in this repository is the canonical version; rerun this after changing tecmag_io.py or tecmag.py. The copies get a header saying so, and tecmag.py's
package-relative imports become plain imports so the two files work side by side without installing nmrdash. Nothing is committed in the other repository.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT.parent / "MeldrumLabCode" / "pythonic"
HEADER = '# Standalone copy of nmr-dashboard/nmrdash/{name} (canonical version there; copied by tools/sync_pythonic.py: edit it there, not here).\n'


def make_copies(dest):
    dest = Path(dest)
    out = []
    reader = (ROOT / "nmrdash" / "tecmag_io.py").read_text()
    adapter = (ROOT / "nmrdash" / "tecmag.py").read_text()
    adapter = adapter.replace("from . import inference, physics\n", "")                                    # not used by the adapter itself
    adapter = adapter.replace("from .tecmag_io import TntError, parse_value, read_header, read_tnt", "from tecmag_io import TntError, parse_value, read_header, read_tnt")
    assert "from ." not in adapter and not re.search(r"\b(inference|physics)\.", adapter), "tecmag.py uses more of nmrdash than the standalone copy carries"
    for name, text in (("tecmag_io.py", reader), ("tecmag.py", adapter)):
        (dest / name).write_text(HEADER.format(name=name) + text)
        out.append(dest / name)
    return out


if __name__ == "__main__":
    for p in make_copies(sys.argv[1] if len(sys.argv) > 1 else DEFAULT):
        print("wrote", p)
