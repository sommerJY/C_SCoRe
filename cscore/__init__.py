"""C-SCoRe: Correlation Signal Co-cluster Reduction (two-condition comparison).

    import cscore as cscr
    cscr.cap.xi_cor(x, y)
    cscr.shift.cap_shift(caps, base=0, target=1)

All functions are also available directly at the top level (cscr.xi_cor, cscr.graph_process_ckpt ...).
"""
from . import cap, modules, plotting, pipeline, shift
from .cap import *          # noqa: F401,F403
from .modules import *      # noqa: F401,F403
from .plotting import *     # noqa: F401,F403
from .pipeline import *     # noqa: F401,F403
from .shift import *        # noqa: F401,F403

__version__ = '0.1.0'