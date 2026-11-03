"""C-SCoRe: Correlation Signal Co-cluster Reduction.

    import cscore as cs
    cs.cap.xi_cor(x, y)
    cs.modules.module_eigengenes(expr, partition)

"""
from . import cap, modules, compare, enrich, plotting
from .cap import *          # noqa: F401,F403
from .modules import *      # noqa: F401,F403
from .compare import *      # noqa: F401,F403
from .enrich import *       # noqa: F401,F403
from .plotting import *     # noqa: F401,F403

__version__ = '0.1.0'






