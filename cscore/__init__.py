"""C-SCoRe: Correlation Signal Co-cluster Reduction.

    import cscore as cs
    cs.cap.xi_cor(x, y)
    cs.modules.module_eigengenes(expr, partition)

자주 쓰는 함수는 최상위에서도 바로 쓸 수 있음 (cs.xi_cor, cs.graph_process_ckpt ...).
"""
from . import cap, modules, compare, enrich, plotting, pipeline, shift
from .cap import *          # noqa: F401,F403
from .modules import *      # noqa: F401,F403
from .compare import *      # noqa: F401,F403
from .enrich import *       # noqa: F401,F403
from .plotting import *     # noqa: F401,F403
from .pipeline import *     # noqa: F401,F403
from .shift import *        # noqa: F401,F403

__version__ = '0.1.0'



