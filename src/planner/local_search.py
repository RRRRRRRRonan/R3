"""Local search heuristics that complement the ALNS optimizer.

This module is reserved for neighbourhood operators that fine-tune
single-vehicle routes after the large neighbourhood search phase.  The
implementation was previously embedded inside :mod:`planner.alns`, and the
module is intentionally kept as a placeholder so new heuristics (2-opt, relocate
chains, etc.) can live in a dedicated namespace.  The detailed operators were
removed during the refactor and will be added back when the hybrid solver
pipeline is expanded.
"""

