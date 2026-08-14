"""Reference implementation of MAML-Select.

Companion code for "MAML-Select: An Online Adaptive Client Selection Method for
Federated Learning via Meta-Learning".  The modules here are a faithful,
dependency-light extraction of the selector used to produce the reported
results, so that every equation in the paper can be read directly against code.
"""

from .cost import client_cost, cohort_surrogate, normalized_latency_penalty, target_latency
from .policy import ClientState, Feedback, MetaPolicy, build_state_matrix, parameter_count, standardize
from .selector import Diagnostics, MAMLSelect, top_k_by_cost

__all__ = [
    "MAMLSelect",
    "Diagnostics",
    "top_k_by_cost",
    "MetaPolicy",
    "ClientState",
    "Feedback",
    "build_state_matrix",
    "standardize",
    "parameter_count",
    "client_cost",
    "cohort_surrogate",
    "normalized_latency_penalty",
    "target_latency",
]
__version__ = "1.0.0"
