"""L2 Dispatcher plugins: within-model instance selection.

A Dispatcher receives the model_id chosen by L1 Router plus the list of
instances serving that model, and returns one instance.
"""
from .base import DispatchBase, DispatchRequest, DispatchDecision
from .factory import create_dispatcher

__all__ = [
    "DispatchBase",
    "DispatchRequest",
    "DispatchDecision",
    "create_dispatcher",
]
