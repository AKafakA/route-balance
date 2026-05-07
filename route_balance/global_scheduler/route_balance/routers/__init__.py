"""L1 Router plugins: model selection.

A Router selects *which model* should handle a request. It does not decide
*which instance* of that model — that is the L2 Dispatcher's job.

Plug-in system parallel to filters/ and dispatch/.
"""
from .base import RouterBase, RouterRequest, RouterDecision
from .factory import create_router

__all__ = ["RouterBase", "RouterRequest", "RouterDecision", "create_router"]
