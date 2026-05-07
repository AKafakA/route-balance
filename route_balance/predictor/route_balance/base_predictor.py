"""
Base predictor class for RouteBalance predictors.

Separate from RouteBalance's Predictor to use ROUTE_BALANCE-specific PredictRequest.
"""
from abc import ABC, abstractmethod
from typing import Dict
from route_balance.predictor.route_balance.data_structures import PredictRequest


class ROUTE_BALANCE_BasePredictor(ABC):
    """Base class for RouteBalance predictors.

    Uses PredictRequest instead of Vidur Request to stay independent from RouteBalance/Vidur.
    """

    def __init__(self, config, port: int) -> None:
        """Initialize predictor.

        Args:
            config: Predictor configuration (RouteBalanceBasePredictorConfig)
            port: Backend instance port
        """
        self._config = config
        self._instance_port = port

    @abstractmethod
    async def predict(self, target_request: PredictRequest) -> Dict:
        """Predict metrics for a target request.

        Args:
            target_request: Request information

        Returns:
            Dict with prediction metrics:
            {
                "target_metric": float,  # Lower is better for scheduling
                "gpu_blocks": int,
                "num_requests": int,
                "num_preempted": int,
                "predictor_type": str
            }
        """
        pass