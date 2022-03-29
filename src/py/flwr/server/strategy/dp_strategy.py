"""DP-FedAvg [McMahan et al., 2018] strategy.

Paper: https://arxiv.org/pdf/1710.06963.pdf
"""


from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy

from .aggregate import aggregate_qffl, weighted_loss_avg

class DP_Strategy(Strategy):
    """Configurable DP strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    # to edit
    # parameter for choosing kind of sampling - random or poisson
    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float = 1,
        clip_norm:float = 0.1,
        adaptive_clip_enabled = False,
    ) -> None:
        super().__init__(
            fraction_fit=strategy.fraction_fit,
            fraction_eval=strategy.fraction_eval,
            min_fit_clients=strategy.min_fit_clients,
            min_eval_clients=strategy.min_eval_clients,
            min_available_clients=strategy.min_available_clients,
            eval_fn=strategy.eval_fn,
            on_fit_config_fn=strategy.on_fit_config_fn,
            on_evaluate_config_fn=strategy.on_evaluate_config_fn,
            accept_failures=strategy.accept_failures,
            initial_parameters=strategy.initial_parameters,
        )
        self.noise_multiplier = noise_multiplier
        self.clip_norm = clip_norm
        self.adaptive_clip_enabled = adaptive_clip_enabled

    
    def __repr__(self) -> str:
        rep = f"Strategy with DP enabled."
        return rep



    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        client_instructions = super.configure_fit(rnd, parameters, client_manager)
        
        for _, fit_ins in client_instructions:
            fit_ins.config["clip_norm"] = self.clip_norm

        return client_instructions

    
    # do clip norm calculation and then call super.aggregate_fit()
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        # add noise to individual updates
        return super.aggregate_fit(rnd, results, failures)



