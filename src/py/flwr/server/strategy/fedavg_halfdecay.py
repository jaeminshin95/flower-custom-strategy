"""Custom FedAvg strategy.

Half of the clients use constant learning rate, while the other half of the clients 
are told to perform learning rate decay at each round.
"""


from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import FitIns, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .fedavg import FedAvg


# pylint: disable=line-too-long
class FedAvgHalfDecay(FedAvg):
    r"""Custom FedAvg strategy.

    Half of the clients use constant learning rate, while the other half of the
    clients are told to perform learning rate decay at each round.

    The strategy in itself will not be different than FedAvg, the client needs to
    be adjusted.

    The client needs to maintain its own learning rate and use it during training.
    If the train config contains a decay factor (learning_rate_decay_factor), the
    client needs to multiply the factor to its learning rate:

    .. code:: python

      if "learning_rate_decay_factor" in config:
          self.learning_rate *= config["learning_rate_decay_factor"]

    This strategy maintains a list of ids of half of the clients which continuously
    decay the learning rate. Currently, such clients are determined based on the
    order of sampled clients list provided by the client_manager at the first round.
    This policy could be further changed.

    Full client participation per round (cross-silo setting) is assumed.

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    learning_rate_decay_factor : float
        A factor used to decay the learning rate of a client.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        learning_rate_decay_factor: float,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.decay_clients = []

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedHalfDecay(accept_failures={self.accept_failures})"
        return rep

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Pass the learning rate decay factor to half of the clients at each round.
        """
        # Get the standard client/config pairs from the FedAvg super-class
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Identify the middle index that divides the clients set into half
        mid_client_idx = len(client_config_pairs) // 2

        # At round 1, identify clients to decay the learning rate. Currently, the clients
        # with index >= mid_client_idx in the client_config_pairs list are determined to
        # decay the learning rate.
        # learning_rate_decay_factor is not passed to clients at round 1.
        if server_round == 1:
            self.decay_clients = [
                client.cid for client, fit_ins in client_config_pairs[mid_client_idx:]
            ]
            return client_config_pairs
        # At round > 1, learning_rate_decay_factor is passed to half of the clients.
        else:
            new_client_config_pairs = []

            for client, fit_ins in client_config_pairs:
                if client.cid in self.decay_clients:
                    fit_ins = FitIns(
                        fit_ins.parameters,
                        {
                            **fit_ins.config,
                            "learning_rate_decay_factor": self.learning_rate_decay_factor,
                        },
                    )
                new_client_config_pairs.append((client, fit_ins))

            return new_client_config_pairs
