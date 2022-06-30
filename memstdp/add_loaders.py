from typing import Iterable, Iterator, Optional, Union

import torch

from bindsnet.encoding.encodings import bernoulli, poisson, rank_order, rank_order_TTFS

def rank_order_TTFS_loader(
    data: Union[torch.Tensor, Iterable[torch.Tensor]],
    time: int,
    dt: float = 1.0,
    **kwargs,
) -> Iterator[torch.Tensor]:
    # language=rst
    """
    Lazily invokes ``bindsnet.encoding.rank_order`` to iteratively encode a sequence of
    data.

    :param data: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of rank order-encoded spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensors of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.
    """
    for i in range(len(data)):
        # Encode datum as rank order-encoded spike trains.
        yield rank_order_TTFS(datum=data[i], time=time, dt=dt)
