import torch

from enum import Enum


class ExperimentLoss(Enum):
    """
    Enum that saves which kind of losses can be used in the experiment.
    """
    # CrossEntropy Loss
    CROSS_ENTROPY = 'CrossEntropy'
    # Negative Log Likelyhood Loss
    NEG_LL = 'NLL'


DEFAULT_LOSS = ExperimentLoss.CROSS_ENTROPY

LOSSES = {
    ExperimentLoss.CROSS_ENTROPY: torch.nn.CrossEntropyLoss,
    ExperimentLoss.NEG_LL: torch.nn.NLLLoss
}


def load_loss(loss: ExperimentLoss) -> torch.nn.modules.loss._Loss:
    """
    Loads a loss function from torch.nn. It must one of those in `Losses`

    Parameters
    ------
    loss: ExperimentLoss
        The name of the loss.

    Returns
    -------
    loss: torch.nn.modules.loss._Loss
        The loss function, ready to be used.

    Raises
    ------
    NotImplementedError
        If the name of the loss is not valid.
    """
    # Check if the loss name is valid
    if loss not in ExperimentLoss:
        raise NotImplementedError(
            f'{loss} is not a valid name, must be one of {list(LOSSES.keys())}'
        )

    return LOSSES[loss]()
