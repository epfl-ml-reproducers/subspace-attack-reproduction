import torch

from torchvision import datasets, transforms
from typing import Tuple, Dict

""" Dict that contains data about datasets to be used,
The dataset has a name, and the structure of each entry is the following:
dataset_name: {
    classes: the classes of the dataset,
    dataset: the `torchvision.datasets` dataset to be used,
    default: whether it is the default dataset to be used
}
"""
DATASETS_DATA = {
    'CIFAR-10': {
        'classes': ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'),
        'dataset': datasets.CIFAR10,
        'default': True
    }
}


def load_data(dataset: str) -> Tuple[torch.utils.data.DataLoader, Tuple[str]]:
    """
    Loads a a dataset from `torchvision.datasets`. It loads the dataset, it transforms each entry
    to a torch.Tensor, and finally returns a PyTorch dataloader. The dataset is returned as a
    training dataset, as we need the true label to check if the attack is successful.
    
    The dataset is being downloaded in the `data/` folder, which is .gitignore-d by default.
    For the moment, only CIFAR-10 is implemented.

    Parameters
    ------
    dataset: str
        The name of the dataset to be loaded.

    Returns
    -------
    dataset, classes: Tuple[torch.utils.data.DataLoader, Tuple[str]]
        The pretrained model, ready to be used.

    Raises
    ------
    NotImplementedError
        If the name of the dataset is not valid.
    """
    # Check if the dataset name is valid
    if dataset not in list(DATASETS_DATA.keys()):
        raise NotImplementedError(
            f'{dataset} is not a valid name, must be one of {list(DATASETS_DATA.keys())}'
        )

    # Get data about the dataset
    dataset_info = DATASETS_DATA[dataset]

    # Load the dataset from torchvision.datasets
    data = dataset_info['dataset'](root='./data', train=True,
                                   download=True, transform=transforms.ToTensor())

    # Put the dataset in a DataLoader and return it with the classes
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=True, num_workers=2)

    return data_loader, dataset_info['classes']
