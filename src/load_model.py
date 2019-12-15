import torch

from typing import Dict
from enum import Enum

from src.models.cifar.gdas import load_gdas
from src.models.cifar import vgg
from src.models.cifar import alexnet

class ModelType(Enum):
    """
    Enum that saves whether a model is VICTIM or REFERENCE
    """
    VICTIM = 'victim'
    REFERENCE = 'reference'

# Name of the directory that contains the pretrained models
MODELS_DIRECTORY = 'pretrained/'

""" Dict that contains data about location and type of different models,
The model has a name, and the structure of each entry is the following:
model_name: {
    folder: folder that contains the pretrained model
    checkpoint: filename of the file that contains the pretrained model (.pth)
    config: (only for gdas, at the moment), filename of the file that contains configurations for the model
    type: the kind of model (either VICTIM of REFERENCE)
    default: whether is a model to be used by default in the experiment
}
"""
MODELS_DATA = {
    'gdas': {
        'folder': 'gdas/',
        'checkpoint': 'gdas-cifar10.pth',
        'config': 'gdas-cifar10.config',
        'type': ModelType.VICTIM,
        'default': True
    },
    'vgg11_bn': {
        'folder': 'vgg11_bn/',
        'checkpoint': 'vgg11_bn.pth',
        'model': vgg.vgg11_bn,
        'type': ModelType.REFERENCE,
        'default': True
    },
    'vgg13_bn': {
        'folder': 'vgg13_bn/',
        'checkpoint': 'vgg13_bn.pth',
        'model': vgg.vgg13_bn,
        'type': ModelType.REFERENCE,
        'default': True
    },
    'vgg16_bn': {
        'folder': 'vgg16_bn/',
        'checkpoint': 'vgg16_bn.pth',
        'model': vgg.vgg16_bn,
        'type': ModelType.REFERENCE,
        'default': True
    },
    'vgg19_bn': {
        'folder': 'vgg19_bn/',
        'checkpoint': 'vgg19_bn.pth',
        'model': vgg.vgg19_bn,
        'type': ModelType.REFERENCE,
        'default': True
    },
    'AlexNet_bn': {
        'folder': 'alexnet_bn/',
        'checkpoint': 'alexnet_bn.pth',
        'model': alexnet.alexnet_bn,
        'type': ModelType.REFERENCE,
        'default': True
    },
}


def load_model(name: str, num_classes: int) -> torch.nn.Module:
    """
    Loads a pretrained model from storage. Pretrained models must be stored in `/pretrained` folder, and
    must be in the `MODELS_DATA` dict. The pretrained models we are using for the experiment can be
    downloaded [here](https://drive.google.com/file/d/1TA-UWYVDkCkNPOy1INjUU9321s-HA6RF/view).

    Parameters
    ------
    name: str
        The name of the model to be loaded.
    
    num_classes: int
        The number of classes the model must be loaded with.

    Returns
    -------
    model: torch.nn.Module
        The pretrained model, ready to be used.
    
    Raises
    ------
    NotImplementedError
        If the name of the model is not valid.
    """

    # Check if the model name is valid
    if name not in MODELS_DATA:
        raise NotImplementedError(
            f'{name} is not a valid name, must be one of {list(MODELS_DATA.keys())}'
        )
    
    # Get the data about the model
    model_data = MODELS_DATA[name]

    # If the model is GDAS, treat it differently
    if name == 'gdas':
        # Get GDAS data
        gdas_data = MODELS_DATA['gdas']

        # Load GDAS using config and data
        gdas = load_gdas(
            MODELS_DIRECTORY + gdas_data['folder'] + gdas_data['checkpoint'],
            MODELS_DIRECTORY + gdas_data['folder'] + gdas_data['config']
        )

        return gdas

    # Initialize the model with the right number of classes
    model = model_data['model'](num_classes=num_classes)

    # Load the state dict of the pretrained model
    model_raw_state_dict = (
        torch
        .load(
            MODELS_DIRECTORY + model_data['folder'] + model_data['checkpoint'],
            map_location='cpu'
        )['state_dict']
    )

    # Rename the keys of the state dict (the original ones are not those expected by the model)
    model_state_dict = {key.replace(
        'module.', ''): model_raw_state_dict[key] for key in model_raw_state_dict}

    # Load state dict in the model and return
    model.load_state_dict(model_state_dict)

    return model
