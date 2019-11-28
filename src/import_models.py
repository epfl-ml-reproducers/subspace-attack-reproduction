import torch

from src.models.cifar.gdas import load_gdas
from src.models.cifar import vgg

MODELS = 'pretrained/'

MODELS_DATA = {
    'gdas': {
        'folder': 'gdas/',
        'checkpoint': 'gdas-cifar10.pth',
        'config': 'gdas-cifar10.config'
    },
    'vgg11': {
        'folder': 'vgg11_bn/',
        'checkpoint': 'vgg11_bn.pth',
        'model': vgg.vgg11_bn
    },
    'vgg13': {
        'folder': 'vgg13_bn/',
        'checkpoint': 'vgg13_bn.pth',
        'model': vgg.vgg13_bn
    },
    'vgg16': {
        'folder': 'vgg16_bn/',
        'checkpoint': 'vgg16_bn.pth',
        'model': vgg.vgg16_bn
    },
    'vgg19': {
        'folder': 'vgg19_bn/',
        'checkpoint': 'vgg19_bn.pth',
        'model': vgg.vgg19_bn
    },
}

def load_model(models_data, name, num_classes):
    model_data = models_data[name]
    
    if name == 'gdas':
        gdas_data = models_data['gdas']
        gdas = load_gdas(
            MODELS + gdas_data['folder'] + gdas_data['checkpoint'],
            MODELS + gdas_data['folder'] + gdas_data['config']
        )
        
        return gdas
    
    try:
        model = model_data['model'](num_classes=num_classes)
    except KeyError:
        print('{} is not a valid name'.format(name))
        raise
    
    model_raw_state_dict = (
        torch
            .load(
                MODELS + model_data['folder'] + model_data['checkpoint'],
                map_location='cpu'
            )['state_dict']
    )
    
    model_state_dict = { key.replace('module.', ''): model_raw_state_dict[key] for key in model_raw_state_dict }
    
    model.load_state_dict(model_state_dict)
    
    return model