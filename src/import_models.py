import torch

from src.models.cifar.gdas import load_gdas
from src.models.cifar import vgg
from src.models.cifar import alexnet

MODELS_DIRECTORY = 'pretrained/'

MODELS_DATA = {
    'gdas': {
        'folder': 'gdas/',
        'checkpoint': 'gdas-cifar10.pth',
        'config': 'gdas-cifar10.config'
    },
    'vgg11_bn': {
        'folder': 'vgg11_bn/',
        'checkpoint': 'vgg11_bn.pth',
        'model': vgg.vgg11_bn
    },
    'vgg13_bn': {
        'folder': 'vgg13_bn/',
        'checkpoint': 'vgg13_bn.pth',
        'model': vgg.vgg13_bn
    },
    'vgg16_bn': {
        'folder': 'vgg16_bn/',
        'checkpoint': 'vgg16_bn.pth',
        'model': vgg.vgg16_bn
    },
    'vgg19_bn': {
        'folder': 'vgg19_bn/',
        'checkpoint': 'vgg19_bn.pth',
        'model': vgg.vgg19_bn
    },
    'AlexNet_bn': {
        'folder': 'alexnet_bn/',
        'checkpoint': 'alexnet_bn.pth',
        'model': alexnet.alexnet_bn
    },
}

def load_model(models_directory, models_data, name, num_classes):
    
    if name not in models_data:
        raise NotImplementedError(
            '{} is not a valid name, must be one of {}'.format(name, list(models_data.keys()))
        )
    
    model_data = models_data[name]
        
    if name == 'gdas':
        gdas_data = models_data['gdas']
        gdas = load_gdas(
            models_directory+ gdas_data['folder'] + gdas_data['checkpoint'],
            models_directory+ gdas_data['folder'] + gdas_data['config']
        )
        
        return gdas
    
    model = model_data['model'](num_classes=num_classes)
    
    model_raw_state_dict = (
        torch
            .load(
                models_directory+ model_data['folder'] + model_data['checkpoint'],
                map_location='cpu'
            )['state_dict']
    )
    
    model_state_dict = { key.replace('module.', ''): model_raw_state_dict[key] for key in model_raw_state_dict }
    
    model.load_state_dict(model_state_dict)
    
    return model