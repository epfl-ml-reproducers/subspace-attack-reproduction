import random
import torch
import argparse
import os
import datetime
import time
import numpy as np

from torchvision import datasets, transforms
from tqdm import tqdm
from typing import List

from src.load_model import load_model, MODELS_DATA, ModelType
from src.load_data import load_data, DATASETS_DATA
from src.plots import imshow
from src.subspace_attack import attack

INF = float('inf')

OUTPUT_DIR = 'outputs/'


def run_experiment(victim_model_name: str, reference_model_names: List[str], dataset: str,
                   tau: float, epsilon: float, delta: float, eta: float, eta_g: float,
                   n_images: int, image_limit: int, compare_gradients: bool, verbose: bool) -> None:

    print('----- Running experiment with the following settings -----')

    print('\n----- Models information -----')
    print(f'Victim model: {victim_model_name}')
    print(f'Reference models names: {reference_model_names}')
    print(f'Dataset: {dataset}')

    print(f'\n------ Hyperparameters -----')
    print(f'tau: {tau}')
    print(f'epsilon: {epsilon}')
    print(f'delta: {delta}')
    print(f'eta: {eta}')
    print(f'eta_g: {eta_g}')

    print('\n----- General settings -----')
    print(f'Number of images: {n_images}')
    print(f'Limit of iterations per image: {image_limit}')
    print(f'Compare gradients: {compare_gradients}')
    print(f'Verbose: {verbose}')
    print(f'GPU in use: {torch.cuda.is_available()}')

    experiment_info = {
        'experiment_baseline': {
            'victim_model': victim_model_name,
            'reference_model_names': reference_model_names,
            'dataset': dataset
        },
        'hyperparameters': {
            'tau': tau,
            'epsilon': epsilon,
            'delta': delta,
            'eta': eta,
            'eta_g': eta_g
        },
        'settings': {
            'n_images': n_images,
            'image_limit': image_limit,
            'compare_gradients': compare_gradients,
            'verbose': verbose,
            'gpu': torch.cuda.is_available()
        },
        'results': {

        }

    }

    data_loader, classes = load_data(dataset)

    num_classes = len(classes)

    # Load reference models
    reference_models = list(map(lambda name: load_model(
        name, num_classes), reference_model_names))

    # Load victim model
    victim_model = load_model(victim_model_name, num_classes)

    if torch.cuda.is_available():
        reference_models = list(
            map(lambda model: model.to('cuda'), reference_models))
        victim_model = victim_model.to('cuda')

    victim_model.eval()

    counter = 0

    queries = []
    all_true_gradient_norms = []
    all_estimated_gradient_norms = []
    all_gradient_products = []

    run_time = datetime.datetime.now().replace(microsecond=0)
    tic = time.time()

    print(f'\n----- Beginning at {run_time} -----')

    for data, target in data_loader:
        print(f'\n--------------------------------------------\n')
        print(f'Target number {counter}\n')

        queries_counter, gradient_products, true_gradient_norms, estimated_gradient_norms = \
            attack(data, target, tau, epsilon, delta,
                   eta_g, eta, victim_model, reference_models,
                   image_limit, verbose, compare_gradients)

        counter += 1

        queries.append(queries_counter)
        all_gradient_products.append(gradient_products)
        all_true_gradient_norms.append(true_gradient_norms)
        all_estimated_gradient_norms.append(estimated_gradient_norms)

        if counter == n_images:
            break

    total_time = time.time() - tic

    queries_array = np.array(queries)
    failed = queries_array == -1

    print(f'\n-------------\n')
    print(f'Experiment finished:\n')
    print(f'Mean number of queries: {queries_array[~failed].mean()}')
    print(f'Median number of queries: {np.median(queries_array[~failed])}')
    print(f'Number of failed queries: {len(queries_array[failed])}')
    print(f'Total time: {total_time} s')
    print(f'\n-------------\n')

    results_path = OUTPUT_DIR
    experiment_info_filename = run_time.strftime('%Y-%m-%d.%H-%M')

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    experiment_info['results']['queries'] = queries_array

    if compare_gradients:
        experiment_info['results']['gradient_products'] = np.array(
            all_gradient_products)
        experiment_info['results']['true_gradient_norms'] = np.array(
            all_true_gradient_norms)
        experiment_info['results']['estimated_gradient_norms'] = np.array(
            all_estimated_gradient_norms)

    np.save(results_path + experiment_info_filename,
            experiment_info, allow_pickle=True)


def boolean_string(s: str) -> bool:
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':

    # Get the possible reference models
    reference_models = [
        model for model in MODELS_DATA if MODELS_DATA[model]['type'] == ModelType.REFERENCE]

    # Get the possible victim models
    victim_models = [
        model for model in MODELS_DATA if MODELS_DATA[model]['type'] == ModelType.VICTIM]

    # Get default reference models
    default_reference_models = [
        model for model in reference_models if MODELS_DATA[model]['default'] == True]

    # Get default victim model (it is just 1, thus we take the first element)
    default_victim_model = [
        model for model in victim_models if MODELS_DATA[model]['default'] == True][0]

    parser = argparse.ArgumentParser()

    # Dataset and models to be used
    parser.add_argument('-ds', '--dataset', help='The dataset to be used.',
                        default='CIFAR-10', choices=list(DATASETS_DATA.keys()))
    parser.add_argument('--reference-models', help='The reference models to be used.',
                        nargs='+', default=default_reference_models, choices=reference_models)
    parser.add_argument('--victim-model', help='The model to be attacked.',
                        default=default_victim_model, choices=victim_models)

    # Hyperparamters
    parser.add_argument('--tau', help='Bandit exploration.',  # TODO: understand what tau really is
                        default=1, type=float)
    parser.add_argument('--epsilon', help='The norm budget.',
                        default=8/255, type=float)
    parser.add_argument(
        '--delta', help='Finite difference probe', default=0.1, type=float)  # TODO: understand what delta really is
    parser.add_argument('--eta', help='Image learning rate.',
                        default=1/255, type=float)
    parser.add_argument('--eta_g', help='OCO learning rate.',
                        default=100, type=float)

    # Experiment settings
    parser.add_argument('--n-images', help='The number of images on which the attack has to be run',
                        default=1000, type=int)
    parser.add_argument('--image-limit', help='Limit of iterations to be done for each image',
                        default=10000, type=int)
    parser.add_argument('--compare-gradients', help='Whether the program should output a comparison between the estimated and the true gradients.',
                        default=False, type=boolean_string)
    parser.add_argument(
        '--verbose', help='Prints information every 50 image-iterations if true', default=True, type=boolean_string)

    args = parser.parse_args()

    victim_model = args.victim_model
    reference_models = args.reference_models
    dataset = args.dataset

    tau = args.tau
    epsilon = args.epsilon
    delta = args.delta
    eta = args.eta
    eta_g = args.eta_g

    n_images = args.n_images
    image_limit = args.image_limit
    compare_gradients = args.compare_gradients
    verbose = args.verbose

    run_experiment(victim_model, reference_models, dataset, tau, epsilon,
                   delta, eta, eta_g, n_images, image_limit, compare_gradients, verbose)
