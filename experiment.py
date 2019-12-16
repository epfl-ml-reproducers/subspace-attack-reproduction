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

from src.helpers import boolean_string
from src.load_model import load_model, MODELS_DATA, ModelType
from src.load_data import Dataset, DEFAULT_DATASET, load_data
from src.load_loss import ExperimentLoss, DEFAULT_LOSS, load_loss
from src.plots import imshow
from src.subspace_attack import attack

INF = float('inf')

OUTPUT_DIR = 'outputs/'


def run_experiment(victim_model_name: str, reference_model_names: List[str], dataset: str,
                   loss: str, epsilon: float, tau: float, delta: float, eta: float, eta_g: float,
                   n_images: int, image_limit: int, compare_gradients: bool, show_images: bool,
                   seed: int = 0) -> None:
    """
    Runs an experiment of the subspace attack on a batch of images. It outputs the results in the
    `outputs/` folder, in a file named `YYYY-MM-DD.HH-MM.npy` The output file is a dictionary
    exported with `numpy.save`. The format of the dictionary is:

    ```python
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
            'gpu': # If the GPU has been used for the experiment,
            'seed': seed
        },
        'results': {
            'queries': # The number of queries run
            'total_time' # The time it took to run the experiment
            # The following are present only if compare_gradients == True
            'gradient_products': # The cosine similarities for each image
            'true_gradient_norms': # The norms of the true gradients for each image
            'estimated_gradient_norms': # The norms of the estimated gradients for each image
    }
    ```

    The name of the hyperparameters are the same used in [1]. The equivalents in [2] are also
    explaned for each parameter.

    Parameters
    ----------
    victim_model_name: str
        The name of the model to be attacked.

    reference_model_names: int
        The list of names of the models to be used as references.

    dataset: str
        The dataset from which the examples should be generated.

    loss: str
        The name of the loss function to be used.

    epsilon: float
        The maximum perturbation allowed $\ell\infty$ norm. In [2] it has the same name.

    tau: float
        The Bandit exploration ($\delta$ in [2]).

    delta: float
        Finite difference probe (The lower $\eta$ in [2]).

    eta_g: float
        OCO learning rate (The upper $\eta$ in [2]).

    eta: float
        Image learning rate (h in [2]).

    n_images: int
        The number of images on which the attack should be run.

    limit: int
        The maximum number of queries to be attempted.

    compare_gradients: bool
        Whether the real and the estimated gradients should be estimated after each loop.
        **Warning**: the use of this feature slows down the attack. It should be used just to
        check experimetally the behavior of the gradients.

    show_images: bool
        Whether each image to be attacked, and its corresponding adversarial examples should be shown.

    seed: int
        The seed to be used to initialize pseudo-random generators. To be used for reproducibility
        purposes.

    References
    ----------
    [1] Guo, Yiwen, Ziang Yan, and Changshui Zhang. "Subspace Attack: Exploiting Promising Subspaces
        for Query-Efficient Black-box Attacks." Advances in Neural Information Processing Systems 2019.

    [2] Ilyas, Andrew, Logan Engstrom, and Aleksander Madry. "Prior convictions: Black-box adversarial
        attacks with bandits and priors." arXiv preprint arXiv:1807.07978 (2018).
    """
    # Fix the seeds for reproducibility purposes
    torch.manual_seed(seed)
    random.seed(seed)

    # Print introductory message
    print('----- Running experiment with the following settings -----')
    print('\n----- Models information -----')
    print(f'Victim model: {victim_model_name}')
    print(f'Reference models names: {reference_model_names}')
    print(f'Dataset: {dataset.value}')
    print(f'Loss function: {loss.value}')

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
    print(f'Show images: {show_images}')
    print(f'Seed: {seed}')
    print(f'GPU in use: {torch.cuda.is_available()}')

    # Save experiment initial information
    experiment_info = {
        'experiment_baseline': {
            'victim_model': victim_model_name,
            'reference_model_names': reference_model_names,
            'dataset': dataset.value,
            'loss': loss.value
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
            'gpu': torch.cuda.is_available(),
            'seed': seed
        },
        'results': {
            # Initialize dict entry to save results later.
        }

    }

    # Load data using required dataset
    data_loader, classes = load_data(dataset, True)
    num_classes = len(classes)

    # Load reference models
    reference_models = list(map(lambda name: load_model(
        name, num_classes), reference_model_names))

    # Load victim model
    victim_model = load_model(victim_model_name, num_classes)

    # Move models to CUDA, if available
    if torch.cuda.is_available():
        reference_models = list(
            map(lambda model: model.to('cuda'), reference_models))
        victim_model = victim_model.to('cuda')

    # Get loss function
    criterion = load_loss(loss)

    # Set victim model to `eval()` mode to avoid dropout and batch normalization
    victim_model.eval()

    # Initialize images counter
    counter = 0

    # Initalize the arrays to save results
    queries = []
    final_models = []
    all_true_gradient_norms = []
    all_estimated_gradient_norms = []
    all_gradient_products = []

    # Initialize timing information
    run_time = datetime.datetime.now().replace(microsecond=0)
    tic = time.time()

    print(f'\n----- Beginning at {run_time} -----')

    # Loop over the dataset
    for data, target in data_loader:
        print(f'\n--------------------------------------------\n')
        print(f'Target number {counter}\n')

        # Attack the image
        queries_counter, gradient_products, true_gradient_norms, estimated_gradient_norms, final_model = \
            attack(data, criterion, target, tau, epsilon, delta,
                   eta_g, eta, victim_model, reference_models,
                   image_limit, compare_gradients, show_images)

        counter += 1

        # Save the results of the attack
        queries.append(queries_counter)
        final_models.append(final_model)
        all_gradient_products.append(gradient_products)
        all_true_gradient_norms.append(true_gradient_norms)
        all_estimated_gradient_norms.append(estimated_gradient_norms)

        # Stop if all the required images have been attacked
        if counter == n_images:
            break

    # Save the total time
    total_time = time.time() - tic

    # Make an np.array aout of the queries array to print some stats
    queries_array = np.array(queries)
    failed = queries_array == -1

    print(f'\n-------------\n')
    print(f'Experiment finished:\n')
    print(f'Mean number of queries: {queries_array[~failed].mean()}')
    print(f'Median number of queries: {np.median(queries_array[~failed])}')
    print(f'Number of failed queries: {len(queries_array[failed])}')
    print(f'Total time: {total_time} s')
    print(f'\n-------------\n')

    # Save experiment run information
    experiment_info['results']['queries'] = queries_array
    experiment_info['results']['total_time'] = total_time
    experiment_info['results']['final_model'] = final_models

    # Save gradients information, if required by experiment run
    if compare_gradients:
        experiment_info['results']['gradient_products'] = np.array(
            all_gradient_products)
        experiment_info['results']['true_gradient_norms'] = np.array(
            all_true_gradient_norms)
        experiment_info['results']['estimated_gradient_norms'] = np.array(
            all_estimated_gradient_norms)

    # Take care of results output folder
    results_path = OUTPUT_DIR
    experiment_info_filename = run_time.strftime('%Y-%m-%d.%H-%M')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Save results
    np.save(results_path + experiment_info_filename,
            experiment_info, allow_pickle=True)


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

    # Dataset, models and loss to be used
    parser.add_argument('-ds', '--dataset', help='The dataset to be used.',
                        default=DEFAULT_DATASET, choices=[d.value for d in Dataset])
    parser.add_argument('--reference-models', help='The reference models to be used.',
                        nargs='+', default=default_reference_models, choices=reference_models)
    parser.add_argument('--victim-model', help='The model to be attacked.',
                        default=default_victim_model, choices=victim_models)
    parser.add_argument('--loss', help='The loss function to be used', default=DEFAULT_LOSS,
                        choices=[l.value for l in ExperimentLoss], type=ExperimentLoss)

    # Hyperparamters
    parser.add_argument('--tau', help='Bandit exploration.',
                        default=1, type=float)
    parser.add_argument('--epsilon', help='The norm budget.',
                        default=8/255, type=float)
    parser.add_argument('--delta', help='Finite difference probe',
                        default=0.1, type=float)
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
    parser.add_argument('--show-images', help='Whether each image to be attacked, and its corresponding adversarial examples should be shown',
                        default=False, type=boolean_string)
    parser.add_argument('--seed', help='The random seed with which the experiment should be run, to be used for reproducibility purposes.',
                        default=0, type=int)
    args = parser.parse_args()

    victim_model = args.victim_model
    reference_models = args.reference_models
    dataset = args.dataset
    loss = args.loss

    tau = args.tau
    epsilon = args.epsilon
    delta = args.delta
    eta = args.eta
    eta_g = args.eta_g

    n_images = args.n_images
    image_limit = args.image_limit
    compare_gradients = args.compare_gradients
    show_images = args.show_images
    seed = args.seed

    run_experiment(victim_model, reference_models, dataset, loss, tau, epsilon,
                   delta, eta, eta_g, n_images, image_limit, compare_gradients, show_images, seed)
