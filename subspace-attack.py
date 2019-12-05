import random
import torch
import argparse
import os
import datetime
import time
from torchvision import datasets, transforms
import numpy as np

from src.import_models import load_model, MODELS_DATA, MODELS_DIRECTORY
from src.plots import imshow

DATASETS = ['CIFAR-10']
REFERENCE_MODELS = ['vgg11_bn', 'vgg13_bn',
                    'vgg16_bn', 'vgg19_bn', 'AlexNet_bn']
DEFAULT_REFERENCE_MODELS = ['vgg11_bn', 'vgg13_bn',
                            'vgg16_bn', 'vgg19_bn', 'AlexNet_bn']
VICTIM_MODELS = ['gdas']
DEFAULT_VICTIM_MODEL = 'gdas'
INF = float('inf')

OUTPUT_DIR = 'outputs/'

def main(victim_model_name, reference_model_names, dataset, tau, epsilon,
         delta, eta, eta_g, n_images, image_limit, compare_gradients, verbose):

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

    # Load CIFAR10 dataset
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    data = datasets.CIFAR10(root='./data', train=True,
                            download=True, transform=preprocess)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    num_classes = len(classes)
    # Load reference models
    reference_models = list(map(lambda name: load_model(
        MODELS_DIRECTORY, MODELS_DATA, name, num_classes), reference_model_names))

    # Load victim model
    victim_model = load_model(
        MODELS_DIRECTORY, MODELS_DATA, victim_model_name, num_classes)
    _ = victim_model.eval()

    if torch.cuda.is_available():
        reference_models = list(
            map(lambda model: model.to('cuda'), reference_models))
        victim_model = victim_model.to('cuda')

    counter = 0

    queries = []

    run_time = datetime.datetime.now().replace(microsecond=0)
    tic = time.time()
    
    print(f'\n----- Beginning at {run_time} -----')

    for data, target in data_loader:
        print(f'\n--------------------------------------------\n')
        print(f'Target image number {counter}')

        queries_counter, gradient_differences, true_gradient_norms, estimated_gradient_norms = \
            attack(data, target, tau, epsilon, delta,
                   eta_g, eta, victim_model, reference_models,
                   image_limit, verbose)

        counter += 1

        queries.append(queries_counter)

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

    run_subfolder = run_time.strftime('%Y-%m-%d.%H-%M')

    results_path = OUTPUT_DIR + '/' + run_subfolder + '/'
    

    os.makedirs(results_path)

    np.save(results_path + 'queries.npy', queries_array)
    np.save(results_path + 'gradient_differences.npy', gradient_differences)
    np.save(results_path + 'true_gradient_norms.npy', true_gradient_norms)
    np.save(results_path + 'estimated_gradient_norms.npy', estimated_gradient_norms)


def attack(input_batch, true_label, tau, epsilon, delta, eta_g, eta, victim, references, limit, verbose, show_images=False):

    # Regulators
    regmin = input_batch - epsilon
    regmax = input_batch + epsilon

    # Initialize the adversarial example
    x_adv = input_batch.clone()

    # Set the label to be used (the predicted one vs the true one)
    y = true_label

    # Set CrossEntropy loss
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize the gradient to be estimated
    g = torch.zeros_like(input_batch)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        x_adv = x_adv.to('cuda')
        y = y.to('cuda')
        g = g.to('cuda')
        regmin = regmin.to('cuda')
        regmax = regmax.to('cuda')

    # initialize quiries counter
    q_counter = 0

    if show_images:
        with torch.no_grad():
            imshow(x_adv[0].cpu())

    # Create the array where we save the difference between
    # the true gradient and the estimated one
    gradient_differences = []
    true_gradient_norms = []
    estimated_gradient_norms = []

    success = False

    while not success and q_counter < limit:

        if q_counter % 50 == 0 and verbose:
            # imshow(x_adv[0].cpu())
            print(f'Iteration number: {str(q_counter // 2)}')
            print(f'{str(q_counter)} queries have been made')

        # Load random reference model
        random_model = random.randint(0, len(references) - 1)
        reference_model = references[random_model]
        reference_model.eval()

        # calculate the prior gradient - L8
        x_adv.requires_grad_(True)
        reference_model.zero_grad()
        output = reference_model(x_adv)

        loss = criterion(output, y)

        loss.backward()

        u = x_adv.grad

        # Calculate delta - L11
        with torch.no_grad():
            # Calculate g_plus and g_minus - L9-10
            g_plus = g + tau * u
            g_minus = g - tau * u

            g_minus = g_minus / g_minus.norm()
            g_plus = g_plus / g_plus.norm()
            x_plus = x_adv + delta * g_plus
            x_minus = x_adv + delta * g_minus

            victim.eval()
            query_minus = victim(x_minus)

            victim.eval()
            query_plus = victim(x_plus)

        delta_t = ((criterion(query_plus, y) -
                    criterion(query_minus, y)) / (tau * epsilon)) * u

        # Update gradient - L12
        g = g + eta_g * delta_t

        # Compute the true gradient to check the difference
        victim.zero_grad()
        victim.eval()
        predicted_y = victim(x_adv)
        true_loss = criterion(predicted_y, y)
        true_loss.backward()
        true_gradient = x_adv.grad.clone()

        gradient_difference = torch.dist(
            g, true_gradient, 2) / (true_gradient.norm(2) + g.norm(2)) * 2
        gradient_differences.append(gradient_difference.item())
        true_gradient_norms.append(true_gradient.norm(2).item())
        estimated_gradient_norms.append(g.norm(2).item())

        # Update the adverserial example - L13-15
        with torch.no_grad():
            x_adv = x_adv + eta * torch.sign(g)
            x_adv = torch.max(x_adv, regmin)
            x_adv = torch.min(x_adv, regmax)
            x_adv = torch.clamp(x_adv, 0, 1)

            # Check success
            label_minus = query_minus.max(1, keepdim=True)[1].item()
            label_plus = query_plus.max(1, keepdim=True)[1].item()

        q_counter += 2

        if label_minus != true_label.item() or label_plus != true_label.item():
            print('Success! after {} queries'.format(q_counter))
            print("True: {}".format(true_label.item()))
            print("Label minus: {}".format(label_minus))
            print("Label plus: {}".format(label_plus))
            if show_images:
                imshow(x_adv[0].cpu())
            success = True

    return q_counter if success else -1, gradient_differences, true_gradient_norms, estimated_gradient_norms


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Dataset and models to be used
    parser.add_argument('-ds', '--dataset', help='The dataset to be used.',
                        default='CIFAR-10', choices=DATASETS)
    parser.add_argument('--reference-models', help='The reference models to be used.',
                        nargs='+', default=REFERENCE_MODELS, choices=REFERENCE_MODELS)
    parser.add_argument('--victim-model', help='The model to be attacked.',
                        default=DEFAULT_VICTIM_MODEL, choices=VICTIM_MODELS)

    # Hyperparamters
    parser.add_argument('--tau', help='Bandit exploration.',
                        default=1, type=int)
    parser.add_argument('--epsilon', help='The norm budget.',  # TODO: understand what tau really is
                        default=8/255, type=float)
    parser.add_argument(
        '--delta', help='Finite difference probe', default=1, type=float)
    parser.add_argument('--eta', help='Image learning rate.',
                        default=0.01, type=float)
    parser.add_argument('--eta_g', help='OCO learning rate.',
                        default=100, type=float)

    # Experiment settings
    parser.add_argument('--n-images', help='The number of images on which the attack has to be run',
                        default=1000, type=int)
    parser.add_argument('--image-limit', help='Limit of iterations to be done for each image',
                        default=10000, type=int)
    parser.add_argument('--compare-gradients', help='Whether the program should output a comparison between the estimated and the true gradients.',
                        default=True, type=bool)
    parser.add_argument(
        '--verbose', help='Prints information every 50 image-iterations if true', default=True, type=bool)

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

    main(victim_model, reference_models, dataset, tau, epsilon,
         delta, eta, eta_g, n_images, image_limit, compare_gradients, verbose)
