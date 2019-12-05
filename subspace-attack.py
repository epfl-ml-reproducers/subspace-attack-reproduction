import random
import torch
from torchvision import datasets, transforms
import numpy as np

from src.import_models import load_model, MODELS_DATA, MODELS_DIRECTORY
from src.plots import imshow


def main():

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
    reference_model_names = ['vgg11_bn', 'vgg13_bn',
                             'vgg16_bn', 'vgg19_bn', 'AlexNet_bn']
    reference_models = list(map(lambda name: load_model(
        MODELS_DIRECTORY, MODELS_DATA, name, num_classes), reference_model_names))

    # Load victim model
    victim_model_name = 'gdas'
    victim_model = load_model(
        MODELS_DIRECTORY, MODELS_DATA, victim_model_name, num_classes)
    _ = victim_model.eval()

    if torch.cuda.is_available():
        reference_models = list(
            map(lambda model: model.to('cuda'), reference_models))
        victim_model = victim_model.to('cuda')

    # Hyper-parameters
    tau = 1
    epsilon = 8 / 255
    delta = 1.0
    eta_g = 100
    eta = 0.1

    counter = 0
    limit = 1000

    queries = []

    for data, target in data_loader:
        print(f'\n-------------\n')
        print(f'Target image number {counter}')

        queries_counter = attack(data, target, tau, epsilon, delta,
                                 eta_g, eta, victim_model, reference_models, verbose=True)

        counter += 1

        queries.append(queries_counter)

        if counter == limit:
            break

    results = np.array(queries)
    failed = results == -1

    print(f'\n-------------\n')
    print(f'Experiment finished:\n')
    print(f'Mean number of queries: {results[~failed].mean()}')
    print(f'Median number of queries: {np.median(results[~failed])}')
    print(f'Number of failed queries: {len(results[failed])}')
    print(f'\n-------------\n')


def attack(input_batch, true_label, tau, epsilon, delta, eta_g, eta, victim, references, limit=10000, verbose=False, show_images=False, truest_label=True):

    # Regulators
    regmin = input_batch - epsilon
    regmax = input_batch + epsilon

    # Initialize the adversarial example
    x_adv = input_batch.clone()
    x_adv.requires_grad_(True)

    victim.zero_grad()
    victim.eval()
    predicted_y = victim(x_adv)

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
    
    success = False

    while not success and q_counter < limit:

        if q_counter % 50 == 0 and verbose:
            # imshow(x_adv[0].cpu())
            print(f'Iteration number: {str(q_counter / 2)}')
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

        gradient_difference = torch.dist(g, true_gradient, float('inf'))
        gradient_differences.append(gradient_difference.item())

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

    print(gradient_differences)

    return q_counter if success else -1


if __name__ == '__main__':
    main()
