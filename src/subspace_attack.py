import torch
import random
import numpy as np

from tqdm import tqdm

from src.plots import imshow


def attack(input_batch, true_label, tau, epsilon, delta, eta_g,
           eta, victim, references, limit, verbose, compare_gradients, show_images=False):

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

    # Initialize the dropout ratio
    p = 0.05
    MAX_P = 0.5

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
    gradient_products = []
    true_gradient_norms = []
    estimated_gradient_norms = []

    for q_counter in tqdm(range(0, limit, 2)):

        # Load random reference model
        random_model = random.randint(0, len(references) - 1)
        reference_model = references[random_model]

        # Applying the corresponsing dropout ratio
        reference_model.drop = min(p, MAX_P)
        reference_model.train()

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

            query_minus = victim(x_minus)
            query_plus = victim(x_plus)

            delta_t = ((criterion(query_plus, y) -
                        criterion(query_minus, y)) / (tau * epsilon)) * u

            # Update gradient - L12
            g = g + eta_g * delta_t

        # Compute the true gradient to check the difference
        if compare_gradients:

            victim.zero_grad()
            predicted_y = victim(x_adv)
            true_loss = criterion(predicted_y, y)
            true_loss.backward()
            true_gradient = x_adv.grad.clone()

            with torch.no_grad():
                # Mesure the difference between the gradients
                true_vector = true_gradient.reshape(-1)
                est_vector = g.reshape(-1)
                gradients_product = true_vector @ est_vector / \
                    (true_vector.norm() * est_vector.norm())

                if est_vector.norm() == 0:
                    print('est_vector norm is 0!')

                # Save everything to an array
                gradient_products.append(gradients_product.item())
                true_gradient_norms.append(true_gradient.norm(2).item())
                estimated_gradient_norms.append(g.norm(2).item())

                # Update dropout ratio according to the paper
        p += 0.01

        # Update the adverserial example - L13-15
        with torch.no_grad():
            x_adv = x_adv + eta * torch.sign(g)
            x_adv = torch.max(x_adv, regmin)
            x_adv = torch.min(x_adv, regmax)
            x_adv = torch.clamp(x_adv, 0, 1)

            # Check success
            label_minus = query_minus.max(1, keepdim=True)[1].item()
            label_plus = query_plus.max(1, keepdim=True)[1].item()

        if label_minus != true_label.item() or label_plus != true_label.item():
            print(f'\nSuccess! after {q_counter + 2} queries')
            print(f'True: {true_label.item()}')
            print(f'Label minus: {label_minus}')
            print(f'Label plus: {label_plus}')

            if show_images:
                imshow(x_adv[0].cpu())

            return q_counter + 2, np.array(gradient_products), np.array(true_gradient_norms), np.array(estimated_gradient_norms)

    print(f'\nFailed!')

    return -1, np.array(gradient_products), np.array(true_gradient_norms), np.array(estimated_gradient_norms)
