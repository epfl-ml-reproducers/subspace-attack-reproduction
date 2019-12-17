import torch
import random
import numpy as np

from tqdm import tqdm
from typing import List, Tuple

from src.plots import imshow


def attack(input_batch: torch.Tensor, criterion: torch.nn.modules.loss._Loss, true_label: int,
           epsilon: float, tau: float, delta: float, eta_g: float, eta: float, victim: torch.nn.Module,
           references: List[torch.nn.Module], limit: int, compare_gradients: bool,
           show_images: bool) -> Tuple[int, np.array, np.array, np.array]:
    """
    Runs the subspace attack on one image given as input.

    The name of the hyperparameters are the same used in [1]. The equivalents in [2]
    are also explaned for each parameter.

    The lines indicated in the comments match those in the Algorithm 1 in our reproducibility report.

    Parameters
    ----------
    input_batch: torch.Tensor
        The image to be attacked, it is a tensor, as it should come from a DataLoader.

    criterion: torch.nn.modules.loss._Loss
        The loss function to be used for the attack.

    true_label: int
        The true label assigned to the input

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

    victim: torch.nn.Module:
        The model to be attacked.

    references: List[torch.nn.Module]
        The reference models to be used.

    limit: int
        The maximum number of queries to be attempted.

    compare_gradients: bool
        Whether the real and the estimated gradients should be estimated after each loop.
        **Warning**: the use of this feature slows down the attack. It should be used just to
        check experimetally the behavior of the gradients.

    show_images: bool
        Whether each image to be attacked, and its corresponding adversarial examples should be shown.

    Returns
    -------
    n_queries: int
        The number of queries made before the success. It is -1 if the attack failed.

    gradient_products: np.array
        The cosine similarities between real and estimated gradients (empty if compare_gradients == False).

    true_gradient_norms: np.array
        The norms of the true gradients (empty if compare_gradients == False).

    estimated_gradient_norms: np.array
        The norms of the estimated gradients (empty if compare_gradients == False).

    final_model: str
        The last reference model used.

    References
    ----------
    [1] Guo, Yiwen, Ziang Yan, and Changshui Zhang. "Subspace Attack: Exploiting Promising Subspaces
        for Query-Efficient Black-box Attacks." Advances in Neural Information Processing Systems 2019.

    [2] Ilyas, Andrew, Logan Engstrom, and Aleksander Madry. "Prior convictions: Black-box adversarial
        attacks with bandits and priors." arXiv preprint arXiv:1807.07978 (2018).
    """

    # Boundaries for the perturbed input
    regmin = input_batch - epsilon
    regmax = input_batch + epsilon

    # Initialize the adversarial example - L1
    x_adv = input_batch.clone()

    # Set the label to be used (the predicted one vs the true one)
    y = true_label

    # Initialize the gradient to be estimated - L2
    g = torch.zeros_like(input_batch)

    # Initialize the dropout ratio - L3
    p = 0.05
    # Set the maximum dropout ratio to be used
    MAX_P = 0.5

    # move the input and other variables to GPU for speed, if available
    if torch.cuda.is_available():
        x_adv = x_adv.to('cuda')
        y = y.to('cuda')
        g = g.to('cuda')
        regmin = regmin.to('cuda')
        regmax = regmax.to('cuda')

    # Show original image, if required
    if show_images:
        with torch.no_grad():
            imshow(x_adv[0].cpu())

    # Create the array where we save the cosine similarity between
    # the true gradient and the estimated one
    gradient_products = []
    true_gradient_norms = []
    estimated_gradient_norms = []
    
    # Loop until the attack is successful - L4
    for q_counter in tqdm(range(0, limit, 2)):

        # Load random reference model - L5
        random_model_index = random.randint(0, len(references) - 1)
        reference_model = references[random_model_index]
        
        # Applying the corresponsing dropout ratio:
        # it takes either the current p, or MAX_P if the
        # maximum has been reached
        reference_model.drop = min(p, MAX_P)

        # Calculate the prior gradient - L6
        # --- Compute f(x_adv) using a the reference model as f
        x_adv.requires_grad_(True)
        reference_model.zero_grad()
        victim.zero_grad()
        output = reference_model(x_adv)

        # --- Compute the loss function and get the gradient of the
        # --- reference model w.r.t. x_adv
        loss = criterion(output, y)
        loss.backward()
        u = x_adv.grad

        # No gradient required for the following operations
        with torch.no_grad():
            # Calculate g_plus and g_minus - L7
            g_plus = g + tau * u
            g_minus = g - tau * u

            # Normalize g+ and g-, to get direction - L8
            g_plus_prime = g_plus / g_plus.norm()
            g_minus_prime = g_minus / g_minus.norm()

            # Compute finite difference - L9
            # --- Compute antithetic samples for finite differences
            x_plus = x_adv + delta * g_plus_prime
            x_minus = x_adv + delta * g_minus_prime

            # --- Compute f(x_adv+) and f(x_adv-), using the victim model as f
            query_minus = victim(x_minus)
            query_plus = victim(x_plus)

            # --- Compute finite difference Delta_t
            delta_t = ((criterion(query_plus, y) -
                        criterion(query_minus, y)) / (tau * epsilon)) * u

            # Update esimated gradient - L10
            g += eta_g * delta_t

            # Update the adverserial example - L11
            x_adv += eta * torch.sign(g)

            # Fit the new example into the epsilon requirements - L12
            x_adv = torch.max(x_adv, regmin)
            x_adv = torch.min(x_adv, regmax)

            # Make all the pixels in [0,1] to have a valid image - L13
            x_adv = torch.clamp(x_adv, 0, 1)

        # Update dropout ratio - L14
        p += 0.01

        # Compute the true gradient to check the difference (not in the original algorithm)
        if compare_gradients:
            # Compute f(x_adv)
            victim.zero_grad()
            reference_model.zero_grad()

            x_test = x_adv.clone()
            x_test.requires_grad_(True)
            predicted_y = victim(x_test)
            true_loss = criterion(predicted_y, y)
            true_loss.backward()
            true_gradient = x_test.grad.clone()

            with torch.no_grad():
                # Mesure the cosine similarity between the gradients
                true_vector = true_gradient.reshape(-1)
                est_vector = g.reshape(-1)
                gradients_product = (true_vector @ est_vector /
                                     (true_vector.norm() * est_vector.norm()))

                if est_vector.norm() == 0:
                    print('est_vector norm is 0!')

                # Save everything to an array
                gradient_products.append(gradients_product.item())
                true_gradient_norms.append(true_gradient.norm(2).item())
                estimated_gradient_norms.append(g.norm(2).item())

        with torch.no_grad():
            # Check if the example succeeded in being misclassified
            label_minus = query_minus.max(1, keepdim=True)[1].item()
            label_plus = query_plus.max(1, keepdim=True)[1].item()

            # If it is successful, print information about the attack, and return
            if label_minus != true_label.item() or label_plus != true_label.item():
                print(f'\nSuccess! After {q_counter + 2} queries')
                print(f'True: {true_label.item()}')
                print(f'Label minus: {label_minus}')
                print(f'Label plus: {label_plus}')
                print(f'Final model: {reference_model.type}')

                if show_images:
                    imshow(x_adv[0].cpu())

                return q_counter + 2, np.array(gradient_products), np.array(true_gradient_norms), np.array(estimated_gradient_norms), reference_model.__class__.__name__

    print(f'\nFailed! After {q_counter + 2} queries')

    return -1, np.array(gradient_products), np.array(true_gradient_norms), np.array(estimated_gradient_norms), reference_model.__class__.__name__
