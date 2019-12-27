# Subspace Attack Reproduction

## Motivation

Attempt to reproduce the NeurIPS 2019 paper [Subspace Attack: Exploiting Promising Subspaces for Query-Efficient Black-box Attacks](https://papers.nips.cc/paper/8638-subspace-attack-exploiting-promising-subspaces-for-query-efficient-black-box-attacks).

The original code of the paper can be found [here](https://github.com/ZiangYan/subspace-attack.pytorch). We are trying to reproduce the attack to GDAS and WRN model trained on CIFAR-10 dataset, without using and looking at the original code.

This project is done as project for the [CS-433 Machine Learning Course](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) at [EPFL](https://epfl.ch/en), and as part of the [NeurIPS 2019 Reproducibility Challenge](https://reproducibility-challenge.github.io/neurips2019/).

## Usage

We make use of some pretrained models, that can be downloaded [here](https://drive.google.com/file/d/1TA-UWYVDkCkNPOy1INjUU9321s-HA6RF/view?usp=sharing). They are a subset of the [models](https://drive.google.com/file/d/1aXTmN2AyNLdZ8zOeyLzpVbRHZRZD0fW0/view?usp=sharing) provided with the code of the original paper. They need to be unzipped and put in the `./pretrained` folder, in the root directory of the repo.

The dataset ([CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)) is automatically downloaded via `torchvision.datasets` when first running the experiment, and will be saved in the `data/` folder (more info [here](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar)).

The paper is implemented and tested using Python 3.7. Dependencies are listed in [requirements.txt](requirements.txt).

For the moment, it is possible to run the experiment using [VGG nets](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) and [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) as reference models and [GDAS](https://arxiv.org/pdf/1910.04465.pdf), [WRN](https://arxiv.org/pdf/1605.07146.pdf) and [PyramidNet](https://arxiv.org/pdf/1610.02915.pdf) as victim models.

In order to test our implemenation, install the dependencies with `pip3 install --user --requirement requirements.txt`, and run the following command:

```bash
python run.py
```

This will run the experiment on line 5 of table II of our report, with the following settings:

- Reference models: AlexNet+VGGs
- Victim model: GDAS
- Number of images: 1000
- Maximum queries per image: 10000
- 0 seed
  
And hyperparameters:

- eta_g = 0.1
- eta = 1/255
- delta = 0.1
- tau = 1.0
- epsilon = 8/255

N.B.: it takes 7 hours 45 minutes to run on a Google Cloud Platform n1-highmem-8 virtual machine, with 8 vCPU, 52 GB memory and an Nvidia Tesla T4.

Moreover, the following settings can be used to customize the experiment:

```bash
usage: run.py [-h] [-ds {Dataset.CIFAR_10}]
                     [--reference-models {vgg11_bn,vgg13_bn,vgg16_bn,vgg19_bn,AlexNet_bn} [{vgg11_bn,vgg13_bn,vgg16_bn,vgg19_bn,AlexNet_bn} ...]]
                     [--victim-model {gdas,wrn,pyramidnet}]
                     [--loss {ExperimentLoss.CROSS_ENTROPY,ExperimentLoss.NEG_LL}]
                     [--tau TAU] [--epsilon EPSILON] [--delta DELTA]
                     [--eta ETA] [--eta_g ETA_G] [--n-images N_IMAGES]
                     [--image-limit IMAGE_LIMIT]
                     [--compare-gradients COMPARE_GRADIENTS]
                     [--check-success CHECK_SUCCESS]
                     [--show-images SHOW_IMAGES] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  -ds {Dataset.CIFAR_10}, --dataset {Dataset.CIFAR_10}
                        The dataset to be used.
  --reference-models {vgg11_bn,vgg13_bn,vgg16_bn,vgg19_bn,AlexNet_bn} [{vgg11_bn,vgg13_bn,vgg16_bn,vgg19_bn,AlexNet_bn} ...]
                        The reference models to be used.
  --victim-model {gdas,wrn,pyramidnet}
                        The model to be attacked.
  --loss {ExperimentLoss.CROSS_ENTROPY,ExperimentLoss.NEG_LL}
                        The loss function to be used
  --tau TAU             Bandit exploration.
  --epsilon EPSILON     The norm budget.
  --delta DELTA         Finite difference probe.
  --eta ETA             Image learning rate.
  --eta_g ETA_G         OCO learning rate.
  --n-images N_IMAGES   The number of images on which the attack has to be run
  --image-limit IMAGE_LIMIT
                        Limit of iterations to be done for each image
  --compare-gradients COMPARE_GRADIENTS
                        Whether the program should output a comparison between
                        the estimated and the true gradients.
  --check-success CHECK_SUCCESS
                        Whether the attack on each image should stop if it has
                        been successful.
  --show-images SHOW_IMAGES
                        Whether each image to be attacked, and its
                        corresponding adversarial examples should be shown
  --seed SEED           The random seed with which the experiment should be
                        run, to be used for reproducibility purposes.
```

In order to run an experiment on 100 images in which the loss of the true model and the cosine similarity between the estimated and true gradient, for all 5000 iterations per image, regardless of the success of the attack (i.e. the one used for figures 1 and 2 of our report), you should run

```bash
python3 run.py --check-success=False --n-images=100 --compare-gradients=True
```

N.B.: it takes around 20 hours to run the experiment on the aforementioned machine.

The experiment results are saved in the `outputs/` folder, in a file named `YYYY-MM-DD.HH-MM.npy` a dictionary exported with `numpy.save()`. The format of the dictionary is:

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
        'true_losses': # The true losses each iteration
        'common_signs': # The percentages of common signs between true and est gradients
        'subs_common_signs': # The percentages of common signs between subsequent gradients
}
```

The file can be imported in Python using `np.load(output_path, allow_pickle=True).item()`.

## Project structure

The repository is structured in the following way:

```bash
.
├── black-box_attack_reproduce.ipynb
├── data # Should contain the dataset used
├── experiment.py # Contains the experiment
├── img # Contains images used in notebooks
│   └── algo1.png
├── LICENSE
├── notebooks # Contains some notebooks used to analyze the experiments
│   └── experiment_analysis.ipynb
├── outputs # Contains the .npy files obtained in the reported experiments
├── pretrained # Should contain the pretrained models (.pth files)
├── README.md # This file :)
├── requirements.txt # Contains information about dependencies
└── src
    ├── helpers.py # Some helper functions
    ├── __init__.py
    ├── load_data.py # Some functions used to load the dataset
    ├── load_loss.py # Some functions used to load the loss function
    ├── load_model.py # Some functions to load pretrained models
    ├── models # Contains the classes of the models (not made by us, link to original repo above)
    ├── plots.py # A function to plot images
    └── subspace_attack.py # The very attack, the core of the repo
```
