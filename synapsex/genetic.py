"""Simple genetic algorithm for ANN hyper-parameter search.

This module provides a light‑weight genetic algorithm that searches over a
small space of ``HyperParameters`` in order to maximise the F1 score of a
``PyTorchANN`` model.  It is intentionally minimal and dependency free so it
can run in constrained environments.

The search space currently covers the dropout rate and learning rate, two
hyper‑parameters that strongly influence classification metrics such as
accuracy, recall and precision.  Each individual in the population represents a
unique combination of these parameters.  Individuals are evaluated by training a
model for a few epochs and scoring it on a held‑out validation set.  The best
performer is returned as the suggested configuration.

The algorithm is purposely small so that projects can quickly experiment with
genetic optimisation without pulling in heavy third‑party libraries.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import List

import torch

from .config import HyperParameters, hp


def _random_hyperparameters() -> HyperParameters:
    """Create a random set of hyper-parameters within sensible bounds."""

    return HyperParameters(
        image_size=hp.image_size,
        dropout=random.uniform(0.1, 0.5),
        learning_rate=10 ** random.uniform(-4, -2),
        epochs=max(2, hp.epochs // 2),
        batch_size=hp.batch_size,
        mc_dropout_passes=hp.mc_dropout_passes,
    )


def _evaluate(hparams: HyperParameters, X_train: torch.Tensor, y_train: torch.Tensor,
              X_val: torch.Tensor, y_val: torch.Tensor) -> float:
    """Train a model with ``hparams`` and return the validation F1 score."""

    # Local import to avoid a circular dependency at module load time.
    from .neural import PyTorchANN

    ann = PyTorchANN(hparams)
    ann.train(X_train, y_train)
    metrics = ann.evaluate(X_val, y_val)
    return metrics["f1"]


def genetic_search(X: torch.Tensor, y: torch.Tensor, generations: int = 5,
                   population_size: int = 8) -> HyperParameters:
    """Run a tiny genetic algorithm and return the best ``HyperParameters``.

    Parameters
    ----------
    X, y:
        Training data tensors.  They are internally split into a training and
        validation subset.
    generations:
        Number of evolutionary generations to perform.
    population_size:
        Number of individuals in each generation's population.
    """

    # Create a deterministic train/validation split so that all individuals are
    # evaluated on the same data partition.
    indices = torch.randperm(len(X))
    split = int(0.8 * len(X))
    train_idx = indices[:split]
    val_idx = indices[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    population: List[HyperParameters] = [_random_hyperparameters() for _ in range(population_size)]

    for _ in range(generations):
        scores = [_evaluate(ind, X_train, y_train, X_val, y_val) for ind in population]
        # Sort individuals by fitness (descending F1).
        population = [pop for _, pop in sorted(zip(scores, population), key=lambda p: p[0], reverse=True)]
        # Elitism: carry over the two best individuals unchanged.
        new_population: List[HyperParameters] = population[:2]

        # Breed new individuals until the population is replenished.
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:4], 2)
            child_dropout = (parent1.dropout + parent2.dropout) / 2
            child_lr = (parent1.learning_rate + parent2.learning_rate) / 2

            # Mutation – small random perturbations.
            child_dropout += random.uniform(-0.05, 0.05)
            child_dropout = min(max(child_dropout, 0.05), 0.6)
            child_lr *= 10 ** random.uniform(-0.5, 0.5)
            child_lr = min(max(child_lr, 1e-4), 1e-2)

            child = replace(parent1, dropout=child_dropout, learning_rate=child_lr)
            new_population.append(child)

        population = new_population

    # Final evaluation to select the best individual from the last generation.
    scores = [_evaluate(ind, X_train, y_train, X_val, y_val) for ind in population]
    best = population[scores.index(max(scores))]
    return best

