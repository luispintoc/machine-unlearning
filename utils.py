import os
import requests
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection
from scipy.stats import wasserstein_distance

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import prune
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18

# manual random seed is used for dataset partitioning
# to ensure reproducible results across runs
SEED = 42
RNG = torch.Generator().manual_seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


def accuracy(net, loader):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total


def unstructure_prune(model, pruning_amount=0.2, global_pruning=False, random_init=False, only_fc=False):

    parameters_to_prune = []
    if global_pruning:
        for name, module in model.named_modules():
            if only_fc:
                if isinstance(module, torch.nn.Linear):
                    parameters_to_prune.append((module, 'weight'))
            else:
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    parameters_to_prune.append((module, 'weight'))

        #Global pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_amount
        )

    else:
         for name, module in model.named_modules():
            if only_fc:
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=pruning_amount)
                    parameters_to_prune.append((module, 'weight'))
            else:
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=pruning_amount)
                    parameters_to_prune.append((module, 'weight'))
                

    # Randomly re-initialize pruned weights while preserving the mask
    for module, param_name in parameters_to_prune:
        if random_init:
            mask = getattr(module, f"{param_name}_mask")  # Get the binary mask used for pruning
            init_weights = getattr(module, param_name)  # Get the current weights
            # Randomly initialize new weights
            new_weights = torch.randn_like(init_weights)
            # Apply the pruning mask to keep the pruned weights zero
            new_weights = new_weights * mask
            # Assign the new weights
            setattr(module, param_name, torch.nn.Parameter(new_weights))
        # Make the pruning permanent by removing the mask
        prune.remove(module, param_name)


def prune_test(model, pruning_amount=0.2, only_fc=False):

    parameters_to_prune = []
    for name, module in model.named_modules():
        if only_fc:
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        else:
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))

    # Global pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_amount
    )

    # Make the pruning permanent
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

def plot_teacher_student_outputs(teacher_logits, student_logits):
    teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=0).cpu().numpy()
    student_probs = torch.nn.functional.softmax(student_logits, dim=0).cpu().numpy()
    plt.plot(teacher_probs, 'ko', label='teacher')
    plt.plot(student_probs, 'ro', label='student')
    plt.legend()
    plt.yscale('log')
    plt.show()


def compute_losses(net, loader):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )


def calc_mia_acc(forget_loss, test_loss):
    # make sure we have a balanced dataset for the MIA
    assert len(test_loss) == len(forget_loss)

    ft_samples_mia = np.concatenate((test_loss, forget_loss)).reshape((-1, 1))
    labels_mia = [0] * len(test_loss) + [1] * len(forget_loss)

    ft_mia_scores = simple_mia(ft_samples_mia, labels_mia)

    return ft_mia_scores.mean()


def get_all_metrics(test_losses, student_model, retain_loader, forget_loader, val_loader, test_loader):

    
    print(f"Retain set accuracy: {100.0 * accuracy(student_model, retain_loader):0.1f}%")
    print(f"Forget set accuracy: {100.0 * accuracy(student_model, forget_loader):0.1f}%")
    print(f"Val set accuracy: {100.0 * accuracy(student_model, val_loader):0.1f}%")
    print(f"Test set accuracy: {100.0 * accuracy(student_model, test_loader):0.1f}%")

    ft_forget_losses = compute_losses(student_model, forget_loader)
    # ft_test_losses = compute_losses(model, test_loader)

    ft_mia_scores = calc_mia_acc(ft_forget_losses, test_losses)

    print(
        f"The MIA has an accuracy of {ft_mia_scores.mean():.3f} on forgotten vs unseen images"
    )

    print(f'Earth movers distance = {wasserstein_distance(ft_forget_losses, test_losses):.3f}')

    return ft_forget_losses, test_losses, ft_mia_scores

def calculate_kl_loss(student_logits, teacher_logits, T=2.0, forget_T=2.0, forget_flag=False):
    
    teacher_logits = teacher_logits/T

    if forget_flag:
        teacher_logits = teacher_logits/forget_T
        teacher_logits = teacher_logits + 0.05*torch.rand(teacher_logits.shape).to('cuda')

    # Calculate soft labels from teacher
    teacher_probs = F.softmax(teacher_logits, dim=1)

    # Compute distillation loss
    student_log_probs = F.log_softmax(student_logits/T, dim=1)
    distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)

    return distillation_loss

# Function to update learning rate
def adjust_learning_rate(optimizer, current_batch, total_batches, initial_lr):
    """Sets the learning rate for warmup over total_batches"""
    lr = initial_lr * (current_batch / total_batches)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr