import os
import requests
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection
import pandas as pd
from torchvision.models.feature_extraction import create_feature_extractor
from scipy.stats import wasserstein_distance
from sklearn.metrics import make_scorer, accuracy_score
from typing import Callable
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

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
    net.eval()
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    net.train()
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


def get_embeddings(
    net, 
    retain_loader,
    val_loader
):
    
    '''
    Feature extraction
    '''
    
    feat_extractor = create_feature_extractor(net, {'avgpool': 'feat1'})
    
    '''
    Get class weights
    '''
    
    # Retain logits
    data = np.empty((len(retain_loader.dataset), 513), dtype=object)
    idx = 0
    
    with torch.no_grad():
        for sample in retain_loader:
            # Get logits
            targets = sample[1]
            
            # Feature extraction
            inputs = sample[0]
            person_id = sample[1]
            outputs = feat_extractor(inputs.to('cuda'))['feat1']
            feats = torch.flatten(outputs, start_dim=1)
        
            for i in range(len(targets)):
                data[idx] = [targets[i].item()] + feats[i].cpu().numpy().tolist()
                idx +=1
       
    columns = ['unique_id'] + [f'feat_{i}' for i in range(512)]
    embeddings_retain_df = pd.DataFrame(data, columns=columns)
    

    # Val logits
    data = np.empty((len(val_loader.dataset), 513), dtype=object)
    idx = 0
    
    with torch.no_grad():
        for sample in val_loader:
            # Get logits
            targets = sample[1]
            
            # Feature extraction
            inputs = sample[0]
            person_id = sample[1]
            outputs = feat_extractor(inputs.to('cuda'))['feat1']
            feats = torch.flatten(outputs, start_dim=1)
        
            for i in range(len(targets)):
                data[idx] = [str(person_id[i])] + feats[i].cpu().numpy().tolist()
                idx +=1

    columns = ['unique_id'] + [f'feat_{i}' for i in range(512)]
    embeddings_val_df = pd.DataFrame(data, columns=columns)
    

    return embeddings_retain_df, embeddings_val_df

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
# Extract feature and pooling layers to create a Custom Model
class CustomResNet18(nn.Module):
    def __init__(self, original_model):
        super(CustomResNet18, self).__init__()

        # Extract features and pooling layers
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.pooling = list(original_model.children())[-2]

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = torch.squeeze(x)
        return x
    
def simple_mia(test_losses, forget_losses, n_splits=3, random_state=0):

    sample_loss = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    members = [0] * len(test_losses) + [1] * len(forget_losses)

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

def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the false positive rate (FPR)."""
    fp = np.sum(np.logical_and((y_pred == 1), (y_true == 0)))
    n = np.sum(y_true == 0)
    return fp / n


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the false negative rate (FNR)."""
    fn = np.sum(np.logical_and((y_pred == 0), (y_true == 1)))
    p = np.sum(y_true == 1)
    return fn / p

SCORING = {
    'false_positive_rate': make_scorer(false_positive_rate),
    'false_negative_rate': make_scorer(false_negative_rate)
}

def logistic_regression_attack(
        outputs_U, outputs_R, n_splits=3, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      outputs_U: numpy array of shape (N)
      outputs_R: numpy array of shape (N)
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      fpr, fnr : float * float
    """
    assert len(outputs_U) == len(outputs_R)
    
    samples = np.concatenate((outputs_R, outputs_U)).reshape((-1, 1))
    labels = np.array([0] * len(outputs_R) + [1] * len(outputs_U))

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    scores =  model_selection.cross_validate(
        attack_model, samples, labels, cv=cv, scoring=SCORING)
    
    fpr = np.mean(scores["test_false_positive_rate"])
    fnr = np.mean(scores["test_false_negative_rate"])
    
    return fpr, fnr


def svm_attack(outputs_U, outputs_R, n_splits=3, random_state=0):

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    samples = np.concatenate((outputs_R, outputs_U)).reshape((-1, 1))
    labels = np.array([0] * len(outputs_R) + [1] * len(outputs_U))

    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('svm', SVC(kernel='rbf'))
    ])

    fpr_list = []
    fnr_list = []

    for train_index, test_index in skf.split(samples, labels):
        X_train, X_test = samples[train_index], samples[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

        fpr_list.append(fpr)
        fnr_list.append(fnr)

    return np.mean(fpr_list), np.mean(fnr_list)

def tree_attack(outputs_U, outputs_R, n_splits=3, random_state=0):

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    samples = np.concatenate((outputs_R, outputs_U)).reshape((-1, 1))
    labels = np.array([0] * len(outputs_R) + [1] * len(outputs_U))

    pipeline = DecisionTreeClassifier(random_state=random_state)

    fpr_list = []
    fnr_list = []

    for train_index, test_index in skf.split(samples, labels):
        X_train, X_test = samples[train_index], samples[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

        fpr_list.append(fpr)
        fnr_list.append(fnr)

    return np.mean(fpr_list), np.mean(fnr_list)

def best_threshold_attack(
        outputs_U: np.ndarray, 
        outputs_R: np.ndarray, 
        random_state: int = 0
    ) -> tuple[list[float], list[float]]:
    """Computes FPRs and FNRs for an attack that simply splits into 
    predicted positives and predited negatives based on any possible 
    single threshold.

    Args:
      outputs_U: numpy array of shape (N)
      outputs_R: numpy array of shape (N)
    Returns:
      fpr, fnr : list[float] * list[float]
    """
    assert len(outputs_U) == len(outputs_R)
    
    samples = np.concatenate((outputs_R, outputs_U))
    labels = np.array([0] * len(outputs_R) + [1] * len(outputs_U))

    N = len(outputs_U)
    
    fprs, fnrs = [], []
    for thresh in sorted(list(samples.squeeze())):
        ypred = (samples > thresh).astype("int")
        fprs.append(false_positive_rate(labels, ypred))
        fnrs.append(false_negative_rate(labels, ypred))
    
    return fprs, fnrs

def compute_epsilon_s(fpr: list[float], fnr: list[float], delta: float) -> float:
    """Computes the privacy degree (epsilon) of a particular forget set example, 
    given the FPRs and FNRs resulting from various attacks.
    
    The smaller epsilon is, the better the unlearning is.
    
    Args:
      fpr: list[float] of length m = num attacks. The FPRs for a particular example. 
      fnr: list[float] of length m = num attacks. The FNRs for a particular example.
      delta: float
    Returns:
      epsilon: float corresponding to the privacy degree of the particular example.
    """
    assert len(fpr) == len(fnr)
    
    per_attack_epsilon = [0.]
    for fpr_i, fnr_i in zip(fpr, fnr):
        if fpr_i == 0 and fnr_i == 0:
            per_attack_epsilon.append(np.inf)
        elif fpr_i == 0 or fnr_i == 0:
            pass # discard attack
        else:
            with np.errstate(invalid='ignore'):
                epsilon1 = np.log(1. - delta - fpr_i) - np.log(fnr_i)
                epsilon2 = np.log(1. - delta - fnr_i) - np.log(fpr_i)
            if np.isnan(epsilon1) and np.isnan(epsilon2):
                per_attack_epsilon.append(np.inf)
            else:
                per_attack_epsilon.append(np.nanmax([epsilon1, epsilon2]))
            
    return np.nanmax(per_attack_epsilon)


def bin_index_fn(
        epsilons: np.ndarray, 
        bin_width: float = 0.5, 
        B: int = 13
        ) -> np.ndarray:
    """The bin index function."""
    bins = np.arange(0, B) * bin_width
    return np.digitize(epsilons, bins)


def F(epsilons: np.ndarray) -> float:
    """Computes the forgetting quality given the privacy degrees 
    of the forget set examples.
    """
    ns = bin_index_fn(epsilons)
    hs = 2. / 2 ** ns
    return np.mean(hs)

def forgetting_quality(
        unlearn_losses: list, # (N, S)
        original_losses: list, # (N, S)
        attacks: list[Callable] = [logistic_regression_attack],
        delta: float = 0.01
    ):
    """
    Both `outputs_U` and `outputs_R` are of numpy arrays of ndim 2:
    * 1st dimension coresponds to the number of samples obtained from the 
      distribution of each model (N=512 in the case of the competition's leaderboard) 
    * 2nd dimension corresponds to the number of samples in the forget set (S).
    """
    
    outputs_U = np.array(unlearn_losses)#.reshape(-1,600)
    outputs_R = np.array(original_losses)#.reshape(-1,600)

    # N = number of model samples
    # S = number of forget samples
    N, S = outputs_U.shape
    
    assert outputs_U.shape == outputs_R.shape, \
        "unlearn and retrain outputs need to be of the same shape"
    
    epsilons = []
    pbar = tqdm(range(S))
    for sample_id in pbar:
        pbar.set_description("Computing F...")
        
        sample_fprs, sample_fnrs = [], []
        
        for attack in attacks: 
            uls = outputs_U[sample_id, :]
            rls = outputs_R[sample_id, :]
            
            fpr, fnr = attack(uls, rls)
            
            if isinstance(fpr, list):
                sample_fprs.extend(fpr)
                sample_fnrs.extend(fnr)
            else:
                sample_fprs.append(fpr)
                sample_fnrs.append(fnr)
        
        sample_epsilon = compute_epsilon_s(sample_fprs, sample_fnrs, delta=delta)
        epsilons.append(sample_epsilon)
        
    return F(np.array(epsilons))
