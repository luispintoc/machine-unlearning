"""
This file is used for the Selective Synaptic Dampening method
Strategy files use the methods from here
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, dataset
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import copy
import os
import gc
import pdb
import math
import shutil
from torch.utils.data import DataLoader
# import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from typing import Dict, List

###############################################
# Clean implementation
###############################################


class ParameterPerturber:
    def __init__(
        self,
        model,
        opt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameters=None,
    ):
        self.model = model
        self.opt = opt
        self.device = device
        self.alpha = None
        self.xmin = None

        # print(parameters)
        self.lower_bound = parameters["lower_bound"]
        self.exponent = parameters["exponent"]
        self.magnitude_diff = parameters["magnitude_diff"]  # unused
        self.min_layer = parameters["min_layer"]
        self.max_layer = parameters["max_layer"]
        self.forget_threshold = parameters["forget_threshold"]
        self.dampening_constant = parameters["dampening_constant"]
        self.selection_weighting = parameters["selection_weighting"]
        self.alpha = parameters["alpha"]
        self.beta = parameters["beta"]
        self.theta = parameters["theta"]

    def get_layer_num(self, layer_name: str) -> int:
        layer_id = layer_name.split(".")[1]
        if layer_id.isnumeric():
            return int(layer_id)
        else:
            return -1

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter valuse
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params
        """
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    def fulllike_params_dict(
        self, model: torch.nn, fill_value, as_tensor: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict like named_parameters(), with parameter values replaced with fill_value

        Parameters:
        model (torch.nn): model to get param dict from
        fill_value: value to fill dict with
        Returns:
        dict(str,torch.Tensor): dict of named_parameters() with filled in values
        """

        def full_like_tensor(fillval, shape: list) -> list:
            """
            recursively builds nd list of shape shape, filled with fillval
            Parameters:
            fillval: value to fill matrix with
            shape: shape of target tensor
            Returns:
            list of shape shape, filled with fillval at each index
            """
            if len(shape) > 1:
                fillval = full_like_tensor(fillval, shape[1:])
            tmp = [fillval for _ in range(shape[0])]
            return tmp

        dictionary = {}

        for n, p in model.named_parameters():
            _p = (
                torch.tensor(full_like_tensor(fill_value, p.shape), device=self.device)
                if as_tensor
                else full_like_tensor(fill_value, p.shape)
            )
            dictionary[n] = _p
        return dictionary

    def subsample_dataset(self, dataset: dataset, sample_perc: float) -> Subset:
        """
        Take a subset of the dataset

        Parameters:
        dataset (dataset): dataset to be subsampled
        sample_perc (float): percentage of dataset to sample. range(0,1)
        Returns:
        Subset (float): requested subset of the dataset
        """
        sample_idxs = np.arange(0, len(dataset), step=int((1 / sample_perc)))
        return Subset(dataset, sample_idxs)

    def split_dataset_by_class(self, dataset: dataset) -> List[Subset]:
        """
        Split dataset into list of subsets
            each idx corresponds to samples from that class

        Parameters:
        dataset (dataset): dataset to be split
        Returns:
        subsets (List[Subset]): list of subsets of the dataset,
            each containing only the samples belonging to that class
        """
        n_classes = len(set([target for _, target in dataset]))
        subset_idxs = [[] for _ in range(n_classes)]
        for idx, (x, y) in enumerate(dataset):
            subset_idxs[y].append(idx)

        return [Subset(dataset, subset_idxs[idx]) for idx in range(n_classes)]

    def calc_gradient_importance(self, dataloader: DataLoader, debug=False) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        criterion = nn.CrossEntropyLoss()
        importances = self.zerolike_params_dict(self.model)
        for batch in dataloader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            out = self.model(x)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances


    def calc_occlusion_sensitivity(self, dataloader: DataLoader, debug=False) -> Dict[str, torch.Tensor]:
        criterion = nn.CrossEntropyLoss()
        importances = {name: torch.tensor(0.0) for name, _ in self.model.named_modules() if isinstance(_, (nn.Conv2d, nn.Linear))}
        original_model = copy.deepcopy(self.model)

        self.model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass to get the baseline output
                baseline_out = self.model(x)
                baseline_loss = criterion(baseline_out, y).item()
                if debug:
                    print("Baseline Loss:", baseline_loss)

                for name, imp in importances.items():
                    def hook(module, input, output):
                        if debug:
                            print("Hook executed")
                        output *= 0

                    module = dict(self.model.named_modules()).get(name, None)
                    if module is not None:
                        if debug:
                            print(f"Module {name} is not None, registering hook.")
                        handle = module.register_forward_hook(hook)
                        
                        # Get new output and calculate loss
                        new_out = self.model(x)
                        new_loss = criterion(new_out, y).item()

                        # Remove hook
                        handle.remove()

                        # Calculate importance as the change in loss
                        if debug:
                            print("Importance before update:", imp)
                        imp += abs(new_loss - baseline_loss)
                        if debug:
                            print("Importance after update:", imp)

        # Average over mini-batch length
        for _, imp in importances.items():
            imp /= float(len(dataloader))

        return importances


    def calc_activation_maximization(self, dataloader: DataLoader, debug=False) -> Dict[str, torch.Tensor]:
        importances = self.zerolike_params_dict(self.model)
            
        self.model.eval()
            
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if debug:
                    print(f"Registering hook for module: {name}")

                imp = importances.get(name, None)
                if imp is None:
                    if debug:
                        print(f"Importance tensor not found for module: {name}")
                    continue

                def hook(module, input, output, imp=imp):  # Note the default argument to capture 'imp'
                    if imp is None:
                        if debug:
                            print("Importance tensor is None inside hook.")
                        return
                    summed_activation = output.abs().sum(dim=list(range(1, output.dim())))
                    imp += summed_activation.mean(dim=0)

                target_module = self.model._modules.get(name)
                if target_module is not None:
                    handle = target_module.register_forward_hook(hook)
                else:
                    if debug:
                        print(f"Could not register hook for module: {name}")
                    continue

                self.model(x)
                handle.remove()

        # Average over the number of batches
        for name, imp in importances.items():
            importances[name] /= float(len(dataloader))

        return importances



    def modify_weight_neuron_level(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None

        """

        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)

                # Synapse Dampening with parameter lambda
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )
                update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)


    def modify_weight_neuron_level_new(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None

        """

        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):
                # Calculate the mean and standard deviation of fimp
                mean_fimp = torch.mean(fimp)
                std_fimp = torch.std(fimp)
                gc.collect()
                # Calculate the Z-score for fimp
                z_scores_fimp = (fimp - mean_fimp) / (std_fimp + 1e-10)  # Added epsilon to avoid division by zero
                # import matplotlib.pyplot as plt
                # plt.hist(z_scores_fimp.cpu().flatten())
                # plt.show()
        
                # Define an outlier threshold for Z-score
                outlier_threshold = 2  # You can adjust this threshold as needed
        
                # Find where fimp is greater than oimp and is an outlier based on Z-score
                outlier_and_greater = (z_scores_fimp > outlier_threshold) & (fimp > oimp)
        
                # Zero out weights where fimp > oimp
                # locations = torch.where(fimp > oimp*1e3)
                # if locations[0].numel() > 0:
                #     # Print the layer name and parameter name (n or fimp_n should work)
                #     print(f"Layer and Parameter: {n}")

                # If any such conditions are met, print the layer name and the indices
                # if torch.any(outlier_and_greater):
                #     print(f"Layer: {n}")
                #     print(f"Indices where condition is True: {torch.nonzero(outlier_and_greater)}")
                

                p[outlier_and_greater] = 0

                # Decay weights where fimp > oimp*exp(-beta*value)
                # values = torch.linspace(0, 1e6, 100)
                # decay_weights = torch.exp(-self.theta * values)
                # for idx, value in enumerate(values):
                #     p[fimp * torch.exp(-self.beta * value) > self.alpha * oimp] *= decay_weights[idx]


    # def modify_weight_layer_level(
            
    def modify_weight_layer_level(
        self,
        original_importance: Dict[str, torch.Tensor],
        forget_importance: Dict[str, torch.Tensor],
    ) -> None:
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                # Extract the layer name from the parameter name
                layer_name = n.split('.')[0]

                # Fetch the layer-level importances
                oimp = original_importance.get(layer_name, None)
                fimp = forget_importance.get(layer_name, None)

                if oimp is None or fimp is None:
                    continue  # Skip if layer-level importance is not available

                # Create tensors of the same shape as p filled with the scalar oimp and fimp
                oimp_tensor = torch.full_like(p, oimp.item())
                fimp_tensor = torch.full_like(p, fimp.item())

                # Synapse Selection with parameter alpha
                oimp_norm = oimp_tensor * self.selection_weighting
                locations = torch.where(fimp_tensor > oimp_norm)

                # Synapse Dampening with parameter lambda
                weight = ((oimp_tensor * self.dampening_constant) / fimp_tensor).pow(self.exponent)
                update = weight[locations]
                # Bound by 1 to prevent parameter values from increasing
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations] * update


