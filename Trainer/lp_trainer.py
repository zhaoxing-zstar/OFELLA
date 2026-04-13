import os
from logging import getLogger
from time import time
from datetime import date
import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn import functional as F
from collections import OrderedDict
import collections

class LPRecTrainer:
    """Federated LPRec Trainer Management Class
    """

    def __init__(self, client_trainers: dict, communicatoin_rounds: int, logger = None):
        self.client_trainers = client_trainers
        self.communication_rounds = communicatoin_rounds
        self.client_domains = list(client_trainers.keys())
        self.num_clients = len(self.client_domains)
        self.train_metrics = collections.defaultdict(dict)
        self.test_metrics = collections.defaultdict(dict)
        self.logger = logger

    def _cross_domain_attention(self, X):
        """
        Cross-domain attention mechanism
        """
        N, E = X.shape
        scores = torch.matmul(X, X.T)
        weights = F.softmax(scores, dim=-1) 
        Z_seq = torch.matmul(weights, X).reshape(self.num_clients, -1, E)
        return Z_seq, weights
    
    # def _average_attention_weight(self, attention_weights):
    #     """
    #     Average attention weights within small blocks.
        
    #     Args:
    #         attention_weights: Tensor of shape [N*V, N*V] where N is number of clients and V is the number of virtual tokens
    #     Returns:
    #         averaged_weights: Tensor of shape [N, N] with averaged attention weights
    #     """
    #     V = attention_weights.shape[0] // self.num_clients
    #     # Reshape to [N, V, N, V] to separate the blocks
    #     weights_reshaped = attention_weights.view(self.num_clients, V, self.num_clients, V)
    #     averaged_weights = weights_reshaped.sum(dim=(1, 3)) / V
    #     return averaged_weights
    
    # def _update_single_module_weights(self, module_weights_dict, averaged_attention_weights):
    #     """
    #     Update weights for a single module across all domains using averaged attention weights.
    #     """
    #     all_weights = torch.stack([module_weights_dict[domain] for domain in self.client_domains])
    #     updated_weights = torch.einsum('ij,jkl->ikl', averaged_attention_weights, all_weights)
    #     updated_weights_dict = {}
    #     for i, client_domain in enumerate(self.client_domains):
    #         updated_weights_dict[client_domain] = updated_weights[i]
    #     return updated_weights_dict

    def _update_single_module_weights(self, module_weights_dict, stacked_general_prompts):
        """
        Update weights for a single module across all domains using cross-domain fusion.
        """
        all_weights = []
        for domain in self.client_domains:
            weight = module_weights_dict[domain]
            all_weights.append(weight)
        W = torch.stack(all_weights, dim=0)  # (N, R, E)
        fused_W, attention_weights = self._match_shape_and_fuse(stacked_general_prompts, W)
        
        # Convert back to dictionary format
        updated_weights_dict = {}
        for i, domain in enumerate(self.client_domains):
            updated_weights_dict[domain] = fused_W[i]
        return updated_weights_dict
    
    def _match_shape_and_fuse(self, E, W):
        """ Split W into chunks matched with the last dimension of E
        Args:
            W: (N, R, E) - certain module weights from all domains with heterogenous shape
        """
        if E.shape[-1] == W.shape[-1]:
            return self._cross_domain_fuse(E, W)
        else:
            W_chunks = W.split(E.shape[-1], dim = -1)
            results, attention_weights = [], []
            for W_i in W_chunks:
                fused_i, a_i = self._cross_domain_fuse(E, W_i)
                results.append(fused_i)
                attention_weights.append(a_i)
            return torch.cat(results, dim = -1), attention_weights
    
    def _cross_domain_fuse(self, E, W):
        """
        Cross-domain fusion using attention mechanism.
        
        Args:
            E: (N, V, E) - stacked general prompts
            W: (N, R, E) - certain module weights from all domains
        """
        # use einsum to do batched dot-products
        E_norm, W_norm = F.normalize(E, p = 2, dim = -1), F.normalize(W, p = 2, dim = -1)
        A = torch.einsum('kve,jre->kvjr', E_norm, W_norm).transpose(0,1)  # (V, N, N, R)
        # normalize across j (the source domains)
        A = F.softmax(A, dim=2)  # (V, N, N, R)
        print("Aggregated weight", A[0,:,:,0])
        fused_W = torch.einsum('vkjr,jre->vkre', A, W)  # (V, N, R, E)
        fused_W = fused_W.mean(dim = 0)
        return fused_W, A
    
    def _update_lora_adapters(self, stacked_general_prompts):
        """
        Update lora adapters based on stacked attention weights.

        Args: 
            stacked_general_prompts: [N, V, E]
        """
        # Get LoRA adapters from all clients in nested structure: {'domain': {'module': weight}}
        all_lora_adapters = {}
        for client_domain in self.client_domains:
            all_lora_adapters[client_domain] = self.client_trainers[client_domain].get_lora_adapters()
        tunable_module_names = list(all_lora_adapters[self.client_domains[0]].keys())
        
        for module_name in tunable_module_names:
            module_weights_dict = {}
            for client_domain in self.client_domains:
                module_weights_dict[client_domain] = all_lora_adapters[client_domain][module_name]
            updated_module_weights = self._update_single_module_weights(module_weights_dict, stacked_general_prompts)
            for client_domain, updated_weight in updated_module_weights.items():
                all_lora_adapters[client_domain][module_name] = updated_weight
        
        # Send updated LoRA_A adapters back to each client
        for client_domain in self.client_domains:
            self.client_trainers[client_domain].set_lora_adapters(all_lora_adapters[client_domain])

    def fedtrain(self, train_data_dic, valid_data_dic, verbose=True, show_progress=False):
        """Federated training with multiple clients using cross-domain attention.

        Args:
            train_data_dic (dict): Dictionary of training data for each client
            valid_data_dic (dict): Dictionary of validation data for each client
            verbose (bool): Whether to print training information
            show_progress (bool): Whether to show progress bars
        """
        assert len(train_data_dic) == self.num_clients, f"Expected {self.num_clients} training datasets, got {len(train_data_dic)}"
        
        for commu_round in range(self.communication_rounds):
            for client_domain in self.client_domains:
                train_data = train_data_dic[client_domain]
                valid_data = valid_data_dic[client_domain]
                best_valid_score, best_valid_result = self.client_trainers[client_domain].fit(train_data, valid_data, saved = True, verbose=verbose, show_progress=show_progress, comm_round=commu_round)
                self.train_metrics[client_domain]['best_valid_score'] = best_valid_score
                self.train_metrics[client_domain]['best_valid_result'] = best_valid_result
            # Apply cross-domain attention
            domain_general_prompts = torch.cat([self.client_trainers[client_domain].get_domain_general_embedding() for client_domain in self.client_domains], dim=0)
            stacked_general_prompts = torch.stack([self.client_trainers[client_domain].get_domain_general_embedding() for client_domain in self.client_domains], dim=0)
            updated_prompts, attention_weights = self._cross_domain_attention(domain_general_prompts)
            print("Domain General Prompts attention weight", attention_weights)
            self._update_lora_adapters(stacked_general_prompts)
            # averaged_attention_weights = self._average_attention_weight(attention_weights) # [self.num_clients, self.num_clients]
            # self._update_lora_adapters(averaged_attention_weights)

            for i, client_domain in enumerate(self.client_domains):
                self.client_trainers[client_domain].set_domain_general_embedding(updated_prompts[i])
            
            if verbose:
                self.logger.info(f"Communication round {commu_round}: Applied cross-domain attention with shape {updated_prompts.shape}")
                self.logger.info(f"Attention weights shape: {attention_weights.shape}")
        self.logger.info("Federated training completed")
        return self.train_metrics

    def fedtest(self, test_data_dic, verbose=True, show_progress=False):
        for client_domain in self.client_domains:
            test_data = test_data_dic[client_domain]
            test_result = self.client_trainers[client_domain].evaluate(test_data, load_best_model=True, show_progress=show_progress)
            self.test_metrics[client_domain]['test_result'] = test_result
        self.logger.info("Federated testing completed")
        return self.test_metrics