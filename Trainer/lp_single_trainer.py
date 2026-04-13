import os
from time import time
from tqdm import tqdm
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from recbole.trainer import Trainer
from recbole.utils import dict2str, set_color, get_gpu_usage, early_stopping, calculate_valid_score
# from .utils import early_stopping
from tqdm import tqdm
import numpy as np
from collections import OrderedDict


class SingleTrainer(Trainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super().__init__(config, model)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None, comm_round = 0):
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)
        
        self.eval_collector.data_collect(train_data)
        if self.config['train_neg_sample_args'].get('dynamic', 'none') != 'none':
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            progressive_epoch = comm_round * self.epochs + epoch_idx
            training_start_time = time()
            
            train_loss = self._train_epoch(train_data, progressive_epoch, show_progress=show_progress)
            self.train_loss_dict[progressive_epoch] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(progressive_epoch, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(progressive_epoch, train_loss, tag = f'Loss/Train_{self.config["dataset"]}')
            self.wandblogger.log_metrics({'epoch': progressive_epoch, 'train_loss': train_loss, 'train_step': progressive_epoch, 'communication_round': comm_round},
                                         head=f'train_{self.config["dataset"]}')

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(progressive_epoch, verbose=verbose)
                continue
            if (progressive_epoch + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger)
                validation_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (progressive_epoch, validation_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar(f'{self.config["dataset"]} Vaild_score', valid_score, progressive_epoch)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': progressive_epoch, 'communication_round': comm_round},
                                         head=f'valid_{self.config["dataset"]}')
                
                if update_flag:
                    if saved:
                        self._save_checkpoint(progressive_epoch, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(progressive_epoch, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (progressive_epoch - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result


    def get_domain_general_embedding(self):
        return self.model.domain_general_embedding.weight
    
    def get_lora_adapters(self):
        """Extract lora_A weights from all tunable modules"""
        lora_weights = {}
        for module_name in self.model.tunable_modules:
            # Access the lora_A weight: model.layers[i].module.lora_A.default.weight
            module_parts = module_name.split('.')
            layer_idx = int(module_parts[1])  # layers.0 -> 0
            submodule = module_parts[2]  # mlp, self_attn, etc.
            
            layer = self.model.base_model.layers[layer_idx]
            target_module = getattr(layer, submodule)
            
            # Find the LoRA adapter within the target module
            for name, module in target_module.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module.lora_A, 'default'):
                    weight_key = f"layers.{layer_idx}.{submodule}.{name}"
                    lora_weights[weight_key] = module.lora_A.default.weight.data
        return lora_weights
    
    def set_lora_adapters(self, updated_adapters):
        """Update lora_A weights in the model"""
        for weight_key, new_weight in updated_adapters.items():
            # Parse the weight key: layers.0.mlp.up_proj
            parts = weight_key.split('.')
            # Navigate to the specific module
            layer = self.model.base_model.layers[int(parts[1])]
            target_module = getattr(layer, parts[2])
            current_module = getattr(target_module, parts[3])
            # Update the weight
            if hasattr(current_module, 'lora_A') and hasattr(current_module.lora_A, 'default'):
                current_module.lora_A.default.weight.data = self.config['smoothing_factor'] * \
                     current_module.lora_A.default.weight.data + (1 - self.config['smoothing_factor']) * new_weight.data
    
    def set_domain_general_embedding(self, updated_embedding):
        self.model.domain_general_embedding.weight = self.config['smoothing_factor'] * \
            self.model.domain_general_embedding.weight + (1 - self.config['smoothing_factor']) * updated_embedding

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        """Train a single epoch for a specific client."""
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train_Client_{self.config["dataset"]} {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        
        scaler = torch.amp.GradScaler(enabled=self.enable_scaler)
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.enable_amp):
                losses = loss_func(interaction)
            
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                
            self._check_nan(loss)
            scaler.scale(loss).backward()
            
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
            
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('Loss: ' + str(losses.item()), 'yellow'))
        
        return total_loss

    @torch.no_grad()
    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False
    ):  
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.enable_amp):
            return super().evaluate(eval_data, load_best_model, model_file, show_progress)