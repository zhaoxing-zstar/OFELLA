import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from torch import Tensor
import math
import os
from peft import LoraConfig, get_peft_model, TaskType

class DomainEmbedding(nn.Module):
    """Domain embeddings for federated learning (P-Tuning style continuous embeddings)"""
    def __init__(self, num_virtual_tokens: int, embedding_dim: int, init_range: float = 0.05):
        super().__init__()
        self.domain_embeddings = nn.Embedding(num_virtual_tokens, embedding_dim)
        nn.init.normal_(self.domain_embeddings.weight, mean=0.0, std=init_range)
    
    @property
    def weight(self):
        return self.domain_embeddings.weight

    @weight.setter
    def weight(self, value):
        self.domain_embeddings.weight.data = value.data
    
    @property
    def shape(self):
        return self.domain_embeddings.shape
    
    def forward(self, batch_size, moe_adapter = None):
        """Return domain embeddings expanded for batch"""
        if moe_adapter is None:
            return self.domain_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            return moe_adapter(self.domain_embeddings.weight).unsqueeze(0).expand(batch_size, -1, -1)

class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)

class MoEAdapterLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdapterLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)

class PromptingTemplate:
    def __init__(self, tokenizer, hidden_size, base_model, device):
        super(PromptingTemplate, self).__init__()
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.base_model = base_model
        self.device = device
        
        # Define prompt templates
        self.task_prompt = "For a sequential recommendation task, given a sequence of items that a specific user has interacted within a particular domain, predict the next most likely item the user will interact with."
        self.domain_prompt = "In domain"
        self.items_prompt = "the user has interacted with the following items:"
        
        # Tokenize and get embeddings for text prompts
        self._tokenize_prompts()
    
    def _tokenize_prompts(self):
        """Tokenize all text prompts and store their token IDs"""
        # Tokenize each prompt component
        task_tokens = self.tokenizer.encode(self.task_prompt, add_special_tokens=False)
        domain_tokens = self.tokenizer.encode(self.domain_prompt, add_special_tokens=False)
        items_tokens = self.tokenizer.encode(self.items_prompt, add_special_tokens=False)
        
        # Store token lengths for attention mask calculation
        self.task_prompt_len = len(task_tokens)
        self.domain_prompt_len = len(domain_tokens)
        self.items_prompt_len = len(items_tokens)
        print("Prompting Template Lengths:", self.task_prompt_len, self.domain_prompt_len, self.items_prompt_len)

         # Get token embeddings from the base model's input embedding layer (compatible with LoRA)
        input_embeddings = self.base_model.get_input_embeddings()
        self.task_emb = input_embeddings(torch.tensor(task_tokens, device=self.device))
        self.domain_emb = input_embeddings(torch.tensor(domain_tokens, device=self.device))
        self.items_emb = input_embeddings(torch.tensor(items_tokens, device=self.device))
    
    def get_prompt_embeddings(self, batch_size):
        """Get token embeddings for all prompt components"""
        # Expand to batch size
        return self.task_emb.unsqueeze(0).expand(batch_size, -1, -1), \
            self.domain_emb.unsqueeze(0).expand(batch_size, -1, -1), \
            self.items_emb.unsqueeze(0).expand(batch_size, -1, -1) # [batch_size, seq_len, hidden_size]
    
    def get_prompt_attention_masks(self, batch_size, device):
        """Get attention masks for prompt tokens"""
        task_mask = torch.ones(batch_size, self.task_prompt_len, device=device)
        domain_mask = torch.ones(batch_size, self.domain_prompt_len, device=device)
        items_mask = torch.ones(batch_size, self.items_prompt_len, device=device)
        
        return task_mask, domain_mask, items_mask

class LPRec(SequentialRecommender):
    """
    LLM P-Tuning Recommendation model
    
    Input format: [Domain Embeddings] [User Embedding] 
                  [Item 1 id embedding] [Item 1 text description embedding], ..., 
                  [item n id embedding] [item n text description embedding]
    """
    
    input_type = InputType.POINTWISE
    
    def __init__(self, config, dataset):
        super(LPRec, self).__init__(config, dataset)
        # load parameters info
        self.temperature = config['temperature']
        # hyperparameters: related with number of domains
        self.num_virtual_tokens = config['num_virtual_tokens']
        self.base_model_name = config['base_model_name']
        self.hidden_size = config['plm_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        
        # Store dataset for text access
        self.dataset = dataset
        
        # define layers and loss
        self.n_users = self.dataset.num(self.USER_ID)
        # self.user_embedding = nn.Embedding(self.n_users, self.hidden_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.id_gating = nn.Linear(self.hidden_size, 1)
        self.id_gating.weight.data.normal_(mean=0, std=0.02)

        # define MoE adapter for local adaption
        self.domain_moe_adapter = MoEAdapterLayer(
            n_exps = config['n_exps'],
            layers = config['adapter_layers'],
            dropout = config['adapter_dropout_prob']
        )
        
        # Initialize components
        self.setup_domain_embeddings()
        # self.setup_prediction_head()

        # Initialize base model
        self.base_model =  AutoModel.from_pretrained(self.base_model_name, attn_implementation="flash_attention_2", dtype=torch.bfloat16).cuda()
        for param in self.base_model.parameters():
            param.requires_grad = False
        self._add_lora_adapter(config['lora_r'], config['lora_alpha'], config['lora_dropout'], config['lora_target_modules'])
        
        # Initialize tokenizer for prompting template
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize prompting template
        self.prompting_template = PromptingTemplate(self.tokenizer, self.hidden_size, self.base_model, self.device)
        # Store dimensions
        self.base_embedding_dim = self.base_model.config.hidden_size
        
        # Loss function
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        # self._init_weights(self.user_embedding)
        self._init_weights(self.item_embedding)
        # self._init_weights(self.prediction_head)

    def _add_lora_adapter(self, lora_r, lora_alpha, lora_dropout, lora_target_modules):
        # Configure and attach LoRA to attention and MLP of the first and last layers
        num_layers = self.base_model.config.num_hidden_layers
        self.tunable_modules = [f"layers.{i}.{module}" for i in [0, num_layers-1] for module in lora_target_modules]

        lora_config = LoraConfig(
            r = lora_r, 
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout,
            target_modules = self.tunable_modules,
            bias = 'none',  
            task_type = 'FEATURE_EXTRACTION'
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        
    def setup_domain_embeddings(self):
        """Initialize domain embedding layers"""
        # Domain Embeddings for federated learning (PTuning style embeddings sent to server)
        self.domain_general_embedding = DomainEmbedding(
            num_virtual_tokens=self.num_virtual_tokens,
            embedding_dim=self.hidden_size,
            init_range=self.initializer_range
        )

        self.domain_specific_embedding = DomainEmbedding(
            num_virtual_tokens=self.num_virtual_tokens,
            embedding_dim=self.hidden_size,
            init_range=self.initializer_range
        )
        
    def setup_prediction_head(self):
        """Setup sequence modeling components"""
        # Prediction head for next item prediction
        self.prediction_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(self.hidden_dropout_prob),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        )
    
    # def get_item_text_embeddings(self, item_ids):
    #     """
    #     Get text embeddings for given item IDs from dataset
    #     Args:
    #         item_ids: numpy array or tensor of item IDs
    #     Returns:
    #         Text embeddings tensor of shape [batch_size, hidden_size] (last token representation)
    #     """
    #     if isinstance(item_ids, torch.Tensor):
    #         item_ids = item_ids.cpu().numpy()
        
    #     batch_size = len(item_ids)
    #     text_embeddings = []
        
    #     for item_id in item_ids:
    #         if item_id == 0:  # Handle padding tokens
    #             # Use the pre-stored PAD token embedding from the text processor
    #             text_embeddings.append(self.text_processor.pad_embedding)
    #         else:
    #             # Process regular text through the text processor
    #             text = self.dataset.text_dataset.iid2text[item_id]
    #             text_emb, _ = self.text_processor([text])  # Process single text
    #             text_embeddings.append(text_emb.squeeze(0))  # [hidden_size]
        
    #     # Stack all embeddings
    #     text_embeddings = torch.stack(text_embeddings, dim=0)  # [batch_size, hidden_size]
    #     print(text_embeddings)
        
    #     return text_embeddings
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=self.initializer_range, a = -3*self.initializer_range, b = 3*self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, item_seq, item_id_embs, item_text_embs, item_seq_len, user_ids=None):
        """
        Forward pass with prompting template
        Args:
            item_seq: [batch_size, seq_len] interaction sequence
            item_seq_len: [batch_size] sequence lengths
            user_ids: [batch_size] user IDs 
        Returns:
            output: [batch_size, hidden_size] sequence representation
        """
        batch_size, seq_len = item_seq.size()

        task_emb, domain_emb, items_emb= self.prompting_template.get_prompt_embeddings(batch_size)
        
        domain_general_emb = self.domain_general_embedding(batch_size, self.domain_moe_adapter)  # [batch_size, num_virtual_tokens, hidden_size]
        domain_specific_emb = self.domain_specific_embedding(batch_size)  # [batch_size, num_virtual_tokens, hidden_size]
        
        interleaved_embs = torch.cat([item_id_embs, item_text_embs], dim=2)
        interleaved_embs = interleaved_embs.view(batch_size, seq_len*2, self.hidden_size)
        
        # [task_prompt] + [domain_prompt] + [domain_general] + [domain_specific] + [items_prompt] + [interleaved_items] + [output_prompt]
        sequence_parts = [
            task_emb,              # [batch_size, task_len, hidden_size]
            domain_emb,            # [batch_size, domain_len, hidden_size]
            domain_general_emb,    # [batch_size, num_virtual_tokens, hidden_size]
            domain_specific_emb,   # [batch_size, num_virtual_tokens, hidden_size]
            items_emb,             # [batch_size, items_len, hidden_size]
            interleaved_embs,      # [batch_size, seq_len*2, hidden_size]
        ]
        
        attention_parts = [
            torch.ones(batch_size, task_emb.size(1), device=self.device),                    # task prompt
            torch.ones(batch_size, domain_emb.size(1), device=self.device),                  # domain prompt
            torch.ones(batch_size, domain_general_emb.size(1), device=self.device),          # domain general
            torch.ones(batch_size, domain_specific_emb.size(1), device=self.device),         # domain specific
            torch.ones(batch_size, items_emb.size(1), device=self.device),                   # items prompt
            (item_seq > 0).float().repeat_interleave(2, dim=1),                         # interleaved items (masked for padding)
        ]
        
        input_emb = torch.cat(sequence_parts, dim=1)

        attention_mask = torch.cat(attention_parts, dim=1)
        
        base_model_output = self.base_model(
            inputs_embeds=input_emb,
            attention_mask=attention_mask
        )

        # Total sequence length = task_len + domain_len + 2*num_virtual_tokens + items_len + 2*item_seq_len
        total_prompt_len = (self.prompting_template.task_prompt_len + 
                           self.prompting_template.domain_prompt_len + 
                           2 * self.num_virtual_tokens + 
                           self.prompting_template.items_prompt_len + 
                           2 * item_seq_len)
        
        # Apply prediction head, [B, H]
        sequence_output = self.gather_indexes(base_model_output.last_hidden_state, total_prompt_len-1)
        
        return sequence_output
    
    def diffloss(self, input1, input2):
        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        return diff_loss
    
    def calculate_loss(self, interaction):
        """Calculate training loss"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_ids = interaction[self.USER_ID] 
        pos_items = interaction[self.POS_ITEM_ID]
        
        # apply MoE adapter to item text embeddings
        text_embs = self.dataset.item_text_embeddings.weight
        # adaptive fusion of item id and text embeddings
        id_gate = torch.sigmoid(self.id_gating(self.item_embedding.weight))
        gated_id_embs = id_gate * self.item_embedding.weight
        seq_output = self.forward(item_seq, gated_id_embs[item_seq], text_embs[item_seq], item_seq_len, user_ids)

        # Normalize both sequence output and item embeddings for better training
        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(text_embs + gated_id_embs, dim=1)
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        loss = self.loss_fct(logits, pos_items)
        diff_loss = self.diffloss(text_embs, self.item_embedding.weight)
        return loss + diff_loss 
    
    def full_sort_predict(self, interaction):
        """Predict scores for all items"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_ids = interaction[self.USER_ID] if self.USER_ID in interaction else None

         # apply MoE adapter to item text embeddings
        text_embs = self.dataset.item_text_embeddings.weight
        # adaptive fusion of item id and text embeddings
        id_gate = torch.sigmoid(self.id_gating(self.item_embedding.weight))
        gated_id_embs = id_gate * self.item_embedding.weight
        seq_output = self.forward(item_seq, gated_id_embs[item_seq], text_embs[item_seq], item_seq_len, user_ids)

        # Normalize both for consistent similarity computation
        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(text_embs + gated_id_embs, dim=1)
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B n_items]
        return scores
        
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_virtual_tokens': self.num_virtual_tokens,
            'qwen_model': self.base_model_name,
            'hidden_size': self.hidden_size
        }