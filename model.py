import torch
import torch.nn as nn
import torch.nn.functional as F

from minGPT.mingpt.model import GPT, NewGELU

from peft import LoraConfig, get_peft_model, TaskType

class GPTMatchingModel(GPT):
    """GPT-based model for JD-CV matching"""

    @staticmethod
    def get_default_config():
        C = GPT.get_default_config()
        # Add LoRA specific configs
        C.use_lora = False
        C.lora_r = 8  # LoRA rank
        C.lora_alpha = 16  # LoRA alpha scaling
        C.lora_dropout = 0.1
        C.lora_target_modules = ["c_attn", "c_proj"]  # Target attention layers
        C.label_map={
            'No Fit': 0.0,
            'Potential Fit': 0.6,
            'Good Fit': 1.0
        }
        C.sep_token = "###"
        C.pad_token_id = 50256
        C.thresholds = [0.3, 0.8]
        return C
    
    def __init__(self, config):
        # Initialize parent GPT class
        super().__init__(config)

        self.config = config  # Store config for later use
        
        # Remove the language modeling head as we won't need it
        del self.lm_head
        
        # Add matching head for score prediction
        # Using n_embd from the transformer as input size
        self.match_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            NewGELU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(config.n_embd // 2, config.n_embd // 4),
            NewGELU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(config.n_embd // 4, 1),
            nn.Sigmoid()  # Sigmoid for score between 0 and 1
        )
        
        # Initialize the matching head weights
        self.match_head.apply(self._init_weights)

        # Apply LoRA if specified
        if config.use_lora:
            self._apply_lora()
    
    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT matching model by:
        1. Creating a new GPTMatchingModel
        2. Loading pretrained GPT weights
        3. Adding new matching head
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        
        # Create config and model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257  # OpenAI's model vocabulary
        config.block_size = 1024   # OpenAI's model block_size
        model = cls(config)
        
        # Load pretrained weights for the transformer
        pretrained_gpt = GPT.from_pretrained(model_type)
        
        # Copy transformer weights
        model.transformer.load_state_dict(pretrained_gpt.transformer.state_dict())
        
        return model
    
    def _apply_lora(self):
        """Apply LoRA to the model (optional)"""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        
        # Convert transformer to LoRA
        self.transformer = get_peft_model(self.transformer, lora_config)
        
        # Print trainable parameters info
        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        """Print number of trainable parameters"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || "
            f"all params: {all_param} || "
            f"trainable%: {100 * trainable_params / all_param:.2f}"
        )
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        # Forward pass through GPT model
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # Mean pooling and score prediction
        x = x.mean(dim=1)
        score = self.match_head(x).squeeze(-1)
        
        # Loss calculation
        loss = None
        if targets is not None:
            loss = F.mse_loss(score, targets)
                
        return score, loss
    
    def configure_optimizers(self, train_config):
        """
        Configure optimizer with three parameter groups:
        1. Transformer parameters with weight decay 
        2. Transformer parameters without weight decay
        3. Matching head parameters with potentially higher learning rate
        """
        # Get transformer parameters
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        # Handle transformer parameters 
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'transformer.{mn}.{pn}' if mn else f'transformer.{pn}'
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Validate transformer parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if 'transformer' in pn}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        # Create optimizer groups
        optim_groups = [
            # Transformer params with weight decay
            {"params": [param_dict[pn] for pn in sorted(list(decay))], 
                "weight_decay": train_config.weight_decay},
            # Transformer params without weight decay
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0},
            # Matching head params (higher learning rate)
            {"params": [p for n, p in self.match_head.named_parameters()],
                "weight_decay": train_config.weight_decay,
                "lr": train_config.learning_rate * 10}  
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps, fused=True)
        return optimizer