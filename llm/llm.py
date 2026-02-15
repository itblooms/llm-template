import torch
import torch.nn as nn
from omegaconf import DictConfig
from .modules import TransformerBlock, RMSNorm, compute_rope_params


class LLM(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        self.tok_embed = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.trsf_blocks = nn.ModuleList(
            [
                TransformerBlock(config) for _ in range(config["num_blocks"])
            ]
        )
        self.final_norm = RMSNorm(**config.norm)
        self.out = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)
        cos, sin = compute_rope_params(
            head_dim=config.rope["head_dim"],
            theta_base=config.rope["theta_base"],
            context_length=config["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def forward(self, inpus_ids: torch.Tensor) -> torch.Tensor:
        tok_embeds = self.tok_embed(inpus_ids)
        x = tok_embeds
        num_tokens = tok_embeds.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device), diagonal=1)
        
        for block in self.trsf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out(x)
        return logits
