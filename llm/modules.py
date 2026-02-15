import torch
import torch.nn as nn
from typing import Tuple, Optional


class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Normalization) layer with additional possibility
    to add shift.
    
    Args:
        emb_dim (int): Embedding dimension.
        eps (float): Additional small value added to variance to avoid zero division.
        bias (bool): Defines if shift is applied or not.

    Returns: 
        torch.Tensor: Normalized batch of token embeddings 
            with size [batch_size (B), seq_len (L), emb_dim (D)].
    """
    def __init__(
            self, 
            emb_dim: int, 
            eps: float = 1e-6,
            bias: bool = False
    ) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        if self.shift:
            norm_x = norm_x + self.shift
        
        return norm_x
    
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) layer, which shares key and value matricies
    among heads of groups to reduce computation costs related to MHSA.
    Each head has unique query matrix. For key and value matricies head are divided
    to several groups, head in each group share same key and value matricies.

    Args:
        dim_in (int): Embedding dimension.
        num_heads (int): Total number of heads.
        head_dim (int): Dimension of query, key and value vectors.
        num_kv_groups (int): Number of groups of heads, which defines
            total number of key and value matricies used in GQA.
        qk_norm (bool): Defines if RMSNorm is applied to head_dim of 
            key and value matricies.
    Returns:
        torch.Tensor:  Batch of representations of tokens in a sequence. 
            Equal to dim_in to ensure residual connection.
    """
    def __init__(
            self,
            dim_in: int,
            num_heads: int,
            num_kv_groups: int,
            head_dim: Optional[int] = None,
            qk_norm: bool = False
    ) -> None:
        super().__init__()
        if num_heads % num_kv_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_groups ({num_kv_groups})."
            )
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            if dim_in % num_heads != 0:
                raise ValueError(
                    f"If head_dim is not set, dim_in ({dim_in}) must be divisible by "
                    f"num_heads ({num_heads}) to ensure dim_out == dim_in. "
                )
            head_dim = dim_in // num_heads
        self.head_dim = head_dim
        self.dim_out = num_heads * head_dim
        self.dim_in = dim_in

        self.W_query = nn.Linear(dim_in, self.dim_out, bias=False)  # [B, L, D] -> [B, L, h_D, N_h]
        self.W_key = nn.Linear(dim_in, head_dim * num_kv_groups, bias=False)  # [B, L, D] -> [B, L, h_D, N_g]
        self.W_value = nn.Linear(dim_in, head_dim * num_kv_groups, bias=False)  # [B, L, D] -> [B, L, h_D, N_g]
        self.out_proj = nn.Linear(self.dim_out, dim_in, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(
            self, 
            x: torch.Tensor, 
            mask: torch.Tensor, 
            cos: torch.Tensor,
            sin: torch.Tensor, 
            start_pos: int = 0,
            kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, num_tokens, _ = x.shape

        # Get Q, K, V matricies
        queries = self.W_query(x)  # [batch_size, num_tokens, num_heads * head_dim]
        keys = self.W_key(x)  # [batch_size, num_tokens, num_kv_groups * head_dim]
        values = self.W_value(x)  # [batch_size, num_tokens, num_kv_groups * head_dim]

        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [batch_size, num_heads, num_tokens, head_dim]
        keys_new = keys.view(
            batch_size, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)  # [batch_size, num_kv_groups, num_tokens, head_dim]
        values_new = values.view(
            batch_size, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)  # [batch_size, num_kv_groups, num_tokens, head_dim]

        # Optional Q/K RMSNorm used in Qwen3
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # Apply RoPE
        queries = GroupedQueryAttention.apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = GroupedQueryAttention.apply_rope(keys_new, cos, sin, offset=start_pos)

        if kv_cache:  # If kv_cache is not empty and stores previously calculated keys and values
            cached_k, cached_v = kv_cache
            keys = torch.cat([cached_k, keys_new], dim=2)  # num_tokens dim
            values = torch.cat([cached_v, values_new], dim=2)
            next_kv_cache = (keys, values)
        else:  # If kv_cache is empty -> No words have been generated
            start_pos = 0  # Reset RoPE
            keys, values = keys_new, values_new
            next_kv_cache = (keys, values)

        # Exopand keys and values matricies to match queries matrix's num_heads
        keys = keys.repeat_interleave(self.group_size, dim=1)  # num_kv_groups dim
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)  # [B, N_h, L, L]
        # Apply triangluar mask to obscure future tokens
        attn_scores = attn_scores.masked_fill_(mask=mask, value=-torch.inf)
        attn_weights = nn.functional.softmax(attn_scores / self.head_dim ** 2, dim=-1)
        #  [B, N_h, L, D] -> [B, L, N_h, D] -> [B, L, N_h * D]
        #  N_h * D stands for concatenation from all heads
        output = (attn_weights @ values).transpose(1, 2).reshape(batch_size, num_tokens, self.dim_out)
        return self.out_proj(output), next_kv_cache

    @staticmethod
    def apply_rope(
            x: torch.Tensor, 
            cos: torch.Tensor, 
            sin: torch.Tensor, 
            offset: int = 0
    ) -> torch.Tensor:
        """
        Applies RoPE (Rotary Positional Encoding) to x in attention layer.

        Args:
            x (torch.Tensor): Tensor of token embeddings inside attention.
                Has shape [batch_size, num_heads, seq_length, head_dim].
            cos (torch.Tensor): Precomputed table of cosines with argument m * theta_i 
                for each m (token position) in context window and each i (embedding coordinate).
                Has shape [context_length, head_dim].
            sin (torch.Tensor): Precomputed table of sines with argument m * theta_i 
                for each m (token position) in context window and each i (embedding coordinate).
                Has shape [context_length, head_dim].
            offset (int): Current position (current token) in context.
        Returns:
            torch.Tensor: Initial x tensor with applied RoPE to it's tokens.
                Has same shape as x.
        
        Notes:
            rotated_x in final has the following look:
                [-x_{head_dim/2 + 1}, ..., -x_{head_dim}, x_1, ..., x_{head_dim/2}]
            Check this explanation for further explanations: 
                https://nn.labml.ai/transformers/rope/index.html
        """
        batch_size, num_heads, seq_len, head_dim = x.shape  # seq_len = context_length
        if head_dim % 2 != 0:
            raise ValueError(
                f"Head dimension must be even, but you have head_dim = {head_dim}"
            )
        x1 = x[..., : head_dim // 2]  # First half of embeddings
        x2 = x[..., head_dim // 2 :]  # Second half of emdeddings

        cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)

        rotated_x = torch.cat([-x2, x1], dim=-1)  # x rotated by 90 degree
        roped_x = (x * cos) + (rotated_x * sin)
        return roped_x

# TODO Move to llm/llm.py
# Use https://nn.labml.ai/transformers/rope/index.html as a reference to understand details
def compute_rope_params(
    head_dim: int,
    theta_base: int = 10_000,
    context_length: int = 4096
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes sine and cosine tables for each token position and 
    each i in [1, 2, ..., head_dim / 2]. Parameter i is a coordinate of 
    embedding inside attention layer.

    Args:
        head_dim (int): Embeddings dimension inside attention layer.
        theta_base (int): Base used to calculate theta_i's used in RoPE's rotation matrix.
            Note: theta_base ^ {-2 * (i - 1) / head_dim} for i in [1, 2, ..., head_dim / 2]
        context_length (int): Context window of a model, i.e., how much tokens can be processed at once.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Final tables of sines and cosines. 

    Notes: 
        m's rows in sine and cosine tables are calculated with the following arguments:
            [m*theta_0, m*theta_1, ..., m*theta_{head_dim/2}, m*theta_0, m*theta_1, ..., m*theta_{head_dim/2}]
        Check this explanation for further explanations: 
            https://nn.labml.ai/transformers/rope/index.html
    """
    if head_dim % 2 != 0:
        raise ValueError(
        f"Head dimension must be even, but you have head_dim = {head_dim}"
        )
    # theta_i angles, for i in [1, 2, ..., head_dim / 2]. theta_i = 10000^(-2 * (i-1) / d)
    theta = 1. / theta_base ** (torch.arange(0, head_dim // 2, 2) / head_dim)
    positions = torch.arange(context_length)  # Token positions
    angles = positions.unsqueeze(1) * theta.unsqueeze(0)  # [context_length, head_dim // 2]
    # Total table of angles
    angles = torch.cat([angles, angles], dim=1)  # [context_length, head_dim]

    # Compute sines and cosines tables for further final calculations of RoPE
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin
    
class MultiHeadSelfAttention(GroupedQueryAttention):
    """
    Implementation of Multi-Head Self-Attention (MHSA) as a 
    special case of Grouped-Query Attention.
    Each head has unique keys and values matricies. num_kv_groups = num_heads

    Args:
        dim_in (int): Embedding dimension.
        num_heads (int): Total number of heads.
        head_dim (int): Dimension of query, key and value vectors.
        qk_norm (bool): Defines if RMSNorm is applied to head_dim of 
            key and value matricies.
    Returns:
        torch.Tensor:  Batch of representations of tokens in a sequence. 
            Equal to dim_in to ensure residual connection.
    """
    def __init__(
            self, 
            dim_in: int, 
            num_heads: int,
            head_dim: Optional[int] = None, 
            qk_norm: bool = False
    ) -> None:
        super().__init__(
            dim_in=dim_in, 
            num_heads=num_heads, 
            num_kv_groups=num_heads,
            head_dim=head_dim, 
            qk_norm=qk_norm
        )

class MultiQueryAttention(GroupedQueryAttention):
    """
    Implementation of Multi-Query Attention (MQA) as a 
    special case of Grouped-Query Attention.
    Every head shares same keys and values matricies. num_kv_groups = 1

    Args:
        dim_in (int): Embedding dimension.
        num_heads (int): Total number of heads.
        head_dim (int): Dimension of query, key and value vectors.
        qk_norm (bool): Defines if RMSNorm is applied to head_dim of 
            key and value matricies.
    Returns:
        torch.Tensor:  Batch of representations of tokens in a sequence. 
            Equal to dim_in to ensure residual connection.
    """
    def __init__(
            self, 
            dim_in: int, 
            num_heads: int, 
            head_dim: Optional[int] = None, 
            qk_norm: bool = False
        ) -> None:
        super().__init__(
            dim_in=dim_in, 
            num_heads=num_heads, 
            num_kv_groups=1, 
            head_dim=head_dim, 
            qk_norm=qk_norm
        )

class FeedForwardNet(nn.Module):
    """
    Feed-Forward Network (FFN) after attention layer in Transformer block.

    Args:
        dim_in (int): Embedding dimension.
        hidden_dim (int): Dimension of hidden layers (number of neurons in a layer).
        activation_fn (str): Activation function used as a non-linearity.
    Returns:
        torch.Tensor: Final tokens representation tensor of Transformer block.
            (Note: before adding residual)
    """
    def __init__(
            self,
            dim_in: int,
            hidden_dim: int,
            activation_fn: str = "silu"
    ) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.dim_out = dim_in
        self.activation_fn = activation_fn

        self.fc1 = nn.Linear(dim_in, hidden_dim, bias=False)
        self.fc2 = nn.Linear(dim_in, hidden_dim, bias=False)
        self.fc_out = nn.Linear(hidden_dim, dim_in, bias=False)
        
        match activation_fn:
            case "silu":
                self.act_fn = nn.SiLU()
            case "gelu":
                self.act_fn = nn.GELU()
            case _:
                raise ValueError(
                    f"Unsupported activation function: {activation_fn}. "
                    f"Please, choose between 'silu' and 'gelu'."
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = self.act_fn(x1) * x2
        return self.fc_out(x)

class MoEFeedForwardNet(nn.Module):
    """
    Mixture-of-Experts (MoE) layer implementation.
    Args:
        ...
    Returns:
        ...
    """
    def __init__(
            self, 
            num_experts: int, 
            num_experts_per_token: int, 
            dim_in: int,
            moe_hidden_dim: int
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.dim_in = dim_in
        self.moe_hidden_dim = moe_hidden_dim
        self.gate = nn.Linear(dim_in, moe_hidden_dim, bias=False)

    
    def forward(self, x: torch.Tensor):
        pass

class TransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        match config.attention_type:
            case "MHSA":
                self.attn = MultiHeadSelfAttention(**config.attention)
            case "GQA":
                self.attn = GroupedQueryAttention(**config.attention)
            case "MQA":
                self.attn = MultiQueryAttention(**config.attention)
            case _:
                raise ValueError(
                    f"Unsupported attention type: {config.attention_type}. "
                    f"Please, choose between 'MHSA', 'GQA' and 'MQA'."
                )
        
        match config.ffn_type:
            case "Dense":
                self.ffn = FeedForwardNet(**config.ffn)
            case "MoE":
                self.ffn = MoEFeedForwardNet(**config.ffn)
            case _:
                raise ValueError(
                    f"Unsupported FFN type: {config.ffn_type}. "
                    f"Please, choose between 'Dense' and 'MoE'."
                )
        self.norm1 = RMSNorm(
            emb_dim=config.norm["embedding_dim"], 
            eps=config.norm["epsilon"],
            bias=config.norm["bias"]
        )
        self.norm2 = RMSNorm(
            emb_dim=config.norm["embedding_dim"],
            eps=config.norm["eps"],
            bias=config.norm["bias"]
        )
    
    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            start_pos: int = 0,
            kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        residual = x
        x = self.norm1(x)
        x, next_cache = self.attn(x, mask, cos, sin, start_pos=start_pos, kv_cache=kv_cache)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        return x, next_cache
    