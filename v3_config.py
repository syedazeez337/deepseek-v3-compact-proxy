from dataclasses import dataclass


@dataclass(frozen=True)
class CompactV3Config:
    vocab_size: int = 32_000
    context_length: int = 264
    n_layer: int = 6
    d_model: int = 512
    n_heads: int = 8
    q_lora_rank: int = 128
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 48
    qk_rope_head_dim: int = 16
    v_head_dim: int = 48
    rope_base: float = 10_000.0
    rms_norm_eps: float = 1e-6
    use_moe: bool = True
    n_dense_layers: int = 1
    n_routed_experts: int = 32
    n_shared_experts: int = 1
    top_k: int = 2
    expert_hidden_dim: int = 512
    route_scale: float = 0.75
    router_bias_update_rate: float = 1e-4
    sequence_balance_coefficient: float = 1e-4
    mtp_depth: int = 1
    mtp_weight: float = 0.3
    mtp_weight_final: float = 0.1
    mtp_decay_step_fraction: float = 0.6757

    def validate(self) -> None:
        if self.vocab_size < 1 or self.context_length < 1 or self.n_layer < 1:
            raise ValueError("vocab_size, context_length, and n_layer must be positive")
        if self.d_model < 1 or self.n_heads < 1:
            raise ValueError("d_model and n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be even")
        if min(self.q_lora_rank, self.kv_lora_rank, self.qk_nope_head_dim, self.v_head_dim) < 1:
            raise ValueError("MLA dimensions must be positive")
        if self.use_moe and (self.n_routed_experts < 1 or self.n_shared_experts < 1):
            raise ValueError("expert counts must be positive when MoE is enabled")
        if not 0 <= self.n_dense_layers <= self.n_layer:
            raise ValueError("n_dense_layers must be between 0 and n_layer")
        if not 1 <= self.top_k <= self.n_routed_experts:
            raise ValueError("top_k must be between 1 and n_routed_experts")
        if self.expert_hidden_dim < 1 or self.route_scale <= 0:
            raise ValueError("expert_hidden_dim and route_scale must be positive")
        if self.mtp_depth < 0 or self.mtp_depth > 1 or self.mtp_weight < 0:
            raise ValueError("mtp_depth must be 0 or 1 and mtp_weight must be non-negative")
        if self.mtp_weight_final < 0:
            raise ValueError("mtp_weight_final must be non-negative")
        if not 0 <= self.mtp_decay_step_fraction <= 1:
            raise ValueError("mtp_decay_step_fraction must be between 0 and 1")

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def q_content_dim(self) -> int:
        return self.n_heads * self.qk_nope_head_dim

    @property
    def q_rope_dim(self) -> int:
        return self.n_heads * self.qk_rope_head_dim

    @property
    def value_dim(self) -> int:
        return self.n_heads * self.v_head_dim

    def cache_values_per_token(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    def full_kv_values_per_token(self) -> int:
        return self.n_heads * (self.qk_nope_head_dim + self.v_head_dim + self.qk_rope_head_dim)
