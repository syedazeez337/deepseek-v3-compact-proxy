from dataclasses import dataclass, fields
from typing import Any

# Values these fields had before the gate that introduced them. A checkpoint
# saved before a field existed does not record it, so rebuilding its config
# from `model_config` would silently substitute today's default. For
# `n_dense_layers` that changes the architecture (block 0 becomes dense instead
# of MoE) and the checkpoint stops loading at all. Gates I-M were unloadable
# from Gate N onward for exactly this reason.
LEGACY_FIELD_DEFAULTS: dict[str, Any] = {
    "n_dense_layers": 0,          # added in Gate N; every layer was MoE before
    "mtp_weight_final": None,     # added in Gate Q; resolved to mtp_weight (no annealing existed)
    "mtp_decay_step_fraction": 1.0,  # added in Gate Q; 1.0 means the switch never fires
}


def config_from_checkpoint(stored: dict[str, Any]) -> "CompactV3Config":
    """Rebuild a config from a checkpoint, honouring the era it was saved in.

    Absent fields take the value they effectively had when the checkpoint was
    written, not the current default.
    """
    known = {field.name for field in fields(CompactV3Config)}
    resolved = {key: value for key, value in stored.items() if key in known}
    for name, legacy in LEGACY_FIELD_DEFAULTS.items():
        if name in resolved:
            continue
        resolved[name] = resolved.get("mtp_weight", 0.3) if legacy is None else legacy
    return CompactV3Config(**resolved)


@dataclass(frozen=True)
class CompactV3Config:
    vocab_size: int = 32_000
    context_length: int = 520
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


__all__ = ["CompactV3Config", "LEGACY_FIELD_DEFAULTS", "config_from_checkpoint"]
