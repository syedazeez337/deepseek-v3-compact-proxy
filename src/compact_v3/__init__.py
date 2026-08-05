"""A mechanism-faithful compact proxy for DeepSeek-V3's architecture.

Multi-head Latent Attention, DeepSeekMoE (shared + fine-grained routed experts
with auxiliary-loss-free load balancing), and Multi-Token Prediction, sized to
train on a single consumer GPU.
"""
