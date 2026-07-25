# Mixed-Precision RL

This project investigates numerically stable mixed-precision training for
tabula-rasa reinforcement learning. The initial goal is to run compute in BF16
while matching the performance of an all-FP32 baseline. Later experiments may
extend this to FP8 and NVFP4.

The first experimental setting is PPO on the DeepMind Control Suite through
MuJoCo Playground, with the full training stack implemented in JAX.

Treat reproducibility against the FP32 baseline as the primary success
criterion: lower-precision changes should preserve learning performance, and
their numerical stability should be measured rather than assumed.
