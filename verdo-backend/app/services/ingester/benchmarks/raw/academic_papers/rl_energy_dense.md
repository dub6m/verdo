# Reinforcement Learning for Energy Optimization

## Abstract
We evaluate reinforcement learning policies for balancing demand response and grid stability. The study compares model-free methods with rule-based baselines.

## Introduction
Energy systems require coordinated control of loads, storage, and generation. Demand response programs shift usage to reduce peak strain.

## Methodology
We model the grid as a Markov decision process with states representing net load, storage level, and renewable output. Actions include charging, discharging, and curtailment. The reward function penalizes peak demand and instability while encouraging renewable utilization.

## Results
Policies trained with proximal policy optimization reduce peak demand by 12% while maintaining frequency stability. Rule-based strategies reduce peaks by 4% but fail under rapid renewable fluctuations.

## Discussion
Dense policy gradients improve performance but require careful tuning of entropy coefficients. Future work explores multi-agent coordination across distributed assets.
