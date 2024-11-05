# Awesome MoE LLM Inference System and Algorithm
![Awesome](https://awesome.re/badge.svg)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/JustQJ/awesome-moe-inference/pulls)

A curated list of awesome papers about optimizing the inference of MoE-based LLMs.

Example: [Conference'year] [Paper Title]() [[Code]()]

## Contents


## Survey
[Arxiv'24] [A Survey on Mixture of Experts](https://arxiv.org/abs/2407.06204) [[Code](https://github.com/withinmiaov/A-Survey-on-Mixture-of-Experts)]

[Arxiv'22] [A Review of Sparse Expert Models in Deep Learning](https://arxiv.org/abs/2209.01667)

## SOTA MoE LLMs




## Model-Level Optimizations

### Efficient Architecture Design

#### Attention Module

#### MoE Module
[Arxiv'23] [Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference](https://arxiv.org/abs/2308.12066) [[Code](https://github.com/ranggihwang/Pregated_MoE)]


### Model Compression

#### Pruning

[Arxiv'24] [MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router](https://arxiv.org/abs/2410.12013)

[Arxiv'24] [Diversifying the Expert Knowledge for Task-Agnostic Pruning in Sparse Mixture-of-Experts](https://arxiv.org/abs/2407.09590)


#### Quantization
[Arxiv'24] [MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More](https://arxiv.org/abs/2410.06270) [[Code](https://github.com/Aaronhuang-778/MC-MoE)] 

[Arxiv'24] [Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness](https://arxiv.org/abs/2310.02410)

[Arxiv'24] [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](https://arxiv.org/abs/2310.16795) [[Code](http://github.com/IST-DASLab/qmoe)]  

[Arxiv'24] [Mixture of Experts with Mixture of Precisions for Tuning Quality of Service](https://arxiv.org/abs/2407.14417)

#### Knowledge Distillation
[ICML'22] [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://proceedings.mlr.press/v162/rajbhandari22a.html) [[Code](https://github.com/microsoft/DeepSpeed)]   

#### Low Rank Decomposition

## Algorithm-Level Optimization

### Expert Skip/Adaptive Gating

[Arxiv'24] [AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference](https://arxiv.org/abs/2408.10284) [[Code](https://github.com/PKU-SEC-Lab/AdapMoE)]

[Arxiv'23] [Adaptive Gating in Mixture-of-Experts based Language Models](https://arxiv.org/abs/2310.07188)



### Speculative Decoding


## System-Level Optimization

### Expert Parallel

#### Inference Only
[Arxiv'24] [EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference](https://arxiv.org/abs/2410.12247)
#### Inference and Training
[Arxiv'22] [MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services](https://arxiv.org/abs/2205.10034)

### Expert Offloading



[Arxiv'24] [ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inference](https://arxiv.org/abs/2410.17954)

[Arxiv'24] [AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference](https://arxiv.org/abs/2408.10284) [[Code](https://github.com/PKU-SEC-Lab/AdapMoE)]

[MLSys'24] [SiDA: Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models](https://proceedings.mlsys.org/paper_files/paper/2024/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html) [[Code](https://github.com/timlee0212/SiDA-MoE)]

[Arxiv'24] [MoE-Infinity: Offloading-Efficient MoE Model Serving](https://arxiv.org/abs/2401.14361) [[Code](https://github.com/TorchMoE/MoE-Infinity)]

[Arxiv'24] [Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models](https://arxiv.org/abs/2402.07033) [[Code](https://github.com/efeslab/fiddler)]


[Electronics'24] [Efficient Inference Offloading for Mixture-of-Experts Large Language Models in Internet of Medical Things](https://www.mdpi.com/2079-9292/13/11/2077)

[Arxiv'23] [Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference](https://arxiv.org/abs/2308.12066) [[Code](https://github.com/ranggihwang/Pregated_MoE)]


[Arxiv'23] [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/abs/2312.17238) [[Code](https://github.com/dvmazur/mixtral-offloading)]

[Arxiv'23] [Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference](https://arxiv.org/abs/2303.06182)

[Arxiv'23] [EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models](https://arxiv.org/abs/2308.14352)

[Arxiv'23] [SwapMoE: Serving Off-the-shelf MoE-based Large Language Models with Tunable Memory Budget](https://arxiv.org/abs/2308.15030)




### Load balancing/scheduling


## Hareware-Level Optimization
[ICCAD'23] [Edge-MoE: Memory-Efficient Multi-Task Vision Transformer Architecture with Task-Level Sparsity via Mixture-of-Experts](https://ieeexplore.ieee.org/abstract/document/10323651) [[Code](https://github.com/sharc-lab/Edge-MoE)]


## Contribute

