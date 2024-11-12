# Awesome MoE LLM Inference System and Algorithm
![Awesome](https://awesome.re/badge.svg)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/JustQJ/awesome-moe-inference/pulls)

A curated list of awesome papers about optimizing the inference of MoE-based LLMs.

Example: [Conference'year] [Paper Title]() [[Code]()]

## Contents


## Survey

[Preprints'24] [The Evolution of Mixture of Experts: A Survey from Basics to Breakthroughs](https://www.preprints.org/manuscript/202408.0583/v2)


[Arxiv'24] [A Survey on Mixture of Experts](https://arxiv.org/abs/2407.06204) [[Code](https://github.com/withinmiaov/A-Survey-on-Mixture-of-Experts)]

[Arxiv'22] [A Review of Sparse Expert Models in Deep Learning](https://arxiv.org/abs/2209.01667)



## SOTA MoE LLMs




## Model-Level Optimizations

### Efficient Architecture Design

#### Attention Module
[TMM'21] [Enhancing Mixture-of-Experts by Leveraging Attention for Fine-Grained Recognition](https://ieeexplore.ieee.org/abstract/document/9565360)[[Code](https://github.com/lbzhang/Enhanced-Expert-FGVC-Pytorch.git )]

#### MoE Module
[Arxiv'23] [Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference](https://arxiv.org/abs/2308.12066) [[Code](https://github.com/ranggihwang/Pregated_MoE)]


### Model Compression

#### Pruning

[Arxiv'24] [MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router](https://arxiv.org/abs/2410.12013)

[Arxiv'24] [Diversifying the Expert Knowledge for Task-Agnostic Pruning in Sparse Mixture-of-Experts](https://arxiv.org/abs/2407.09590)

[Arxiv'24] [Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs](https://arxiv.org/abs/2407.00945) [[Code](https://github.com/imagination-research/EEP)] 

#### Quantization
[Arxiv'24] [MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More](https://arxiv.org/abs/2410.06270) [[Code](https://github.com/Aaronhuang-778/MC-MoE)] 

[Arxiv'24] [Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness](https://arxiv.org/abs/2310.02410)

[Arxiv'24] [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](https://arxiv.org/abs/2310.16795) [[Code](http://github.com/IST-DASLab/qmoe)]  

[Arxiv'24] [Mixture of Experts with Mixture of Precisions for Tuning Quality of Service](https://arxiv.org/abs/2407.14417)

#### Knowledge Distillation

[Arxiv'24] [LaDiMo: Layer-wise Distillation Inspired MoEfier](https://arxiv.org/abs/2408.04278)


[ICML'22] [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://proceedings.mlr.press/v162/rajbhandari22a.html) [[Code](https://github.com/microsoft/DeepSpeed)]   




#### Low Rank Decomposition

## Algorithm-Level Optimization

### Expert Skip/Adaptive Gating




[Arxiv'24] [AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference](https://arxiv.org/abs/2408.10284) [[Code](https://github.com/PKU-SEC-Lab/AdapMoE)]

[ACL'24] [AXMoE: Sparse Models with Fine-grained and Adaptive Expert Selection](https://aclanthology.org/2024.findings-acl.694/)

[Arxiv'23] [Dynamic Mixture of Experts: An Auto-Tuning Approach for Efficient Transformer Models](https://arxiv.org/abs/2405.14297) [[Code](https://github.com/LINs-lab/DynMoE)]

[Arxiv'23] [Adaptive Gating in Mixture-of-Experts based Language Models](https://arxiv.org/abs/2310.07188)



### Speculative Decoding


## System-Level Optimization

### Expert Parallel

#### Inference Only
[Arxiv'24] [EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference](https://arxiv.org/abs/2410.12247)

[Arxiv'24] [Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference](https://arxiv.org/abs/2401.08383)


[Arxiv'24] [Optimizing Mixture-of-Experts Inference Time Combining Model Deployment and Communication Scheduling](https://arxiv.org/abs/2410.17043)

#### Inference and Training


[ASPLOS'25] [FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models](https://shaohuais.github.io/publications/index.html)


[Arxiv'24] [HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy](https://arxiv.org/abs/2411.01288) [[Code](https://github.com/UNITES-Lab/HEXA-MoE)]

[Arxiv'24] [LocMoE: A Low-Overhead MoE for Large Language Model Training](https://arxiv.org/abs/2401.13920)

[TPDS'24] [MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism](https://ieeexplore.ieee.org/abstract/document/10494556)

[INFOCOM'24] [Parm: Efficient Training of Large Sparsely-Activated Models with Dedicated Schedules](https://ieeexplore.ieee.org/abstract/document/10621327)

[EuroSys'24] [ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling](https://dl.acm.org/doi/10.1145/3627703.3650083)


[SIGCOMM'23] [Janus: A Unified Distributed Training Framework for Sparse Mixture-of-Experts Models](https://dl.acm.org/doi/10.1145/3603269.3604869)


[INFOCOM'23] [PipeMoE: Accelerating Mixture-of-Experts through Adaptive Pipelining](https://ieeexplore.ieee.org/abstract/document/10228874)

[ATC'23] [Accelerating Distributed MoE Training and Inference with Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin)

[ATC'23] [SmartMoE: Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization](https://www.usenix.org/conference/atc23/presentation/zhai) [[Code](https://github.com/zms1999/SmartMoE)]

[Arxiv'23] [TA-MoE: Topology-Aware Large Scale Mixture-of-Expert Training](https://arxiv.org/abs/2302.09915) [[Code](https://github.com/chen-chang/ta-moe)]


[Arxiv'23] [FlexMoE: Scaling Large-scale Sparse Pre-trained Model Training via Dynamic Device Placement](https://arxiv.org/abs/2304.03946) [[Code](https://github.com/UNITES-Lab/flex-moe)]


[PPoPP'22] [FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models](https://dl.acm.org/doi/10.1145/3503221.3508418) [[Code](https://github.com/thu-pacman/FasterMoE)]


[Arxiv'22] [Tutel: Adaptive Mixture-of-Experts at Scale](https://arxiv.org/abs/2206.03382) [[Code](https://github.com/microsoft/tutel)]

[Arxiv'22] [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)


[Arxiv'22] [HetuMoE: An Efficient Trillion-scale Mixture-of-Expert Distributed Training System](https://arxiv.org/abs/2203.14685) [[Code](https://github.com/PKU-DAIR/Hetu)]


[Arxiv'21] [FastMoE: A Fast Mixture-of-Expert Training System](https://arxiv.org/abs/2103.13262) [[Code](https://github.com/laekov/fastmoe)]

[KDD'23] [COMET: Learning Cardinality Constrained Mixture of Experts with Trees and Local Search](https://dl.acm.org/doi/pdf/10.1145/3580305.3599278)

### Expert Offloading

[Arxiv'24] [HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference](https://arxiv.org/abs/2411.01433)

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

[Arxiv'24] [WDMoE: Wireless Distributed Large Language Models with Mixture of Experts](https://arxiv.org/abs/2405.03131)
### Load balancing/scheduling

[Arxiv'24] [Prediction Is All MoE Needs: Expert Load Distribution Goes from Fluctuating to Stabilizing](https://arxiv.org/abs/2404.16914)


[CLUSTER'23] [Prophet: Fine-grained Load Balancing for Parallel Training of Large-scale MoE Models](https://ieeexplore.ieee.org/abstract/document/10319949)

[Arxiv'24] [Shortcut-connected Expert Parallelism for Accelerating Mixture-of-Experts](https://arxiv.org/abs/2404.05019)
## Hareware-Level Optimization
[ICCAD'23] [Edge-MoE: Memory-Efficient Multi-Task Vision Transformer Architecture with Task-Level Sparsity via Mixture-of-Experts](https://ieeexplore.ieee.org/abstract/document/10323651) [[Code](https://github.com/sharc-lab/Edge-MoE)]

[NeurIPS'22] [M³ViT: Mixture-of-Experts Vision Transformer for Efficient Multi-task Learning with Model-Accelerator Co-design](https://proceedings.neurips.cc/paper_files/paper/2022/file/b653f34d576d1790481e3797cb740214-Paper-Conference.pdf) [[[Code](https://github.com/VITA-Group/M3ViT)]

[Arxiv'24] [Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching](https://arxiv.org/pdf/2409.01141)

## Contribute

