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


[Arxiv'24] [Mixtral-8x7B](https://arxiv.org/abs/2401.04088) [[Code](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)]

[Arxiv'24] [Mixtral-8x22B](https://arxiv.org/abs/2401.04088) [[Code](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1)]


[Arxiv'24] [DeepseekMoE](https://arxiv.org/abs/2401.06066) [[Code](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base)]

[Arxiv'24] [DeepSeek-V2](https://arxiv.org/abs/2405.04434) [[Code](https://huggingface.co/deepseek-ai/DeepSeek-V2)]


[Arxiv'24] [PhiMoE](https://arxiv.org/abs/2404.14219) [[Code](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)]

[Arxiv'24] [GRadient-INformed MoE](https://arxiv.org/abs/2409.12136) [[Code](https://huggingface.co/microsoft/GRIN-MoE)]


[Arxiv'24] [Qwen2-57B-A14B](https://arxiv.org/abs/2407.10671) [[Code](https://huggingface.co/Qwen/Qwen2-57B-A14B)]

[QwenBlog'24] [Qwen1.5-MoE](https://qwenlm.github.io/blog/qwen-moe/) [[Code](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)]


[Arxiv'24] [Skywork-MoE](https://arxiv.org/abs/2406.06563) [[Code](https://huggingface.co/Skywork/Skywork-MoE-Base)]

[Arxiv'24] [JetMoE: Reaching Llama2 Performance with 0.1M Dollars](https://arxiv.org/abs/2404.07413)[[Code](https://github.com/myshell-ai/JetMoE)]

[Arxiv'24] [Yuan 2.0-M32](https://arxiv.org/abs/2405.17976) [[Code](https://huggingface.co/IEITYuan/Yuan2-M32-hf)]

[MosaicResearchBlog'24] [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) [[Code](https://huggingface.co/databricks/dbrx-base)]

[SnowflakeBlog'24] [Arctic](https://www.snowflake.com/en/blog/arctic-open-efficient-foundation-language-models-snowflake/) [[Code](https://huggingface.co/Snowflake/snowflake-arctic-base)]

[XAIBlog'24] [Grok-1](https://x.ai/blog/grok-os) [[Code](https://github.com/xai-org/grok-1)]

[Arxiv'24] [Jamba](https://arxiv.org/abs/2403.19887) [[Code](https://huggingface.co/ai21labs/Jamba-v0.1)]

[Arxiv'24] [LLaMA-MoE](https://arxiv.org/abs/2406.16554) [[Code](https://github.com/pjlab-sys4nlp/llama-moe)]

[Arxiv'22] [NLLB-MOE](https://arxiv.org/abs/2207.04672) [[Code](https://huggingface.co/facebook/nllb-moe-54b)]

[ICCV'21] [Swin-MoE](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf) [[Code](https://github.com/microsoft/Swin-Transformer)]

## Model-Level Optimizations

### Efficient Architecture Design

#### Attention Module

[Arxiv'24] [MoH: Multi-Head Attention as Mixture-of-Head Attention](https://arxiv.org/abs/2410.11842) [[Code](https://github.com/SkyworkAI/MoH)]

[Arxiv'24] [Dense Training, Sparse Inference: Rethinking Training of Mixture-of-Experts Language Models](https://arxiv.org/abs/2404.05567)

[Arxiv'24] [JetMoE: Reaching Llama2 Performance with 0.1M Dollars](https://arxiv.org/abs/2404.07413)[[Code](https://github.com/myshell-ai/JetMoE)]

[Arxiv'23] [ModuleFormer: Modularity Emerges from Mixture-of-Experts](https://arxiv.org/abs/2306.04640)

[EMNLP'22] [Mixture of Attention Heads: Selecting Attention Heads Per Token](https://arxiv.org/abs/2210.05144)

[TMM'21] [Enhancing Mixture-of-Experts by Leveraging Attention for Fine-Grained Recognition](https://ieeexplore.ieee.org/abstract/document/9565360)[[Code](https://github.com/lbzhang/Enhanced-Expert-FGVC-Pytorch.git )]


#### MoE Module

[Arxiv'24] [MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts](https://arxiv.org/abs/2410.07348) [[Code](https://github.com/SkyworkAI/MoE-plus-plus)]

[Arxiv'24] [MoELoRA: Contrastive Learning Guided Mixture of Experts on Parameter-Efficient Fine-Tuning for Large Language Models](https://arxiv.org/abs/2402.12851)

[Arxiv'24] [SEER-MoE: Sparse Expert Efficiency through Regularization for Mixture-of-Experts](https://arxiv.org/abs/2404.05089)

[Arxiv'23] [Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference](https://arxiv.org/abs/2308.12066) [[Code](https://github.com/ranggihwang/Pregated_MoE)]

[ICLR'23] [SCoMoE: Efficient Mixtures of Experts with Structured Communication](https://openreview.net/forum?id=s-c96mSU0u5)

[KDD'23] [COMET: Learning Cardinality Constrained Mixture of Experts with Trees and Local Search](https://dl.acm.org/doi/pdf/10.1145/3580305.3599278)

[EMNLP'22] [Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production](https://arxiv.org/abs/2211.10017)



### Model Compression

#### Pruning

[Arxiv'24] [MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router](https://arxiv.org/abs/2410.12013)

[Arxiv'24] [Diversifying the Expert Knowledge for Task-Agnostic Pruning in Sparse Mixture-of-Experts](https://arxiv.org/abs/2407.09590)

[Arxiv'24] [Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs](https://arxiv.org/abs/2407.00945) [[Code](https://github.com/imagination-research/EEP)] 

[EMNLP'24] [LaCo: Large Language Model Pruning via Layer Collapse](https://arxiv.org/abs/2402.11187) [[Code](https://github.com/yangyifei729/LaCo)]

[Arxiv'24] [Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models](https://arxiv.org/abs/2402.14800) [[Code](https://github.com/Lucky-Lance/Expert_Sparsity)] 

[Arxiv'24] [Revisiting SMoE Language Models by Evaluating Inefficiencies with Task Specific Expert Pruning](https://arxiv.org/abs/2409.01483)

[Arxiv'24] [STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](https://arxiv.org/abs/2409.06211)

[Arxiv'24] [Demystifying the Compression of Mixture-of-Experts Through a Unified Framework](https://arxiv.org/abs/2406.02500) [[Code](https://github.com/DaizeDong/Unified-MoE-Compression)]

[Arxiv'24] [A Provably Effective Method for Pruning Experts in Fine-tuned Sparse Mixture-of-Experts](https://arxiv.org/abs/2405.16646)

[Arxiv'22] [Task-Specific Expert Pruning for Sparse Mixture-of-Experts](https://arxiv.org/abs/2206.00277)



#### Quantization
[Arxiv'24] [MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More](https://arxiv.org/abs/2410.06270) [[Code](https://github.com/Aaronhuang-778/MC-MoE)] 

[Arxiv'24] [Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness](https://arxiv.org/abs/2310.02410)

[Arxiv'24] [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](https://arxiv.org/abs/2310.16795) [[Code](http://github.com/IST-DASLab/qmoe)]  

[Arxiv'24] [Mixture of Experts with Mixture of Precisions for Tuning Quality of Service](https://arxiv.org/abs/2407.14417)

[Arxiv'24] [Examining Post-Training Quantization for Mixture-of-Experts: A Benchmark](https://arxiv.org/abs/2406.08155) [[Code](https://github.com/UNITES-Lab/moe-quantization)]

[EMNLP'24] [LaCo: Large Language Model Pruning via Layer Collapse](https://arxiv.org/abs/2402.11187) [[Code](https://github.com/yangyifei729/LaCo)]

[INTERSPEECH'23] [Compressed MoE ASR Model Based on Knowledge Distillation and Quantization](https://www.isca-archive.org/interspeech_2023/yuan23c_interspeech.pdf)


#### Knowledge Distillation

[Arxiv'24] [LaDiMo: Layer-wise Distillation Inspired MoEfier](https://arxiv.org/abs/2408.04278)

[INTERSPEECH'23] [Compressed MoE ASR Model Based on Knowledge Distillation and Quantization](https://www.isca-archive.org/interspeech_2023/yuan23c_interspeech.pdf)


[ICML'22] [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://proceedings.mlr.press/v162/rajbhandari22a.html) [[Code](https://github.com/microsoft/DeepSpeed)]   

[MICROSOFT'22] [Knowledge distillation for mixture of experts models in speech recognition](https://www.microsoft.com/en-us/research/uploads/prod/2022/05/MainzSpeech_Interspeech2022_KD_MoE_Network.pdf)




#### Low Rank Decomposition
[Arxiv'24] [MoE-I2: Compressing Mixture of Experts Models through Inter-Expert Pruning and Intra-Expert Low-Rank Decomposition](https://arxiv.org/abs/2411.01016) [[Code](https://github.com/xiaochengsky/MoEI-2)]

## Algorithm-Level Optimization

### Expert Skip/Adaptive Gating




[Arxiv'24] [AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference](https://arxiv.org/abs/2408.10284) [[Code](https://github.com/PKU-SEC-Lab/AdapMoE)]

[ACL'24] [AXMoE: Sparse Models with Fine-grained and Adaptive Expert Selection](https://aclanthology.org/2024.findings-acl.694/)

[Arxiv'23] [Dynamic Mixture of Experts: An Auto-Tuning Approach for Efficient Transformer Models](https://arxiv.org/abs/2405.14297) [[Code](https://github.com/LINs-lab/DynMoE)]

[Arxiv'23] [Adaptive Gating in Mixture-of-Experts based Language Models](https://arxiv.org/abs/2310.07188)

[Arxiv'23] [Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference](https://arxiv.org/abs/2303.06182)
### Merge Expert

[Arxiv'24] [Retraining-Free Merging of Sparse Mixture-of-Experts via Hierarchical Clustering](https://arxiv.org/pdf/2410.08589)

[EMNLP'23] [Merging Experts into One: Improving Computational Efficiency of Mixture of Experts](https://aclanthology.org/2023.emnlp-main.907.pdf)

[Arxiv'24] [Branch-Train-MiX:Mixing Expert LLMs into a Mixture-of-Experts LLM](https://arxiv.org/pdf/2403.07816)

[Arxiv'22] [Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models](https://arxiv.org/pdf/2208.03306)

[ICLR'24] [Fusing Models with Complementary Expertise](https://openreview.net/pdf?id=PhMrGCMIRL)

### Sparse to Dense

[ACL'24] [XFT: Unlocking the Power of Code Instruction Tuning by Simply Merging Upcycled Mixture-of-Experts](https://aclanthology.org/2024.acl-long.699.pdf)

### Speculative Decoding


## System-Level Optimization

### Expert Parallel




[ASPLOS'25] [FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models](https://shaohuais.github.io/publications/index.html)

[OpenReview'24] [Toward Efficient Inference for Mixture of Experts](https://openreview.net/forum?id=stXtBqyTWX&noteId=p7ADDxdU8g)

[Arxiv'24] [EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference](https://arxiv.org/abs/2410.12247)

[IPDPS'24] [Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference](https://arxiv.org/abs/2401.08383)


[Arxiv'24] [Optimizing Mixture-of-Experts Inference Time Combining Model Deployment and Communication Scheduling](https://arxiv.org/abs/2410.17043)


[Arxiv'24] [WDMoE: Wireless Distributed Large Language Models with Mixture of Experts](https://arxiv.org/abs/2405.03131)

[Arxiv'24] [Lynx: Enabling Efficient MoE Inference through Dynamic Batch-Aware Expert Selection](https://arxiv.org/abs/2411.08982)

[Arxiv'24] [Prediction Is All MoE Needs: Expert Load Distribution Goes from Fluctuating to Stabilizing](https://arxiv.org/abs/2404.16914)

[Arxiv'24] [MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts](https://arxiv.org/abs/2410.07348) [[Code](https://github.com/SkyworkAI/MoE-plus-plus)] [MoE Module Design]


[Arxiv'24] [Shortcut-connected Expert Parallelism for Accelerating Mixture-of-Experts](https://arxiv.org/abs/2404.05019)


[TSC'24] [MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10528887)

[Arxiv'24] [HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy](https://arxiv.org/abs/2411.01288) [[Code](https://github.com/UNITES-Lab/HEXA-MoE)]

[Arxiv'24] [LocMoE: A Low-Overhead MoE for Large Language Model Training](https://arxiv.org/abs/2401.13920)

[Arxiv'24] [Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement](https://arxiv.org/abs/2407.04656)

[TPDS'24] [MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism](https://ieeexplore.ieee.org/abstract/document/10494556)

[INFOCOM'24] [Parm: Efficient Training of Large Sparsely-Activated Models with Dedicated Schedules](https://ieeexplore.ieee.org/abstract/document/10621327)

[EuroSys'24] [ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling](https://dl.acm.org/doi/10.1145/3627703.3650083)


[SIGCOMM'23] [Janus: A Unified Distributed Training Framework for Sparse Mixture-of-Experts Models](https://dl.acm.org/doi/10.1145/3603269.3604869)


[INFOCOM'23] [PipeMoE: Accelerating Mixture-of-Experts through Adaptive Pipelining](https://ieeexplore.ieee.org/abstract/document/10228874)

[ATC'23] [Accelerating Distributed MoE Training and Inference with Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin)

[ATC'23] [SmartMoE: Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization](https://www.usenix.org/conference/atc23/presentation/zhai) [[Code](https://github.com/zms1999/SmartMoE)]

[Arxiv'23] [Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference](https://arxiv.org/abs/2303.06182)



[SIGMOD'23] [FlexMoE: Scaling Large-scale Sparse Pre-trained Model Training via Dynamic Device Placement](https://arxiv.org/abs/2304.03946) [[Code](https://github.com/UNITES-Lab/flex-moe)]


[MLSys'23] [Tutel: Adaptive Mixture-of-Experts at Scale](https://arxiv.org/abs/2206.03382) [[Code](https://github.com/microsoft/tutel)]

[OSDI'23] [Optimizing Dynamic Neural Networks with Brainstorm](https://www.usenix.org/conference/osdi23/presentation/cui) [[Code](https://github.com/Raphael-Hao/brainstorm)]


[ICS'23] [A Hybrid Tensor-Expert-Data Parallelism Approach to Optimize Mixture-of-Experts Training](https://arxiv.org/abs/2303.06318)


[CLUSTER'23] [Prophet: Fine-grained Load Balancing for Parallel Training of Large-scale MoE Models](https://ieeexplore.ieee.org/abstract/document/10319949)

[OSDI'22] [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://www.usenix.org/conference/osdi22/presentation/zheng-lianmin) [[Code](https://github.com/alpa-projects/alpa)]

[NeurIPS'22] [TA-MoE: Topology-Aware Large Scale Mixture-of-Expert Training](https://arxiv.org/abs/2302.09915) [[Code](https://github.com/chen-chang/ta-moe)]


[NeurIPS'22] [Mixture-of-Experts with Expert Choice Routing](https://proceedings.neurips.cc/paper_files/paper/2022/file/2f00ecd787b432c1d36f3de9800728eb-Paper-Conference.pdf)

[PPoPP'22] [FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models](https://dl.acm.org/doi/10.1145/3503221.3508418) [[Code](https://github.com/thu-pacman/FasterMoE)]


[PPoPP'22] [BaGuaLu: targeting brain scale pretrained models with over 37 million cores](https://dl.acm.org/doi/10.1145/3503221.3508417)

[SoCC'22] [Accelerating large-scale distributed neural network training with SPMD parallelism](https://dl.acm.org/doi/10.1145/3542929.3563487)

[PMLR'22] [Gating Dropout: Communication-efficient Regularization for Sparsely Activated Transformers](https://proceedings.mlr.press/v162/liu22g/liu22g.pdf)

[ICML'22] [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://proceedings.mlr.press/v162/rajbhandari22a.html) [[Code](https://github.com/microsoft/DeepSpeed)]   

[Arxiv'22] [HetuMoE: An Efficient Trillion-scale Mixture-of-Expert Distributed Training System](https://arxiv.org/abs/2203.14685) [[Code](https://github.com/PKU-DAIR/Hetu)]


[Arxiv'21] [FastMoE: A Fast Mixture-of-Expert Training System](https://arxiv.org/abs/2103.13262) [[Code](https://github.com/laekov/fastmoe)]

[PMLR'21] [BASE Layers: Simplifying Training of Large, Sparse Models](https://proceedings.mlr.press/v139/lewis21a/lewis21a.pdf) [[Code](https://github.com/pytorch/fairseq/)]

[Arxiv'20] [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)






### Expert Offloading

[Arxiv'24] [ProMoE: Fast MoE-based LLM Serving using Proactive Caching](https://arxiv.org/abs/2410.22134)


[Arxiv'24] [Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design](https://arxiv.org/abs/2410.19123) [[Code](https://github.com/VITA-Group/READ-ME)]


[Arxiv'24] [Shortcut-connected Expert Parallelism for Accelerating Mixture-of-Experts](https://arxiv.org/abs/2404.05019)


[Arxiv'24] [MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs](https://arxiv.org/abs/2411.11217)

[Arxiv'24] [HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference](https://arxiv.org/abs/2411.01433)

[Arxiv'24] [ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inference](https://arxiv.org/abs/2410.17954)

[Arxiv'24] [AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference](https://arxiv.org/abs/2408.10284) [[Code](https://github.com/PKU-SEC-Lab/AdapMoE)] [Adaptive Gating]

[MLSys'24] [SiDA: Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models](https://proceedings.mlsys.org/paper_files/paper/2024/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html) [[Code](https://github.com/timlee0212/SiDA-MoE)]

[Arxiv'24] [MoE-Infinity: Offloading-Efficient MoE Model Serving](https://arxiv.org/abs/2401.14361) [[Code](https://github.com/TorchMoE/MoE-Infinity)]

[Arxiv'24] [Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models](https://arxiv.org/abs/2402.07033) [[Code](https://github.com/efeslab/fiddler)]



[Electronics'24] [Efficient Inference Offloading for Mixture-of-Experts Large Language Models in Internet of Medical Things](https://www.mdpi.com/2079-9292/13/11/2077)

[ISCA'24] [Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference](https://arxiv.org/abs/2308.12066) [[Code](https://github.com/ranggihwang/Pregated_MoE)] [MoE Module]


[HPCA'24] [Enabling Large Dynamic Neural Network Training with Learning-based Memory Management](https://ieeexplore.ieee.org/document/10476398)


[Arxiv'23] [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/abs/2312.17238) [[Code](https://github.com/dvmazur/mixtral-offloading)]

[Arxiv'23] [Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference](https://arxiv.org/abs/2303.06182) [Adaptive Gating]

[Arxiv'23] [EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models](https://arxiv.org/abs/2308.14352) [Quantization]

[Arxiv'23] [SwapMoE: Serving Off-the-shelf MoE-based Large Language Models with Tunable Memory Budget](https://arxiv.org/abs/2308.15030)





## Hareware-Level Optimization

[MICRO'24] [Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching](https://arxiv.org/pdf/2409.01141)

[DAC'24] [MoNDE: Mixture of Near-Data Experts for Large-Scale Sparse Models](https://dl.acm.org/doi/pdf/10.1145/3649329.3655951)

[DAC'24] [FLAME: Fully Leveraging MoE Sparsity for Transformer on FPGA](https://dl.acm.org/doi/pdf/10.1145/3649329.3656507)

[ISSCC’24] [Space-Mate: A 303.5mW Real-Time Sparse Mixture-of-Experts-Based NeRF-SLAM Processor for Mobile Spatial Computing](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10454487)

[ICCAD'23] [Edge-MoE: Memory-Efficient Multi-Task Vision Transformer Architecture with Task-Level Sparsity via Mixture-of-Experts](https://ieeexplore.ieee.org/abstract/document/10323651) [[Code](https://github.com/sharc-lab/Edge-MoE)]


[NeurIPS'22] [M³ViT: Mixture-of-Experts Vision Transformer for Efficient Multi-task Learning with Model-Accelerator Co-design](https://proceedings.neurips.cc/paper_files/paper/2022/file/b653f34d576d1790481e3797cb740214-Paper-Conference.pdf) [[Code](https://github.com/VITA-Group/M3ViT)]

## TODOs

7.	MoEsaic: Shared Mixture of Experts






## Contribute

