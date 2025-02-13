# Awesome MoE LLM Inference System and Algorithm
![Awesome](https://awesome.re/badge.svg)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/JustQJ/awesome-moe-inference/pulls)

A curated list of awesome papers about optimizing the inference of MoE-based LLMs.

Example: [Conference'year] [Paper Title]() [[Code]()]

## Contents


## Survey

[Preprints'24.8] [The Evolution of Mixture of Experts: A Survey from Basics to Breakthroughs](https://www.preprints.org/manuscript/202408.0583/v2)


[Arxiv'24.8] [A Survey on Mixture of Experts](https://arxiv.org/abs/2407.06204) [[Code](https://github.com/withinmiaov/A-Survey-on-Mixture-of-Experts)]

[Arxiv'22] [A Review of Sparse Expert Models in Deep Learning](https://arxiv.org/abs/2209.01667)



## SOTA Open Source MoE LLMs

|                                                             Reference                                                            | Para. | Experts | \#L | \#H | $d_{model}$ | $d_{ffn}$ | $d_{expert}$ | Affiliation |   Time  |
|:--------------------------------------------------------------------------------------------------------------------------------:|:-----:|:-------:|:---:|:---:|:-----------:|:---------:|:------------:|:-----------:|:-------:|
|                           [NLLB](https://huggingface.co/facebook/nllb-moe-54b)                           |  54B  |  2/64/0 |  24 |  16 |     1024    |    8192   |     8192     |   FaceBook  | 2022.07 |
|                      [Qwen2-57B-A14B](https://huggingface.co/Qwen/Qwen2-57B-A14B)                      | 57.4B |  8/64/0 |  28 |  28 |     3584    |   18944   |     2560     |   Alibaba   | 2023.05 |
|                  [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)                 | 46.7B |  2/8/0  |  32 |  32 |     4096    |   14336   |     14336    |  Mistral AI | 2023.12 |
|                       [OpenMoE](https://huggingface.co/OrionZheng/openmoe-base)                       |  34B  |  2/16/0 |  12 |  12 |     768     |    2048   |     2048     |  NUS et al. | 2023.12 |
|              [DeepSeekMoE](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base)              | 16.4B |  6/64/2 |  28 |  16 |     2048    |   10944   |     1408     | DeepSeek-AI | 2024.01 |
|                         [Qwen1.5-MoE](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)                        | 14.3B |  4/60/0 |  24 |  16 |     2048    |    5632   |     1408     |   Alibaba   | 2024.02 |
|                           [JetMoE](https://huggingface.co/jetmoe/jetmoe-8b)                           | 8.52B |  2/8/0  |  24 |  32 |     2048    |    5632   |     5632     |  MIT et al. | 2024.03 |
|                          [Jamba](https://huggingface.co/ai21labs/Jamba-v0.1)                         | 51.6B |  2/16/0 |  32 |  32 |     4096    |   14336   |     14336    |   ai21labs  | 2024.03 |
|                               [DBRX](https://huggingface.co/databricks/dbrx-base)                               |  132B |  4/16/0 |  40 |  48 |     6144    |   10752   |     10752    |  Databricks | 2024.03 |
|                                [Grok-1](https://huggingface.co/xai-org/grok-1)                                |  314B |  2/8/0  |  64 |  48 |     6144    |    UNK    |      UNK     |     xAI     | 2024.03 |
|                        [Arctic](https://huggingface.co/Snowflake/snowflake-arctic-base)                       |  482B | 2/128/0 |  35 |  56 |     7168    |    4864   |     4864     |  Snowflake  | 2024.04 |
|                 [Mixtral-8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1)                |  141B |  2/8/0  |  56 |  48 |     6144    |   16384   |     16384    |  Mistral AI | 2024.04 |
|    [DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)   |  236B | 6/160/2 |  60 | 128 |     5120    |   12288   |     1536     | DeepSeek-AI | 2024.04 |
|                     [Skywork-MoE](https://huggingface.co/Skywork/Skywork-MoE-Base)                    |  13B  |  2/16/0 |  52 |  36 |     4608    |   12288   |     12288    | Kunlun Tech | 2024.05 |
|                           [Yuan2](https://huggingface.co/IEITYuan/Yuan2-M32-hf)                           |  40B  |  2/32/0 |  24 |  16 |     2048    |    8192   |     8192     |  IEIT-Yuan  | 2024.05 |
|                         [LLaMA-MoE](https://github.com/pjlab-sys4nlp/llama-moe)                         |  6.7B |  2/8/0  |  32 |  32 |     4096    |   11008   |     11008    |  Zhu et al. | 2024.06 |
|                     [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924)                    | 6.92B |  8/64/0 |  16 |  16 |     2048    |    1024   |     1024     |   AllenAI   | 2024.07 |
|                      [Phi-3](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)                     | 41.9B |  2/16/0 |  32 |  32 |     4096    |    6400   |     6400     |  MicroSoft  | 2024.08 |
|                           [GRIN-MoE](https://huggingface.co/microsoft/GRIN-MoE)                          | 41.9B |  2/16/0 |  32 |  32 |     4096    |    6400   |     6400     |  MicroSoft  | 2024.09 |
| [Hunyuan-Large](https://huggingface.co/tencent/Tencent-Hunyuan-Large/tree/main/Hunyuan-A52B-Pretrain) |  389B |  1/16/1 |  64 |  80 |     6400    |   18304   |     18304    |   Tencent   | 2024.11 |
|    [DeepSeek-V3](huggingface.co/deepseek-ai/DeepSeek-V3-Base)| 671B | 8/256/1 | 61 | 128 | 7168 | 18432 | 2048 | DeepSeek-AI   | 2024.12 |
|    [MiniMax-Text-01](https://huggingface.co/MiniMaxAI/MiniMax-Text-01)| 456B | 2/32/0 | 80 | 64 | 6144 | 9216 | 9216 | MiniMax-AI   | 2025.1 |
|    [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)| 671B | 8/256/1 | 61 | 128 | 7168 | 18432 | 2048 | DeepSeek-AI   | 2025.1 |


[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf) [[Code](https://huggingface.co/deepseek-ai/DeepSeek-R1)]

[Arxiv'25.1] [MiniMax-Text-01](https://arxiv.org/pdf/2501.08313) [[Code](https://huggingface.co/MiniMaxAI/MiniMax-Text-01)]

[Arxiv'24.11] [Hunyuan-Large](https://arxiv.org/abs/2411.02265) [[Code](https://huggingface.co/tencent/Tencent-Hunyuan-Large)]

[Arxiv'24.1] [Mixtral-8x7B](https://arxiv.org/abs/2401.04088) [[Code](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)]

[Arxiv'24.1] [Mixtral-8x22B](https://arxiv.org/abs/2401.04088) [[Code](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1)]


[Arxiv'24.1] [DeepseekMoE](https://arxiv.org/abs/2401.06066) [[Code](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base)]

[Arxiv'24.6] [DeepSeek-V2](https://arxiv.org/abs/2405.04434) [[Code](https://huggingface.co/deepseek-ai/DeepSeek-V2)]


[Arxiv'24.8] [PhiMoE](https://arxiv.org/abs/2404.14219) [[Code](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)]

[Arxiv'24.9] [GRadient-INformed MoE](https://arxiv.org/abs/2409.12136) [[Code](https://huggingface.co/microsoft/GRIN-MoE)]


[Arxiv'24.9] [Qwen2-57B-A14B](https://arxiv.org/abs/2407.10671) [[Code](https://huggingface.co/Qwen/Qwen2-57B-A14B)]

[QwenBlog'24.3] [Qwen1.5-MoE](https://qwenlm.github.io/blog/qwen-moe/) [[Code](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)]

[Arxiv'24.9] [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060) [[Code](https://github.com/allenai/OLMoE)]

[Arxiv'24.3] [OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2402.01739) [[Code](https://github.com/XueFuzhao/OpenMoE)]

[Arxiv'24.6] [Skywork-MoE](https://arxiv.org/abs/2406.06563) [[Code](https://huggingface.co/Skywork/Skywork-MoE-Base)]

[Arxiv'24.4] [JetMoE: Reaching Llama2 Performance with 0.1M Dollars](https://arxiv.org/abs/2404.07413)[[Code](https://github.com/myshell-ai/JetMoE)]

[Arxiv'24.5] [Yuan 2.0-M32](https://arxiv.org/abs/2405.17976) [[Code](https://huggingface.co/IEITYuan/Yuan2-M32-hf)]

[MosaicResearchBlog'24.3] [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) [[Code](https://huggingface.co/databricks/dbrx-base)]

[SnowflakeBlog'24.4] [Arctic](https://www.snowflake.com/en/blog/arctic-open-efficient-foundation-language-models-snowflake/) [[Code](https://huggingface.co/Snowflake/snowflake-arctic-base)]

[XAIBlog'24.3] [Grok-1](https://x.ai/blog/grok-os) [[Code](https://github.com/xai-org/grok-1)]

[Arxiv'24.7] [Jamba](https://arxiv.org/abs/2403.19887) [[Code](https://huggingface.co/ai21labs/Jamba-v0.1)]

[Arxiv'24.6] [LLaMA-MoE](https://arxiv.org/abs/2406.16554) [[Code](https://github.com/pjlab-sys4nlp/llama-moe)]

[Arxiv'22] [NLLB-MOE](https://arxiv.org/abs/2207.04672) [[Code](https://huggingface.co/facebook/nllb-moe-54b)]

[ICCV'21] [Swin-MoE](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf) [[Code](https://github.com/microsoft/Swin-Transformer)]

## Model-Level Optimizations

### Efficient Architecture Design

#### Attention Module

[Arxiv'24.8] [BAM! Just Like That: Simple and Efficient Parameter Upcycling for Mixture of Experts](https://arxiv.org/abs/2408.08274)

[Arxiv'24.10] [MoH: Multi-Head Attention as Mixture-of-Head Attention](https://arxiv.org/abs/2410.11842) [[Code](https://github.com/SkyworkAI/MoH)]

[Arxiv'24.4] [Dense Training, Sparse Inference: Rethinking Training of Mixture-of-Experts Language Models](https://arxiv.org/abs/2404.05567)

[Arxiv'24.4] [JetMoE: Reaching Llama2 Performance with 0.1M Dollars](https://arxiv.org/abs/2404.07413)[[Code](https://github.com/myshell-ai/JetMoE)]

[NeurIPS'24.10] [MoEUT: Mixture-of-Experts Universal Transformers](https://arxiv.org/abs/2405.16039) [[Code](https://github.com/robertcsordas/moeut)]

[NeurIPS'24.9] [SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention](https://arxiv.org/abs/2312.07987) [[Code](https://github.com/robertcsordas/switchhead)]

[Arxiv'23] [ModuleFormer: Modularity Emerges from Mixture-of-Experts](https://arxiv.org/abs/2306.04640) [[Code](https://github.com/IBM/ModuleFormer)]

[Arxiv'23] [Sparse Universal Transformer](https://arxiv.org/abs/2310.07096)

[EMNLP'22] [Mixture of Attention Heads: Selecting Attention Heads Per Token](https://arxiv.org/abs/2210.05144) [[Code](https://github.com/yikangshen/MoA)]

[ACL'20] [A Mixture of h - 1 Heads is Better than h Heads](https://aclanthology.org/2020.acl-main.587/)

#### MoE Module

[Arxiv'24.10] [MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts](https://arxiv.org/abs/2410.07348) [[Code](https://github.com/SkyworkAI/MoE-plus-plus)]

[Arxiv'24.2] [MoELoRA: Contrastive Learning Guided Mixture of Experts on Parameter-Efficient Fine-Tuning for Large Language Models](https://arxiv.org/abs/2402.12851)



[Arxiv'23] [Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference](https://arxiv.org/abs/2308.12066) [[Code](https://github.com/ranggihwang/Pregated_MoE)]

[ICLR'23] [SCoMoE: Efficient Mixtures of Experts with Structured Communication](https://openreview.net/forum?id=s-c96mSU0u5)

[KDD'23] [COMET: Learning Cardinality Constrained Mixture of Experts with Trees and Local Search](https://dl.acm.org/doi/pdf/10.1145/3580305.3599278)





### Model Compression

#### Pruning

[Arxiv'24.10] [MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router](https://arxiv.org/abs/2410.12013)

[ACL'24] [HyperMoE: Towards Better Mixture of Experts via Transferring Among Experts](https://aclanthology.org/2024.acl-long.571/) [[Code](https://github.com/Bumble666/Hyper_MoE)]

[Arxiv'24.4] [SEER-MoE: Sparse Expert Efficiency through Regularization for Mixture-of-Experts](https://arxiv.org/abs/2404.05089)

[Arxiv'24.10] [Diversifying the Expert Knowledge for Task-Agnostic Pruning in Sparse Mixture-of-Experts](https://arxiv.org/abs/2407.09590)

[Arxiv'24.7] [Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs](https://arxiv.org/abs/2407.00945) [[Code](https://github.com/imagination-research/EEP)] 


[ACL'24.5] [Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models](https://arxiv.org/abs/2402.14800) [[Code](https://github.com/Lucky-Lance/Expert_Sparsity)] 

[Arxiv'24.9] [Revisiting SMoE Language Models by Evaluating Inefficiencies with Task Specific Expert Pruning](https://arxiv.org/abs/2409.01483)

[Arxiv'24.9] [STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](https://arxiv.org/abs/2409.06211)

[Arxiv'24.6] [Demystifying the Compression of Mixture-of-Experts Through a Unified Framework](https://arxiv.org/abs/2406.02500) [[Code](https://github.com/DaizeDong/Unified-MoE-Compression)]

[Arxiv'24.5] [A Provably Effective Method for Pruning Experts in Fine-tuned Sparse Mixture-of-Experts](https://arxiv.org/abs/2405.16646)


[Arxiv'24.11] [MoE-I2: Compressing Mixture of Experts Models through Inter-Expert Pruning and Intra-Expert Low-Rank Decomposition](https://arxiv.org/abs/2411.01016) [[Code](https://github.com/xiaochengsky/MoEI-2)]

[ICLR'24.3] [Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy](https://arxiv.org/abs/2310.01334) [[Code](https://github.com/unites-lab/mc-smoe)]


[Arxiv'23] [ModuleFormer: Modularity Emerges from Mixture-of-Experts](https://arxiv.org/abs/2306.04640) [[Code](https://github.com/IBM/ModuleFormer)]

[Arxiv'22] [Task-Specific Expert Pruning for Sparse Mixture-of-Experts](https://arxiv.org/abs/2206.00277)

[SENSYS '24] [LiteMoE: Customizing On-device LLM Serving via Proxy Submodel Tuning](https://dl.acm.org/doi/abs/10.1145/3666025.3699355)


#### Quantization
[Arxiv'24.10] [MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More](https://arxiv.org/abs/2410.06270) [[Code](https://github.com/Aaronhuang-778/MC-MoE)] 

[Arxiv'23] [Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness](https://arxiv.org/abs/2310.02410)

[Arxiv'23] [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](https://arxiv.org/abs/2310.16795) [[Code](http://github.com/IST-DASLab/qmoe)]  

[Arxiv'24.11] [HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference](https://arxiv.org/abs/2411.01433)




[Arxiv'24.9] [Mixture of Experts with Mixture of Precisions for Tuning Quality of Service](https://arxiv.org/abs/2407.14417)

[Arxiv'24.6] [Examining Post-Training Quantization for Mixture-of-Experts: A Benchmark](https://arxiv.org/abs/2406.08155) [[Code](https://github.com/UNITES-Lab/moe-quantization)]


[INTERSPEECH'23] [Compressed MoE ASR Model Based on Knowledge Distillation and Quantization](https://www.isca-archive.org/interspeech_2023/yuan23c_interspeech.pdf)

[Arxiv'23] [EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models](https://arxiv.org/abs/2308.14352) [Quantization]


[EMNLP'22] [Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production](https://arxiv.org/abs/2211.10017)


#### Knowledge Distillation

[Arxiv'24.10] [LLaVA-MoD: Making LLaVA Tiny via MoE-Knowledge Distillation](https://arxiv.org/abs/2408.15881)

[Arxiv'24.8] [LaDiMo: Layer-wise Distillation Inspired MoEfier](https://arxiv.org/abs/2408.04278)

[INTERSPEECH'23] [Compressed MoE ASR Model Based on Knowledge Distillation and Quantization](https://www.isca-archive.org/interspeech_2023/yuan23c_interspeech.pdf)


[ICML'22] [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://proceedings.mlr.press/v162/rajbhandari22a.html) [[Code](https://github.com/microsoft/DeepSpeed)]   

[MICROSOFT'22] [Knowledge distillation for mixture of experts models in speech recognition](https://www.microsoft.com/en-us/research/uploads/prod/2022/05/MainzSpeech_Interspeech2022_KD_MoE_Network.pdf)

[Arxiv'22] [One Student Knows All Experts Know: From Sparse to Dens](https://arxiv.org/abs/2201.10890)


[JMLR'22] [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)

[Arxiv'21] [Efficient Large Scale Language Modeling with Mixtures of Experts](https://arxiv.org/pdf/2112.10684)


#### Low Rank Decomposition
[Arxiv'24.11] [MoE-I2: Compressing Mixture of Experts Models through Inter-Expert Pruning and Intra-Expert Low-Rank Decomposition](https://arxiv.org/abs/2411.01016) [[Code](https://github.com/xiaochengsky/MoEI-2)]

[ICLR'24.3] [Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy](https://arxiv.org/abs/2310.01334) [[Code](https://github.com/unites-lab/mc-smoe)]

[Arxiv'22] [Parameter-Efficient Mixture-of-Experts Architecture for Pre-trained Language Models](https://arxiv.org/abs/2203.01104) [[Code](https://github.com/RUCAIBox/MPOE)]
### Expert Skip/Adaptive Gating




[Arxiv'24.8] [AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference](https://arxiv.org/abs/2408.10284) [[Code](https://github.com/PKU-SEC-Lab/AdapMoE)]

[ACL'24.8] [XMoE: Sparse Models with Fine-grained and Adaptive Expert Selection](https://aclanthology.org/2024.findings-acl.694/)

[Arxiv'23] [Dynamic Mixture of Experts: An Auto-Tuning Approach for Efficient Transformer Models](https://arxiv.org/abs/2405.14297) [[Code](https://github.com/LINs-lab/DynMoE)]

[Arxiv'23] [Adaptive Gating in Mixture-of-Experts based Language Models](https://arxiv.org/abs/2310.07188)

[Arxiv'23] [Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference](https://arxiv.org/abs/2303.06182)

[Arxiv'24.8] [AdaMoLE: Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts](https://arxiv.org/abs/2405.00361)

[ICCV'23] [AdaMV-MoE: Adaptive Multi-Task Vision Mixture-of-Experts](https://ieeexplore.ieee.org/document/10377734)

### Merge Expert

[Arxiv'24.10] [Retraining-Free Merging of Sparse Mixture-of-Experts via Hierarchical Clustering](https://arxiv.org/abs/2410.08589)

[EMNLP'23] [Merging Experts into One: Improving Computational Efficiency of Mixture of Experts](https://aclanthology.org/2023.emnlp-main.907.pdf)

[Arxiv'24.3] [Branch-Train-MiX:Mixing Expert LLMs into a Mixture-of-Experts LLM](https://arxiv.org/abs/2403.07816)

[Arxiv'22] [Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models](https://arxiv.org/abs/2208.03306)

[ICLR'24.5] [Fusing Models with Complementary Expertise](https://openreview.net/pdf?id=PhMrGCMIRL)

[Arxiv'24.5] [Learning More Generalized Experts by Merging Experts in Mixture-of-Experts](https://arxiv.org/abs/2405.11530)

[Arxiv'24.9] [DA-MoE: Towards Dynamic Expert Allocation for Mixture-of-Experts Models](https://arxiv.org/abs/2409.06669)

### Sparse to Dense

[ACL'24.6] [XFT: Unlocking the Power of Code Instruction Tuning by Simply Merging Upcycled Mixture-of-Experts](https://aclanthology.org/2024.acl-long.699.pdf)

[Arxiv'23] [Moduleformer: Learning modular large language models from uncurated data](https://arxiv.org/abs/2306.04640)


[Arxiv'23] [Experts weights averaging: A new general training scheme for vision transformers](https://arxiv.org/pdf/2308.06093)

[JMLR'22] [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)

[Arxiv'22] [One student knows all experts know: From sparse to dense](https://arxiv.org/abs/2201.10890)

[Arxiv'22] [Task-specific expert pruning for sparse mixture-of experts](https://arxiv.org/abs/2206.00277)

[Arxiv'21] [Efficient Large Scale Language Modeling with Mixtures of Experts](https://arxiv.org/pdf/2112.10684)




## System-Level Optimization

### Expert Parallel

[Arxiv'25.1] [Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing](https://arxiv.org/abs/2501.05313)


[ASPLOS'25] [FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models](https://shaohuais.github.io/publications/index.html)

[OpenReview'24.11] [Toward Efficient Inference for Mixture of Experts](https://openreview.net/forum?id=stXtBqyTWX&noteId=p7ADDxdU8g)

[Arxiv'24.10] [EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference](https://arxiv.org/abs/2410.12247)

[IPDPS'24.1] [Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference](https://arxiv.org/abs/2401.08383)


[Arxiv'24.10] [Optimizing Mixture-of-Experts Inference Time Combining Model Deployment and Communication Scheduling](https://arxiv.org/abs/2410.17043)


[IEEE'24.5] [WDMoE: Wireless Distributed Large Language Models with Mixture of Experts](https://arxiv.org/abs/2405.03131)

[Arxiv'24.11] [Lynx: Enabling Efficient MoE Inference through Dynamic Batch-Aware Expert Selection](https://arxiv.org/abs/2411.08982)

[Arxiv'24.4] [Prediction Is All MoE Needs: Expert Load Distribution Goes from Fluctuating to Stabilizing](https://arxiv.org/abs/2404.16914)

[Arxiv'24.10] [MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts](https://arxiv.org/abs/2410.07348) [[Code](https://github.com/SkyworkAI/MoE-plus-plus)] [MoE Module Design]


[Arxiv'24.11] [Shortcut-connected Expert Parallelism for Accelerating Mixture-of-Experts](https://arxiv.org/abs/2404.05019)


[TSC'24.5] [MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10528887)

[Arxiv'24.11] [HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy](https://arxiv.org/abs/2411.01288) [[Code](https://github.com/UNITES-Lab/HEXA-MoE)]

[Arxiv'24.5] [LocMoE: A Low-Overhead MoE for Large Language Model Training](https://arxiv.org/abs/2401.13920)

[Arxiv'24.7] [Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement](https://arxiv.org/abs/2407.04656)

[Arxiv'24.10] [Scattered Mixture-of-Experts Implementation](https://arxiv.org/abs/2403.08245) [[Code](https://github.com/shawntan/scattermoe)]


[TPDS'24.4] [MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism](https://ieeexplore.ieee.org/abstract/document/10494556)

[INFOCOM'24.5] [Parm: Efficient Training of Large Sparsely-Activated Models with Dedicated Schedules](https://ieeexplore.ieee.org/abstract/document/10621327)

[EuroSys'24.4] [ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling](https://dl.acm.org/doi/10.1145/3627703.3650083)


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

[Arxiv'25.02][fMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving](https://www.arxiv.org/abs/2502.05370)

[Arxiv'24.12] [DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference](https://arxiv.org/abs/2501.10375)

[Arxiv'24.11] [Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference](https://arxiv.org/abs/2412.00099)

[Arxiv'24.10] [ProMoE: Fast MoE-based LLM Serving using Proactive Caching](https://arxiv.org/abs/2410.22134)


[NeurIPS'24.10] [Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design](https://arxiv.org/abs/2410.19123) [[Code](https://github.com/VITA-Group/READ-ME)]


[Arxiv'24.11] [Shortcut-connected Expert Parallelism for Accelerating Mixture-of-Experts](https://arxiv.org/abs/2404.05019)


[Arxiv'24.11] [MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs](https://arxiv.org/abs/2411.11217)

[Arxiv'24.11] [HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference](https://arxiv.org/abs/2411.01433) [Quantization, Skip Expert]

[Arxiv'24.10] [ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inference](https://arxiv.org/abs/2410.17954)

[Arxiv'24.8] [AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference](https://arxiv.org/abs/2408.10284) [[Code](https://github.com/PKU-SEC-Lab/AdapMoE)] [Adaptive Gating]


[Arxiv'24.9] [Mixture of Experts with Mixture of Precisions for Tuning Quality of Service](https://arxiv.org/abs/2407.14417)

[MLSys'24.5] [SiDA: Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models](https://proceedings.mlsys.org/paper_files/paper/2024/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html) [[Code](https://github.com/timlee0212/SiDA-MoE)]

[Arxiv'24.8] [MoE-Infinity: Offloading-Efficient MoE Model Serving](https://arxiv.org/abs/2401.14361) [[Code](https://github.com/TorchMoE/MoE-Infinity)]

[Arxiv'24.2] [Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models](https://arxiv.org/abs/2402.07033) [[Code](https://github.com/efeslab/fiddler)]

[Arxiv'24.9] [Mixture of Experts with Mixture of Precisions for Tuning Quality of Service](https://arxiv.org/abs/2407.14417)

[Electronics'24.5] [Efficient Inference Offloading for Mixture-of-Experts Large Language Models in Internet of Medical Things](https://www.mdpi.com/2079-9292/13/11/2077)

[ISCA'24.4] [Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference](https://arxiv.org/abs/2308.12066) [[Code](https://github.com/ranggihwang/Pregated_MoE)] [MoE Module]


[HPCA'24.3] [Enabling Large Dynamic Neural Network Training with Learning-based Memory Management](https://ieeexplore.ieee.org/document/10476398)

[SC'24.11] [APTMoE: Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes](https://dl.acm.org/doi/10.1109/SC41406.2024.00096)

[Arxiv'23] [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/abs/2312.17238) [[Code](https://github.com/dvmazur/mixtral-offloading)]

[Arxiv'23] [Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference](https://arxiv.org/abs/2303.06182) [Adaptive Gating]

[Arxiv'23] [EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models](https://arxiv.org/abs/2308.14352) [Quantization]

[ACL'24.5] [SwapMoE: Serving Off-the-shelf MoE-based Large Language Models with Tunable Memory Budget](https://arxiv.org/abs/2308.15030)


### Others
[SoCC '24.11] [MoEsaic: Shared Mixture of Experts]


## Hareware-Level Optimization

[MICRO'24.9] [Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching](https://arxiv.org/abs/2409.01141)

[DAC'24.5] [MoNDE: Mixture of Near-Data Experts for Large-Scale Sparse Models](https://dl.acm.org/doi/pdf/10.1145/3649329.3655951)

[DAC'24.11] [FLAME: Fully Leveraging MoE Sparsity for Transformer on FPGA](https://dl.acm.org/doi/pdf/10.1145/3649329.3656507)

[ISSCC’24.2] [Space-Mate: A 303.5mW Real-Time Sparse Mixture-of-Experts-Based NeRF-SLAM Processor for Mobile Spatial Computing](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10454487)

[ICCAD'23] [Edge-MoE: Memory-Efficient Multi-Task Vision Transformer Architecture with Task-Level Sparsity via Mixture-of-Experts](https://ieeexplore.ieee.org/abstract/document/10323651) [[Code](https://github.com/sharc-lab/Edge-MoE)]


[NeurIPS'22] [M³ViT: Mixture-of-Experts Vision Transformer for Efficient Multi-task Learning with Model-Accelerator Co-design](https://proceedings.neurips.cc/paper_files/paper/2022/file/b653f34d576d1790481e3797cb740214-Paper-Conference.pdf) [[Code](https://github.com/VITA-Group/M3ViT)]


## Citation

If you find this repo useful, please cite our paper:

```
@misc{liu2024moeinf,
      title={A Survey on Inference Optimization Techniques for Mixture of Experts Models}, 
      author={Jiacheng Liu and Peng Tang and Wenfeng Wang and Yuhang Ren and Xiaofeng Hou and Pheng-Ann Heng and Minyi Guo and Chao Li},
      year={2024},
      archivePrefix={arXiv},
}
```
