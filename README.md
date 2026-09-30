# Tiny TorchTPU

This repo contains my attempt at building a torch.compile backend for Tensor Processing Units (TPUs) that implements static compilation through PyTorch -> StableHLO -> TPU binary lowering path. TorchTPU is the name of a production implementation of this lowering path + a native TPU device backend in PyTorch that is currently under development [1].


## Motivation

Tensor processing units (TPUs) are accelerators that can unlock more cost effective production LLM inference [2]. To realize this goal, we seek a way to connect the TPU compiler (XLA) and runtime with popular ML frameworks such as PyTorch. The TorchTPU blog post from Google points out a key tradeoff that I aim to explore with this project: ease of use vs. performance when designing an inference stack. PyTorch provides a good debugging experience through its eager execution, which can be at odds with static graph compilation exemplified by Google's own ML framework, Tensorflow (which predates PyTorch but has a smaller share of users).    
## References

[1]C. Basile et al., “TorchTPU: Running PyTorch Natively on TPUs at Google Scale,” Googleblog.com, Apr. 07, 2026. https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/ (accessed Sept. 27, 2026).

[2]A. Ibarra et al., “TPU Inference Externalization Full Steam Ahead - InferenceX,” Semianalysis.com. Accessed: Sep. 27, 2026. [Online]. Available: https://newsletter.semianalysis.com/p/tpu-inferencex-full-steam

[3]G. Novack, X. Liu, J. Ma, and W. Kwon, “700 TPS on Kimi K3: A Case for TPU Megakernels,” Inferact. Accessed: Sep. 27, 2026. [Online]. Available: https://inferact.ai/blog/tpu-megakernels