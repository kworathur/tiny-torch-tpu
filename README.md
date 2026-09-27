# Tiny TorchTPU

A PyTorch backend for Tensor Processing Units (TPUs) that implements static compilation through PyTorch -> StableHLO -> TPU binary lowering path [1]. This implementation will be tested with a sample RMSNorm operation.


## Motivation

Implementing native device support for TPUs in PyTorch is a significant, yet worthwhile, engineering effort. Recent InferenceX benchmark results from SemiAnalysis show that TPU can achieve up to 50% better perf/$ than B200 and B300 GPUs when serving FP8 models in aggregated mode[2]. Besides cost benefits, TPUs can offer higher throughput when serving open models [3]


## References

[1]C. Basile et al., “TorchTPU: Running PyTorch Natively on TPUs at Google Scale,” Googleblog.com, Apr. 07, 2026. https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/ (accessed Sept. 27, 2026).

[2]A. Ibarra et al., “TPU Inference Externalization Full Steam Ahead - InferenceX,” Semianalysis.com. Accessed: Sep. 27, 2026. [Online]. Available: https://newsletter.semianalysis.com/p/tpu-inferencex-full-steam

[3]G. Novack, X. Liu, J. Ma, and W. Kwon, “700 TPS on Kimi K3: A Case for TPU Megakernels,” Inferact. Accessed: Sep. 27, 2026. [Online]. Available: https://inferact.ai/blog/tpu-megakernels