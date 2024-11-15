# FGPAnom

This repository is an official code implementation of the paper entitled ***Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding***.

The paper has been submitted to CVPR 2025.

This repository provides a source code of our proposed pre-training method ***FGP***, and used datasets.

## Datasets

We use three datasets: NAS-Bench-101 [1], NAS-Bench-201 [2], and NAS-Bench-301 [3].

In the following link, we provide our pre-processed version of these datasets.
https://drive.google.com/drive/folders/163H_FbKeeng0rWVtKHiYiEY05qdK6GQX?usp=sharing

## How to obtain the flow surrogate, which is our pre-training objective?

Refer to the following notebook:
```
obtaining_flow_surrogate.ipynb
```

## How to run FGP?

One can reproduce the results by running the following code:
```
python3 main.py
```

## References
- Supervised training code is from (Hwang et al., FlowerFormer: Empowering Neural Architecture Encoding using a Flow-aware Graph Transformer, In CVPR 2024).
- [1] Ying et al., NAS-Bench-101: Towards Reproducible Neural Architecture Search, In ICML 2019.
- [2] Dong et al., NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search, In ICLR 2020.
- [3] Zela et al., Surrogate NAS Benchmarks: Going Beyond the Limited Search Spaces of Tabular NAS Benchmarks, In ICLR 2022.
