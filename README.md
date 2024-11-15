# FGPAnom

This repository is an official code implementation of the paper entitled ***Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding***.

The paper has been submitted to CVPR 2025.

This repository provides a source code of our proposed pre-training method ***FGP***, and used datasets.

## Datasets

We use three datasets: NAS-Bench-101 [1], NAS-Bench-201 [2], and NAS-Bench-301 [3].

### Link
In the following link, we provide our pre-processed version of these datasets.
https://drive.google.com/drive/folders/163H_FbKeeng0rWVtKHiYiEY05qdK6GQX?usp=sharing

### Usage
1. Please make the folder named ```./dataset``` in the same hierarchy with ```main.py```
2. Then, please put the datasets in ```./dataset``` folder.

## How to obtain the flow surrogate, which is our pre-training objective?

Refer to the following notebook:
```
obtaining_flow_surrogate.ipynb
```

## How to run FGP?

### Running main.py

One can reproduce the performance prediction results by running the following code:
```
python3 main.py -dname nb101 -device cuda:0 -train_ratio 0.01 -proj_layer 2 -proj_dim1 128 -proj_dim2 64 -ssl_lr 0.001 -wdecay 1e-6 -lamda1 0.5 -lamda2 0.5 -enc gatedgcn
```

### Hyperparameters

Each argument corresponds to each hyperparameter, which is as follows:
- ```-dname``` is a ```str``` that indicates the data name, which should be one of {```nb101, nb201, nb301```}.
- ```-device``` is a ```str``` that indicates the data type that indicates GPU device number, such as ```cuda:0```.
- ```train_ratio``` is a ```float``` that indicates the ratio of the training set to be used for model fine-tuning, which should be within $(0,1]$, such as ```0.01```.
- ```proj_layer``` is a ```int``` that indicates the number of layers of the projection head, such as ```2```.
- ```proj_dim1``` is a ```int``` that indicates the hidden dimension of the surrogate projection head, such as ```128```.
- ```proj_dim2``` is a ```int``` that indicates the hidden dimension of the zero-cost proxy projection head, such as ```64```.
- ```ssl_lr``` is a ```float``` that indicates the pre-training learning rate, such as ```1e-3```.
- ```wdecay``` is a ```float``` that indicates the pre-training weight decay, such as ```1e-6```.
- ```lamda1``` is a ```float``` that indicates the loss coefficient of the surrogate reconstruction loss, such as ```0.5```.
- ```lamda2``` is a ```float``` that indicates the loss coefficient of the zero-cost proxy prediction loss, such as ```0.5```.
- ```enc``` is a ```str``` that indicates the neural architecture encoder type, which should be one of {```gatedgcn, gin, flowerformer```}.

## References
- Supervised training code is from (Hwang et al., FlowerFormer: Empowering Neural Architecture Encoding using a Flow-aware Graph Transformer, In CVPR 2024).
- [1] Ying et al., NAS-Bench-101: Towards Reproducible Neural Architecture Search, In ICML 2019.
- [2] Dong et al., NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search, In ICLR 2020.
- [3] Zela et al., Surrogate NAS Benchmarks: Going Beyond the Limited Search Spaces of Tabular NAS Benchmarks, In ICLR 2022.
