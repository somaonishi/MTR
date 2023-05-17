# Rethinking Data Augmentation for Tabular Data in Deep Learning (NeurIPS 2023)
The official implementation of the paper "Rethinking Data Augmentation for Tabular Data in Deep Learning".
In this README, we provide information about the environment we used and instructions on how to run the code to reproduce our experiments.

Feel free to report [issues]().

## Environment
The experiments in our paper were conducted using the following environment:

- Operating System: Ubuntu 22.04.1 LTS
- CUDA compiler version: 11.7
- Python 3.10.6

## Installation
Use [poetry](https://python-poetry.org/) to create an python environment and activate it.

```bash
poetry install
poetry shell
```

## Running Experiments
Before starting the experiment you need to download Adult data from https://www.kaggle.com/datasets/wenruliu/adult-income-dataset. Please place the downloaded `adult.csv` under `datasets/Adult/raw/`. You can change the location of the `datasets/` directory by changing the `data_dir` in `conf/config.yaml`.

### Supervised Learning
For example, to run supervised learning with MTR on the Adult data set, use the following command:

```bash
python main.py train_mode=supervised data=Adult model=fttrans/mask_token  seed="range(1,30)" model.params.mask_ratio=0.1,0.2,0.3,0.4,0.5,0.6,0.7 model.params.bias_after_mask=false -m 
```

If you want to replicate our experiment, use the following command:
```bash
python script/sl/run_all.py 41143,44,41145,287,4538,45062,45060,45012,CAHousing,1461,Adult,41166,1597 1 10
```

### Self Supervised Learning
For example, to run self-supervised learning with MTR on the Adult data set, use the following command:

```bash
python main.py train_mode=self_sl data=Adult model=fttrans/mask_token model.trainer=FTTransMaskTokenSSLTrainer seed="range(1,30)" model.params.mask_ratio=0.1,0.2,0.3,0.4,0.5,0.6,0.7 model.params.bias_after_mask=false -m
```


If you want to replicate our experiment, use the following command:
```bash
python script/self_sl/run_all.py 41143,44,41145,287,4538,45062,45060,45012,CAHousing,1461,Adult,41166,1597 0.25 10
```
