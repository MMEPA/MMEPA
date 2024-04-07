# MMEPA: Mixture of Multimodal Experts with Parallel Adapters for Sentiment Analysis

This repository contains code of MMEPA: Mixture of Multimodal Experts with Parallel Adapters for Sentiment Analysis

## Introduction

MMEPA is a plug-and-play module, which can be flexibly applied to various pre-trained language models and directly transform these models into a multi-modal model that can handle MSA tasks.


## Usage

1. Download the word-aligned CMU-MOSI dataset from [MMSA](https://github.com/thuiar/MMSA). Download the pre-trained BERT model from [Huggingface](https://huggingface.co/google-bert/bert-base-uncased/tree/main).

2. Set up the environment.

```
conda create -n MMEPA python=3.7
conda activate MMEPA
pip install -r requirements.txt
```

3. Start training.

Training on CMU-MOSI:

```
python main.py --dataset mosi --data_path [your MOSI path] --bert_path [your bert path]
```
Training on other dataset and pre-trained language model:

coming soon.
## Citation



## Contact 

