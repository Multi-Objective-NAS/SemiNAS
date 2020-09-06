# SemiNAS: Semi-Supervised Neural Architecture Search

This repository contains the code used for [Semi-Supervised Neural Architecture Search](https://arxiv.org/abs/2002.10389), by [Renqian Luo](http://home.ustc.edu.cn/~lrq), [Xu Tan](https://www.microsoft.com/en-us/research/people/xuta/), [Rui Wang](https://scholar.google.com/citations?hl=zh-CN&user=h1IrWikAAAAJ), [Tao Qin](https://www.microsoft.com/en-us/research/people/taoqin/), [En-Hong Chen](http://staff.ustc.edu.cn/~cheneh/), [Tie-Yan Liu](https://www.microsoft.com/en-us/research/people/tyliu/).


## Environments and Requirements
The code is built and tested on Pytorch 1.2

## NASBench-101 Experiments
Example) `CUDA_VISIBLE_DEVICES=2 python3 -m nasbench.train_seminas --data ../../datasets/nasbench --output_dir vanilla --semisupervised False`

## ImageNet Experiments
Please refer to imagenet/README.md

## TTS Experiments
Please refer to tts/README.md
