# Understanding and Improving Zero-Reference Deep Curve Estimation for Low-light Image Enhancement

This repository provides the official PyTorch implementation of the following paper:

> Understanding and Improving Zero-Reference Deep Curve Estimation for Low-light Image Enhancement
>
> Jiahao Wu, Dandan Zhan, Zhi Jin
>
> In Applied intelligence. 
>
> Abstract: Zero-Reference Deep Curve Estimation (Zero-DCE) pioneers a new idea for LowLight Image Enhancement (LLIE), which is to formulate LLIE as a task of image-specific curve estimation with a deep network. Despite its success, the underlying mechanisms of Zero-DCE still remain under-explored, which prevents the further development of curve estimation methods for LLIE. In this paper, we take a step further in understanding Zero-DCE and provide an in-depth analysis from the perspective of the curve formula design and the loss function balance. Inspired by our analysis, we make effective modifications to Zero-DCE in terms of the curve formula, the curve estimation network and the loss terms, and propose Zero-Reference Exposure Adjusting Curve Estimation (Zero-EACE). A novel curve formula named EAC, a novel curve estimation network named HGNet, and a novel loss function named HE Loss are proposed. Extensive experimental results show that the proposed Zero-EACE achieves comparable performance to stateof-the-art methods both qualitatively and quantitatively. Moreover, experimental results on multiple exposure images demonstrate the capability of our method to simultaneously tackle over- and under-exposure correction, which expands the practical application scenarios of unsupervised curve estimation-based LLIE methods.

---

## Contents

The contents of this repository are as follows:

1. [Dependencies](#Dependencies)
2. [Train](#Train)
3. [Test](#Test)



---

## Dependencies

- Python
- Pytorch 
- Tensorboard

---

## Train

python train.py

---

## Test

python test.py
