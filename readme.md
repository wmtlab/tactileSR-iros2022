# Tactile pattern super resolution with taxel-based sensors

This repository contains a python implementation of our IROS 2022 paper:

***Tactile pattern super resolution with taxel-based sensors***

Bing Wu, Qian Liu, Qiang Zhang

Dalian University of Technology

*****

## 1. Generate Tactile Pattern Dataset

```bash
python utility/genTactileSRDataSet.py
```

## 2. Train SR model

**tactileSRCNN**
```bash
python train/tactileSRCNN.py
```

**tactileSRGAN**
```bash
python train/tactileSRGAN.py
```
