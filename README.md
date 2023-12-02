
# Diabetic-Foot-Ulcer-segmentation

Dual multi scale networks for diabetic foot ulcer segmentation using contrastive learning approach.

![App Screenshot](https://i.postimg.cc/hvRkfFMs/DFU-image4.png)

Default parameters are as follows : 

```bash
  optimizer_choice = 'AdamW'
  lr = 1e-4
  batchsize = 16
  trainsize = 224
  augmentation = False
  epoch = 50
  n_classes = 1
```
- Run DMSNet.ipynb and save the weights
- Put the weights path and dataset images and labels path in test_DMSNet.py 
- Run plot.py with paths to saved predictions and ground truths to plot the predictions on the input image.

## Data and Weights

Download required weights from :

**PVT:** https://github.com/DengPingFan/Polyp-PVT

**Foot ulcer segmentation challenge 2021 :** [Dataset](https://drive.google.com/file/d/1VY8vt3jtZH6rl_siO__Ns4hGaHeC0MZk/view?usp=sharing)


## Video Demo

https://youtu.be/rVb910-05PI

