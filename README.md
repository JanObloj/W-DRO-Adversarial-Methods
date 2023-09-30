
# Wasserstein distributional robustness of neural networks

This repository is the official implementation of *[Wasserstein distributional robustness of neural networks](https://arxiv.org/abs/2306.09844)*.


## Requirements

1. Install basic requirements.

    ```sh
    conda env create -f environment.yml
    ```

2. Install [RobustBench](https://github.com/RobustBench/robustbench).


    ```sh
    pip install git+https://github.com/RobustBench/robustbench.git
    ```

## Preparation of ImageNet dataset

We use the same ImageNet dataset as Robustbench which is a randomly selected subset of 5000 images from the validation set. 
Below, we reiterate the steps for preparing this dataset:

1. Download the validation set of ILSVRC 2012 via [here](https://image-net.org/download-images.php).

2. Extract and preprocess the data into folder `.src/data/imagenet`.

    ```sh
    mkdir imagenet && mv ILSVRC2012_img_val.tar imagenet/ && cd imagenet && tar -xvf ILSVRC2012_img_val.tar
    wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
    ```
## Evaluation on the clean dataset

Calculate clean accuracy $A$, clean loss $V(0)$, conditional loss $W(0)$ on the misclassified images, and sensitivity $\Upsilon$:

```sh
python main.py $1 $2 $3 $4 $5
```
Input:

- `$1` -- dataset, "cifar10", "cifar100" or "imagenet".

- `$2` -- network index, see the table below.

- `$3` -- $q=1, 2$, conjugate of Wasserstein distance index $p$.

- `$4` -- loss_fn, "CE", "DLR", or "ReDLR".

- `$5` -- attack type, "clean".

Output: 
- a dict with keys `["acc", "loss", "loss_cond", "Upsilon", "q", "s", "mname", "loss_fn"]`.

All output files are stored in `./network_stats`.


## Wasserstein distributionally adversarial attack

Do W-PGD or W-FGSM attack on given networks:

```sh
python main.py $1 $2 $3 $4 $5 $6
```
Input:


- `$1` -- dataset, "cifar10", "cifar100", or "imagenet".

- `$2` -- network index, see the table below.

- `$3` -- $q=1, 2$, conjugate of Wasserstein distance index $p$.

- `$4` -- loss_fn, "CE", "DLR", or "ReDLR".

- `$5` -- attack type, "FGSM" or "PGD".

- `$6` -- attack budget. $\delta$=`$4`/510 if $s=1$, $\delta$=`$4`/32 if $s=2$.

Output:

- a dict with keys `["acc_min", "loss_max", "q", "s", "delta", "attack_type", "mname", "loss_fn"]`.

All output files are stored in `./network_stats`.

## Table of available networks

### CIFAR-10
| index | $l_{\infty}$ networks                | index | $l_2$ networks                       |
| :---: | ------------------------------------ | :---: | ------------------------------------ |
|   0   | Andriushchenko2020Understanding      |  69   | Augustin2020Adversarial              |
|   1   | Carmon2019Unlabeled                  |  70   | Engstrom2019Robustness               |
|   2   | Sehwag2020Hydra                      |  71   | Rice2020Overfitting                  |
|   3   | Wang2020Improving                    |  72   | Rony2019Decoupling                   |
|   4   | Hendrycks2019Using                   |  73   | Standard                             |
|   5   | Rice2020Overfitting                  |  74   | Ding2020MMA                          |
|   6   | Zhang2019Theoretically               |  75   | Wu2020Adversarial                    |
|   7   | Engstrom2019Robustness               |  76   | Gowal2020Uncovering                  |
|   8   | Chen2020Adversarial                  |  77   | Gowal2020Uncovering_extra            |
|   9   | Huang2020Self                        |  78   | Sehwag2021Proxy                      |
|  10   | Pang2020Boosting                     |  79   | Sehwag2021Proxy_R18                  |
|  11   | Wong2020Fast                         |  80   | Rebuffi2021Fixing_70_16_cutmix_ddpm  |
|  12   | Ding2020MMA                          |  81   | Rebuffi2021Fixing_28_10_cutmix_ddpm  |
|  13   | Zhang2019You                         |  82   | Rebuffi2021Fixing_70_16_cutmix_extra |
|  14   | Zhang2020Attacks                     |  83   | Augustin2020Adversarial_34_10        |
|  15   | Wu2020Adversarial_extra              |  84   | Augustin2020Adversarial_34_10_extra  |
|  16   | Wu2020Adversarial                    |  85   | Rebuffi2021Fixing_R18_cutmix_ddpm    |
|  17   | Gowal2020Uncovering_70_16            |  86   | Rade2021Helper_R18_ddpm              |
|  18   | Gowal2020Uncovering_70_16_extra      |  87   | Wang2023Better_WRN-28-10             |
|  19   | Gowal2020Uncovering_34_20            |  88   | Wang2023Better_WRN-70-16             |
|  20   | Gowal2020Uncovering_28_10_extra      |       |                                      |
|  21   | Sehwag2021Proxy                      |       |                                      |
|  22   | Sehwag2021Proxy_R18                  |       |                                      |
|  23   | Sehwag2021Proxy_ResNest152           |       |                                      |
|  24   | Sitawarin2020Improving               |       |                                      |
|  25   | Chen2020Efficient                    |       |                                      |
|  26   | Cui2020Learnable_34_20               |       |                                      |
|  27   | Cui2020Learnable_34_10               |       |                                      |
|  28   | Zhang2020Geometry                    |       |                                      |
|  29   | Rebuffi2021Fixing_28_10_cutmix_ddpm  |       |                                      |
|  30   | Rebuffi2021Fixing_106_16_cutmix_ddpm |       |                                      |
|  31   | Rebuffi2021Fixing_70_16_cutmix_ddpm  |       |                                      |
|  32   | Rebuffi2021Fixing_70_16_cutmix_extra |       |                                      |
|  33   | Sridhar2021Robust                    |       |                                      |
|  34   | Sridhar2021Robust_34_15              |       |                                      |
|  35   | Rebuffi2021Fixing_R18_ddpm           |       |                                      |
|  36   | Rade2021Helper_R18_extra             |       |                                      |
|  37   | Rade2021Helper_R18_ddpm              |       |                                      |
|  38   | Rade2021Helper_extra                 |       |                                      |
|  39   | Rade2021Helper_ddpm                  |       |                                      |
|  40   | Huang2021Exploring                   |       |                                      |
|  41   | Huang2021Exploring_ema               |       |                                      |
|  42   | Addepalli2021Towards_RN18            |       |                                      |
|  43   | Addepalli2021Towards_WRN34           |       |                                      |
|  44   | Gowal2021Improving_70_16_ddpm_100m   |       |                                      |
|  45   | Dai2021Parameterizing                |       |                                      |
|  46   | Gowal2021Improving_28_10_ddpm_100m   |       |                                      |
|  47   | Gowal2021Improving_R18_ddpm_100m     |       |                                      |
|  48   | Chen2021LTD_WRN34_10                 |       |                                      |
|  49   | Chen2021LTD_WRN34_20                 |       |                                      |
|  50   | Standard                             |       |                                      |
|  51   | Kang2021Stable                       |       |                                      |
|  52   | Jia2022LAS-AT_34_10                  |       |                                      |
|  53   | Jia2022LAS-AT_70_16                  |       |                                      |
|  54   | Pang2022Robustness_WRN28_10          |       |                                      |
|  55   | Pang2022Robustness_WRN70_16          |       |                                      |
|  56   | Addepalli2022Efficient_RN18          |       |                                      |
|  57   | Addepalli2022Efficient_WRN_34_10     |       |                                      |
|  58   | Debenedetti2022Light_XCiT-S12        |       |                                      |
|  59   | Debenedetti2022Light_XCiT-M12        |       |                                      |
|  60   | Debenedetti2022Light_XCiT-L12        |       |                                      |
|  61   | Huang2022Revisiting_WRN-A4           |       |                                      |
|  62   | Wang2023Better_WRN-28-10             |       |                                      |
|  63   | Wang2023Better_WRN-70-16             |       |                                      |
|  64   | Xu2023Exploring_WRN-28-10            |       |                                      |
|  65   | Cui2023Decoupled_WRN-28-10           |       |                                      |
|  66   | Cui2023Decoupled_WRN-34-10           |       |                                      |
|  67   | Bai2023Improving_edm                 |       |                                      |
|  68   | Peng2023Robust                       |       |                                      |


### CIFAR-100
| index | $l_{\infty}$ networks                   |
| :---: | --------------------------------------- |
|   0   | Gowal2020Uncovering                     |
|   1   | Gowal2020Uncovering_extra               |
|   2   | Cui2020Learnable_34_20_LBGAT6           |
|   3   | Cui2020Learnable_34_10_LBGAT0           |
|   4   | Cui2020Learnable_34_10_LBGAT6           |
|   5   | Chen2020Efficient                       |
|   6   | Wu2020Adversarial                       |
|   7   | Sehwag2021Proxy                         |
|   8   | Sitawarin2020Improving                  |
|   9   | Hendrycks2019Using                      |
|  10   | Rice2020Overfitting                     |
|  11   | Rebuffi2021Fixing_70_16_cutmix_ddpm     |
|  12   | Rebuffi2021Fixing_28_10_cutmix_ddpm     |
|  13   | Rebuffi2021Fixing_R18_ddpm              |
|  14   | Rade2021Helper_R18_ddpm                 |
|  15   | Addepalli2021Towards_PARN18             |
|  16   | Addepalli2021Towards_WRN34              |
|  17   | Chen2021LTD_WRN34_10                    |
|  18   | Pang2022Robustness_WRN28_10             |
|  19   | Pang2022Robustness_WRN70_16             |
|  20   | Jia2022LAS-AT_34_10                     |
|  21   | Jia2022LAS-AT_34_20                     |
|  22   | Addepalli2022Efficient_RN18             |
|  23   | Addepalli2022Efficient_WRN_34_10        |
|  24   | Debenedetti2022Light_XCiT-S12           |
|  25   | Debenedetti2022Light_XCiT-M12           |
|  26   | Debenedetti2022Light_XCiT-L12           |
|  27   | Cui2020Learnable_34_10_LBGAT9_eps_8_255 |
|  28   | Wang2023Better_WRN-28-10                |
|  29   | Wang2023Better_WRN-70-16                |
|  30   | Bai2023Improving_edm                    |
|  31   | Bai2023Improving_trades                 |
|  32   | Cui2023Decoupled_WRN-28-10              |
|  33   | Cui2023Decoupled_WRN-34-10              |
|  34   | Cui2023Decoupled_WRN-34-10_autoaug      |

### ImageNet

| index | $l_{\infty}$ networks                   |
| :---: | --------------------------------------- |
|   0   | Wong2020Fast                            |
|   1   | Engstrom2019Robustness                  |
|   2   | Salman2020Do_R50                        |
|   3   | Salman2020Do_R18                        |
|   4   | Salman2020Do_50_2                       |
|   5   | Standard_R50                            |
|   6   | Debenedetti2022Light_XCiT-S12           |
|   7   | Debenedetti2022Light_XCiT-M12           |
|   8   | Debenedetti2022Light_XCiT-L12           |
|   9   | Singh2023Revisiting_ViT-S-ConvStem      |
|  10   | Singh2023Revisiting_ViT-B-ConvStem      |
|  11   | Singh2023Revisiting_ConvNeXt-T-ConvStem |
|  12   | Singh2023Revisiting_ConvNeXt-S-ConvStem |
|  13   | Singh2023Revisiting_ConvNeXt-B-ConvStem |
|  14   | Singh2023Revisiting_ConvNeXt-L-ConvStem |
|  15   | Liu2023Comprehensive_ConvNeXt-B         |
|  16   | Liu2023Comprehensive_ConvNeXt-L         |
|  17   | Liu2023Comprehensive_Swin-B             |
|  18   | Liu2023Comprehensive_Swin-L             |
|  19   | Peng2023Robust                          |

