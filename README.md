
# Wasserstein distributional robustness of neural networks

This repository is the official implementation of *Wasserstein distributional robustness of neural networks*.


## Requirements

1. Install basic requirements.

    ```sh
    conda env create -f environment.yml
    ```

2. Install [RobustBench](https://github.com/RobustBench/robustbench).


    ```sh
    pip install git+https://github.com/RobustBench/robustbench.git
    ```

## Evaluation on the clean dataset

Calculate clean accuracy $A$, clean loss $V(0)$, conditional loss $W(0)$ on the misclassified images, and sensitivity Υ:

```sh
python main_clean.py $1 $2 $3
```
Input:

- `$1` -- network index, see the table below.

- `$2` -- $q=1, 2$, conjugate of Wasserstein distance index $p$.

- `$3` -- loss_fn, "CE", "DLR", or "ReDLR".

Output: 
- a dict with keys `["acc", "loss", "loss_cond", "Upsilon", "q", "s", "mname", "loss_fn"]`.

All output files are stored in `./network_stats`.


## Wasserstein distributionally adversarial attack

Do W-PGD or W-FGSM attack on given networks:

```sh
python main_adv.py $1 $2 $3 $4 $5
```
Input:

- `$1` -- network index, see the table below.

- `$2` -- $q=1, 2$, conjugate of Wasserstein distance index $p$.

- `$3` -- loss_fn, "CE", "DLR", or "ReDLR".

- `$4` -- attack budget. δ=`$4`/255 if $s=1$, δ=`$4`/16 if $s=2$.

- `$5` -- attack type, "FGSM" or "PGD".

Output:

- a dict with keys `["acc_min", "loss_max", "q", "s", "delta", "attack_type", "mname", "loss_fn"]`.

All output files are stored in `./network_stats`.

## Table of available networks

| index | l_inf networks                       | index | l_2 networks                         |
| :---: | ------------------------------------ | :---: | ------------------------------------ |
|   0   | Andriushchenko2020Understanding      |  64   | Augustin2020Adversarial              |
|   1   | Carmon2019Unlabeled                  |  65   | Engstrom2019Robustness               |
|   2   | Sehwag2020Hydra                      |  66   | Rice2020Overfitting                  |
|   3   | Wang2020Improving                    |  67   | Rony2019Decoupling                   |
|   4   | Hendrycks2019Using                   |  68   | Standard                             |
|   5   | Rice2020Overfitting                  |  69   | Ding2020MMA                          |
|   6   | Zhang2019Theoretically               |  70   | Wu2020Adversarial                    |
|   7   | Engstrom2019Robustness               |  71   | Gowal2020Uncovering                  |
|   8   | Chen2020Adversarial                  |  72   | Gowal2020Uncovering_extra            |
|   9   | Huang2020Self                        |  73   | Sehwag2021Proxy                      |
|  10   | Pang2020Boosting                     |  74   | Sehwag2021Proxy_R18                  |
|  11   | Wong2020Fast                         |  75   | Rebuffi2021Fixing_70_16_cutmix_ddpm  |
|  12   | Ding2020MMA                          |  76   | Rebuffi2021Fixing_28_10_cutmix_ddpm  |
|  13   | Zhang2019You                         |  77   | Rebuffi2021Fixing_70_16_cutmix_extra |
|  14   | Zhang2020Attacks                     |  78   | Augustin2020Adversarial_34_10        |
|  15   | Wu2020Adversarial_extra              |  79   | Augustin2020Adversarial_34_10_extra  |
|  16   | Wu2020Adversarial                    |  80   | Rebuffi2021Fixing_R18_cutmix_ddpm    |
|  17   | Gowal2020Uncovering_70_16            |  81   | Rade2021Helper_R18_ddpm              |
|  18   | Gowal2020Uncovering_70_16_extra      |  82   | Wang2023Better_WRN-28-10             |
|  19   | Gowal2020Uncovering_34_20            |  83   | Wang2023Better_WRN-70-16             |
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


