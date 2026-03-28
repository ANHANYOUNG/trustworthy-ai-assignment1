# Trustworthy AI Assignment 1

This project trains image classifiers for `MNIST` and `CIFAR-10`, then evaluates four adversarial attacks:

- targeted FGSM
- untargeted FGSM
- targeted PGD
- untargeted PGD

The final submission entry point is `test.py`, and the saved attack visualization images are written to `results/`.

## Setup

- Tested with `Python 3.11`

Create and activate a conda environment first:

```bash
conda create -n trustworthy-ai-a1 python=3.11 -y
conda activate trustworthy-ai-a1
pip install -r requirements.txt
```

## Run

Basic run:

```bash
python test.py
```

If you want to choose a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py
```

Additional analysis scripts:

```bash
python ablation.py
python sweep.py
python tsne.py
python sal_map.py
```

Recommended order:

1. `python test.py`
2. `python ablation.py`
3. `python sweep.py`
4. `python tsne.py`
5. `python sal_map.py`

## What `test.py` Does

- loads `MNIST` and `CIFAR-10`
- trains a baseline `MNISTCNN`
- trains a baseline scratch `CIFARResNet`
- prints clean test accuracy
- evaluates FGSM / PGD, targeted / untargeted
- prints success rate, adversarial test accuracy, and attack time
- saves visualization PNG files into `results/`

## Analysis Scripts

- `ablation.py`
  - runs all `dropout / batchnorm / gelu` combinations for `MNIST` and `CIFAR-10`
  - saves checkpoints under `ckpts/`
  - saves metrics and prediction-comparison figures under `results/ablation/`

- `sweep.py`
  - loads the best clean checkpoint from `results/ablation/ablation_results.csv`
  - runs epsilon sweep for both datasets
  - saves metric plots and epsilon sweep figures under `results/sweep/`

- `tsne.py`
  - loads the best clean checkpoint from `results/ablation/ablation_results.csv`
  - generates `clean / targeted / untargeted` t-SNE figures
  - saves figures under `results/t-SNE/`

- `sal_map.py`
  - loads the best clean checkpoint from `results/ablation/ablation_results.csv`
  - generates `clean / targeted / untargeted` saliency map figures
  - saves figures under `results/saliency_map/`

## Example Final Log

After training and attack evaluation, the final console summary looks like this:

```text
=============================== Final Summary ===============================

eval samples: 100
target cls:   0
total runtime: 160.69s

[MNIST]
clean test acc: 0.9875
fgsm eps:       0.3000
pgd eps:        0.3000
pgd eps_step:   0.0100
pgd k:          40

attack            success_rate    adv_test_acc    time
-------------------------------------------------------
fgsm targeted     0.1200          0.1300          0.08s
fgsm untargeted   0.9700          0.0300          0.02s
pgd targeted      1.0000          0.0000          0.22s
pgd untargeted    1.0000          0.0000          0.11s

[CIFAR-10]
clean test acc: 0.8760
fgsm eps:       0.0314
pgd eps:        0.0314
pgd eps_step:   0.0078
pgd k:          10

attack            success_rate    adv_test_acc    time
-------------------------------------------------------
fgsm targeted     0.3100          0.0800          0.13s
fgsm untargeted   0.9900          0.0100          0.07s
pgd targeted      1.0000          0.0000          0.18s
pgd untargeted    1.0000          0.0000          0.14s
```

## Main Files

- `test.py`
- `ablation.py`
- `sweep.py`
- `tsne.py`
- `sal_map.py`
- `models/mnist_cnn.py`
- `models/cifar_resnet.py`
- `attacks/fgsm.py`
- `attacks/pgd.py`
- `utils/data.py`
- `utils/train.py`
- `utils/eval_attack.py`
- `utils/vis.py`
- `debug/test.ipynb`

## Outputs

Attack visualizations are saved under `results/`, for example:

- `results/mnist_fgsm_targeted.png`
- `results/mnist_fgsm_untargeted.png`
- `results/mnist_pgd_targeted.png`
- `results/mnist_pgd_untargeted.png`
- `results/cifar10_fgsm_targeted.png`
- `results/cifar10_fgsm_untargeted.png`
- `results/cifar10_pgd_targeted.png`
- `results/cifar10_pgd_untargeted.png`

Additional analysis outputs:

- `results/ablation/ablation_results.csv`
- `results/ablation/*.png`
- `results/sweep/epsilon_sweep_results.csv`
- `results/sweep/*.png`
- `results/t-SNE/*.png`
- `results/saliency_map/*.png`

## Notes

- `test.py` uses a scratch `CIFARResNet` baseline with `use_pretrained=False`
- `debug/test.ipynb` is for step-by-step verification and debugging
- `sweep.py`, `tsne.py`, and `sal_map.py` expect `results/ablation/ablation_results.csv` to exist first
