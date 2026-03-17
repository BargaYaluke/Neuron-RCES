# Neuron-RCES

Official research codebase for **Neuron-RCES**, a neuron-level robust criticality framework for robustness-oriented model analysis and selective fine-tuning.

This repository implements a unified experimental pipeline for:

- standard supervised training on image classification benchmarks,
- layer-level robust criticality analysis in the RiFT style,
- neuron-level robust criticality estimation under adversarial perturbations,
- mask-based selective fine-tuning of the least robust neurons,
- clean and robust evaluation under PGD attacks and common corruptions.

The current implementation is centered around a single training entry point, [`main.py`], and is designed for reproducible experiments on `CIFAR10`, `CIFAR100`, and `TinyImageNet`.

## Highlights

- **Neuron-level robust criticality estimation**: identifies fine-grained vulnerable neurons instead of operating only at the module level.
- **Adversarially grounded selection**: computes neuron importance from gradients accumulated on PGD-generated adversarial samples.
- **Selective adaptation**: only the selected low-robustness neurons remain trainable during the fine-tuning stage.
- **Flexible training strategy**: supports standard cross-entropy training and optional contrastive regularization.
- **Unified robustness evaluation**: includes clean accuracy, PGD robustness, and corruption robustness utilities.

## Method Overview

Neuron-RCES follows a two-stage robustness-oriented workflow:

1. **Critical neuron discovery**
   A pretrained or freshly initialized model is evaluated on adversarially perturbed data. Gradient statistics are accumulated over multiple adversarial mini-batches to estimate the neuron-level robust criticality of each trainable parameter group.

2. **Selective fine-tuning**
   The least robust neurons in each layer are retained as trainable, while the remaining parameters are frozen through mask-based gradient or parameter gating. The model is then fine-tuned to improve robustness-efficiency trade-offs.

In addition to the neuron-level pipeline, the codebase also retains a layer-level MRC analysis mode for comparison with prior RiFT-style workflows.

## Repository Structure

- [`main.py`](/c:/Neuron-RCES/Neuron-RCES/main.py): main entry for training, neuron-level MRC computation, selective fine-tuning, and evaluation.
- [`model.py`](/c:/Neuron-RCES/Neuron-RCES/model.py): model factory for supported architectures.
- [`dataloader.py`](/c:/Neuron-RCES/Neuron-RCES/dataloader.py): dataset definitions and dataloader utilities.
- [`utils.py`](/c:/Neuron-RCES/Neuron-RCES/utils.py): evaluation, logging, interpolation, and robustness utilities.
- [`optimizer.py`](/c:/Neuron-RCES/Neuron-RCES/optimizer.py): optimizer and learning-rate scheduler creation.
- [`loss_utils.py`](/c:/Neuron-RCES/Neuron-RCES/loss_utils.py): supervised contrastive loss.
- [`models/`](/c:/Neuron-RCES/Neuron-RCES/models): backbone implementations, including ResNet, WideResNet, DenseNet, VGG, FCN, and ViT variants.
- [`env.yaml`](/c:/Neuron-RCES/Neuron-RCES/env.yaml): Conda environment specification.
- [`requirement.txt`](/c:/Neuron-RCES/Neuron-RCES/requirement.txt): Python package requirements from the original environment.

## Supported Benchmarks

- `CIFAR10`
- `CIFAR100`
- `TinyImageNet`

The codebase also includes dataset wrappers for corruption benchmarks:

- `CIFAR-10-C`
- `CIFAR-100-C`
- `Tiny-ImageNet-C`


## Environment Setup

### Option 1: Conda

```bash
conda env create -f env.yaml
conda activate rift
```

### Option 2: pip

```bash
pip install -r requirement.txt
```

If your environment does not already include the robustness dependencies used by the code, ensure the following packages are available:

```bash
pip install torchattacks robustbench
```

## Data Preparation

The code expects datasets under `./data`.

### CIFAR-10 / CIFAR-100

These datasets are downloaded automatically by `torchvision` when needed.

### Tiny ImageNet

Place the dataset in:

```text
./data/tiny-imagenet-200/
```

with the standard `train/` and `val/` directory structure.

### Corruption Benchmarks

For corruption evaluation, prepare the following optional folders:

```text
./data/CIFAR-10-C/
./data/CIFAR-100-C/
./data/Tiny-ImageNet-C/
```

## Quick Start

### 1. Standard training / selective fine-tuning pipeline

```bash
python main.py --model ResNet18 --dataset CIFAR10 --epochs 10 --neurons_per_layer 1
```

### 2. Compute neuron-level MRC only

```bash
python main.py --model ResNet18 --dataset CIFAR10 --cal_neuron_mrc --neurons_per_layer 1
```

### 3. Compute layer-level MRC only

```bash
python main.py --model ResNet18 --dataset CIFAR10 --cal_mrc
```

### 4. Enable contrastive regularization

```bash
python main.py --model ResNet18 --dataset CIFAR10 --epochs 10 --contrastive --lambda_con 0.1
```

### 5. Run on Tiny ImageNet

```bash
python main.py --model ResNet18 --dataset TinyImageNet --num_classes 200 --input_size 64 --epochs 10
```

## Important Arguments

### Core experiment settings

- `--model`: backbone architecture.
- `--dataset`: one of `CIFAR10`, `CIFAR100`, `TinyImageNet`.
- `--num_classes`: number of output classes.
- `--input_size`: image resolution used by the model.
- `--resume`: path to a checkpoint for loading pretrained weights.

### Neuron-level robust criticality

- `--cal_neuron_mrc`: compute neuron-level MRC and exit.
- `--neurons_per_layer`: number of least-robust neurons selected per layer.
- `--num_grad_batches`: number of adversarial batches used for criticality estimation.
- `--epsilon`: perturbation magnitude parameter used in the analysis pipeline.
- `--adv_source`: adversarial source split, `train` or `test`.
- `--adv_subset`: subset size used to generate adversarial samples.
- `--adv_seed`: random seed for adversarial subset sampling.
- `--mask_mode`: selective update strategy, either `grad` or `param`.

### Optimization

- `--epochs`: number of training epochs.
- `--batch_size`: training batch size.
- `--lr`: learning rate.
- `--wd`: weight decay.
- `--optim`: optimizer type.
- `--lr_scheduler`: scheduler strategy.
- `--momentum`: momentum for SGDM.

### Representation regularization

- `--contrastive`: enable supervised contrastive learning.
- `--lambda_ce`: weight for the cross-entropy term.
- `--lambda_con`: weight for the contrastive term.
- `--temperature`: temperature for contrastive loss.

## Outputs

Experiment artifacts are saved under:

```text
./results/{model}_{dataset}/
```

Typical outputs include:

- checkpoints for initialization, periodic states, and best-performing models,
- `output.log` for detailed run-time logs,
- `neuron_mrc_list.npy` for neuron-level criticality values,
- `neuron_masks.pth` for selected-neuron masks and summary statistics,
- `experiment_record.json` for structured experiment metadata and results.

## Evaluation Protocol

The current implementation includes:

- **Clean accuracy** on the target test set,
- **PGD robustness** via `torchattacks.PGD`,
- **Corruption robustness** on CIFAR-C and Tiny-ImageNet-C style datasets,
- **Interpolation analysis** between initialization and fine-tuned weights.

For CIFAR benchmarks, robustness evaluation uses normalized inputs with PGD perturbations. For Tiny ImageNet, the same evaluation logic is adapted to 64x64 inputs.

## Reproducibility Notes

- Random seeds are explicitly set in the main training script.
- Initial model parameters are saved before selective fine-tuning.
- The repository stores masks, logs, and JSON records for post-hoc analysis.
- For publication-grade experiments, we recommend fixing library versions with [`env.yaml`] and reporting the full command line used for each run.

## Recommended Citation

If you use this repository in academic work, please cite the corresponding paper once bibliographic information is available.


## License

This project is released under the MIT License. See [`LICENSE`](/c:/Neuron-RCES/Neuron-RCES/LICENSE) for details.

