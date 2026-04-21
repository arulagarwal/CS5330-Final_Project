# Unpaired Multimodal Learning for Stanford Cars Classification

**CS 5330 -- Pattern Recognition and Computer Vision -- Final Project**
Northeastern University, Spring 2026

---

## Group Members

| Name | Role |
|------|------|
| Arul Agarwal | Group Member |
| Anirudh Bakare | Group Member |
| Utkarsh Milind Khursale | Group Member |

---

## Project Description

We are implementing an "Unpaired Multimodal Learner" architecture to demonstrate how modality-agnostic training can improve single-modality models without relying on paired datasets. We are building a dual-encoder architecture that projects completely independent, unaligned image and text datasets into a shared semantic latent space. By projecting both modalities into this shared classification space, the goal is to show that the network can extract cross-modal synergies--effectively improving its image understanding and classification capabilities by learning shared semantic concepts from the unpaired text data.

---

---
Resou
---

## Architecture Overview

The system follows a dual-encoder design where each modality is independently encoded and then routed into a shared semantic space:

1. **Image Encoder** -- A pre-trained Vision Transformer (ViT-Small/16 trained via AugReg on ImageNet-21k, accessed via `timm`) extracts visual features and projects them to a 512-dimensional embedding through a linear layer.

2. **Text Encoder** -- A pre-trained DistilBERT model extracts textual features from the [CLS] token and projects them to the same 512-dimensional space through a linear layer.

3. **Shared Classification Head** -- A single linear classification head (196 classes) processes the 512-dimensional embeddings regardless of which modality they originated from. Modalities are fused only at this final linear projection layer, mapping both vision and language features into the same semantic space.

---

## Pipeline Execution (Colab)

The complete end-to-end pipeline is available in `UML_Colab_Pipeline.ipynb`. This notebook handles:
1. Cloning the repository and installing dependencies.
2. Downloading the Stanford Cars dataset and generating synthetic unpaired text descriptions.
3. Pre-computing zero-shot text anchors using DistilBERT.
4. Running hyperparameter tuning.
5. Executing the training loop (with or without frozen text anchors).
6. Generating t-SNE latent space visualizations for evaluation.

---

## Running the Code Locally

If you prefer to run the scripts individually outside of the Colab environment:

### 1. Data Setup

```bash
# Download images and generate unpaired text
python download_data.py --data-dir ./data

# Pre-compute text anchors for zero-shot classifier initialization
python init_weights.py --data-dir ./data --output ./text_anchors.pt
```

### 2. Training on Google Colab

To train the model on a Colab GPU, upload the notebook or run the following in a Colab cell:

```python
# Clone the repository
!git clone [https://github.com/arulagarwal/CS5330-Final_Project.git](https://github.com/arulagarwal/CS5330-Final_Project.git)
%cd CS5330-Final_Project

# Install dependencies
!pip install -r requirements.txt

# Run training
!python train.py --epochs 20 --batch-size 32 --lr 1e-4
```

The training script will automatically detect and use the available CUDA GPU. Model checkpoints are saved to `checkpoints/best_model.pt` based on the best validation accuracy. After training completes, the script outputs the final top-1 accuracy on the held-out image test split.

### 3. Training Locally (Optional)

```bash
python train.py --epochs 20 --batch-size 32 --lr 1e-4 --verbose
```

The script selects CUDA, MPS (Apple Silicon), or CPU automatically. Use `--help` to see all available options.

---

## Links and Resources

| Resource | Link |
|----------|------|
| Demo Video | [Here is the updated README with the links added to the Links and Resources section:

# Unpaired Multimodal Learning for Stanford Cars Classification

**CS 5330 -- Pattern Recognition and Computer Vision -- Final Project**
Northeastern University, Spring 2026

---

## Group Members

| Name | Role |
|------|------|
| Arul Agarwal | Group Member |
| Anirudh Bakare | Group Member |
| Utkarsh Milind Khursale | Group Member |

---

## Project Description

We are implementing an "Unpaired Multimodal Learner" architecture to demonstrate how modality-agnostic training can improve single-modality models without relying on paired datasets. We are building a dual-encoder architecture that projects completely independent, unaligned image and text datasets into a shared semantic latent space. By projecting both modalities into this shared classification space, the goal is to show that the network can extract cross-modal synergies--effectively improving its image understanding and classification capabilities by learning shared semantic concepts from the unpaired text data.

---

## Architecture Overview

The system follows a dual-encoder design where each modality is independently encoded and then routed into a shared semantic space:

1. **Image Encoder** -- A pre-trained Vision Transformer (ViT-Small/16 trained via AugReg on ImageNet-21k, accessed via `timm`) extracts visual features and projects them to a 512-dimensional embedding through a linear layer.

2. **Text Encoder** -- A pre-trained DistilBERT model extracts textual features from the [CLS] token and projects them to the same 512-dimensional space through a linear layer.

3. **Shared Classification Head** -- A single linear classification head (196 classes) processes the 512-dimensional embeddings regardless of which modality they originated from. Modalities are fused only at this final linear projection layer, mapping both vision and language features into the same semantic space.

---

## Pipeline Execution (Colab)

The complete end-to-end pipeline is available in `UML_Colab_Pipeline.ipynb`. This notebook handles:
1. Cloning the repository and installing dependencies.
2. Downloading the Stanford Cars dataset and generating synthetic unpaired text descriptions.
3. Pre-computing zero-shot text anchors using DistilBERT.
4. Running hyperparameter tuning.
5. Executing the training loop (with or without frozen text anchors).
6. Generating t-SNE latent space visualizations for evaluation.

---

## Running the Code Locally

If you prefer to run the scripts individually outside of the Colab environment:

### 1. Data Setup

```bash
# Download images and generate unpaired text
python download_data.py --data-dir ./data

# Pre-compute text anchors for zero-shot classifier initialization
python init_weights.py --data-dir ./data --output ./text_anchors.pt
```

### 2. Training on Google Colab

To train the model on a Colab GPU, upload the notebook or run the following in a Colab cell:

```python
# Clone the repository
!git clone https://github.com/arulagarwal/CS5330-Final_Project.git
%cd CS5330-Final_Project

# Install dependencies
!pip install -r requirements.txt

# Run training
!python train.py --epochs 20 --batch-size 32 --lr 1e-4
```

The training script will automatically detect and use the available CUDA GPU. Model checkpoints are saved to `checkpoints/best_model.pt` based on the best validation accuracy. After training completes, the script outputs the final top-1 accuracy on the held-out image test split.

### 3. Training Locally (Optional)

```bash
python train.py --epochs 20 --batch-size 32 --lr 1e-4 --verbose
```

The script selects CUDA, MPS (Apple Silicon), or CPU automatically. Use `--help` to see all available options.

---

## Links and Resources

| Resource | Link |
|----------|------|
| Demo Video | [Youtube](https://youtu.be/q0Wshn21NB0) |
| Dataset | [https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder) |
| Stanford Cars Dataset | Sourced via Kaggle (`jutrera/stanford-car-dataset-by-classes-folder`) |
| ViT-Small/16 (timm) | Supervised ImageNet-21k pre-trained weights loaded automatically via the `timm` library |
| DistilBERT | Pre-trained weights loaded automatically via HuggingFace `transformers` |

---

## Acknowledgments

This project was completed as the final project for CS 5330 Pattern Recognition and Computer Vision at Northeastern University. The Stanford Cars dataset was originally published by Krause et al. (2013). Pre-trained models are provided by HuggingFace and Ross Wightman's `timm` library.] |
| Stanford Cars Dataset | Sourced via Kaggle (`jutrera/stanford-car-dataset-by-classes-folder`) |
| ViT-Small/16 (timm) | Supervised ImageNet-21k pre-trained weights loaded automatically via the `timm` library |
| DistilBERT | Pre-trained weights loaded automatically via HuggingFace `transformers` |

---

## Acknowledgments

This project was completed as the final project for CS 5330 Pattern Recognition and Computer Vision at Northeastern University. The Stanford Cars dataset was originally published by Krause et al. (2013). Pre-trained models are provided by HuggingFace and Ross Wightman's `timm` library.
