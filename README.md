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

We are implementing an "Unpaired Multimodal Learner" architecture to demonstrate how modality-agnostic training can improve single-modality models without relying on paired datasets. We are building a shared neural network backbone that processes completely independent, unaligned image and text datasets. By forcing both modalities through the same weights, the goal is to show that the network can extract cross-modal synergies--effectively improving its image understanding and classification capabilities by learning shared semantic concepts from the unpaired text data.

---

## Architecture Overview

The system follows a dual-encoder design where each modality is independently encoded and then routed through a shared backbone:

1. **Image Encoder** -- A pre-trained Vision Transformer (DINOv2 ViT-Small/16 via `timm`) extracts visual features and projects them to a 512-dimensional embedding through a linear layer.

2. **Text Encoder** -- A pre-trained DistilBERT model extracts textual features from the [CLS] token and projects them to the same 512-dimensional space through a linear layer.

3. **Shared Backbone** -- A 4-layer PyTorch `TransformerEncoder` (8 attention heads, 2048-wide feed-forward) processes the 512-d embeddings from either modality using shared weights.

4. **Classification Head** -- A linear layer maps the backbone output to the 196 Stanford Cars classes.

The training loop alternates between image and text batches, computing Cross-Entropy loss independently for each modality. The two modalities are never paired -- they share only the backbone and classifier weights.

---

## Repository Structure

```
FinalProject/
├── download_data.py      # Dataset ingestion and synthetic text generation
├── dataset.py            # Unpaired PyTorch Dataset and DataLoader classes
├── model.py              # UnpairedMultimodalLearner network architecture
├── train.py              # Alternating training loop with checkpointing
├── requirements.txt      # Python dependencies
├── data/
│   ├── images/           # Stanford Cars images organized by class
│   └── text/             # Synthetic text descriptions (one CSV per class)
└── checkpoints/          # Saved model weights (best validation accuracy)
```

### File Descriptions

- **download_data.py** -- Downloads the Stanford Cars dataset from Kaggle (via `kagglehub`), organizes images into class-named subfolders under `data/images/`, and generates synthetic unpaired text descriptions saved as per-class CSV files under `data/text/`.

- **dataset.py** -- Defines `ImageDataset` (loads images with standard torchvision transforms) and `TextDataset` (loads CSV descriptions and tokenizes with the DistilBERT tokenizer). Provides separate `get_image_dataloader` and `get_text_dataloader` factory functions for fully independent, unaligned batch iteration.

- **model.py** -- Implements `ImageEncoder`, `TextEncoder`, and the full `UnpairedMultimodalLearner` module. The forward pass accepts either an image batch or a text batch (not both), routes it through the appropriate encoder, and passes the resulting embedding through the shared TransformerEncoder backbone and classification head.

- **train.py** -- Runs the alternating training loop: one image batch followed by one text batch per step, each with its own Cross-Entropy loss and backpropagation. Logs per-step and per-epoch losses, evaluates on a validation split after each epoch, saves the best model to `checkpoints/`, and performs a final top-1 accuracy evaluation on the held-out test split.

- **requirements.txt** -- All Python package dependencies for the project.

---

## Setup and Execution

### Prerequisites

- Python 3.10+
- Kaggle API credentials (place `kaggle.json` in `~/.kaggle/` or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables)

### 1. Local Setup and Data Download

```bash
# Clone the repository
git clone <REPOSITORY_URL>
cd FinalProject

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Download the Stanford Cars dataset and generate synthetic text
python download_data.py
```

After running `download_data.py`, the `data/images/` directory will contain 196 class subfolders with images and the `data/text/` directory will contain 196 CSV files with synthetic descriptions.

### 2. Training on Google Colab Pro (GPU Runtime)

Upload the project to Google Drive, then run the following in a Colab notebook with a GPU runtime selected:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to the project directory
%cd /content/drive/MyDrive/<PATH_TO_PROJECT>/FinalProject

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
| Demo Video | [YouTube URL -- TO BE ADDED] |
| Dataset | [Google Drive URL -- TO BE ADDED] |
| Stanford Cars Dataset | Sourced via Kaggle (`jutrera/stanford-car-dataset-by-classes-folder`) |
| ViT-Small/16 (timm) | Pre-trained weights loaded automatically via the `timm` library |
| DistilBERT | Pre-trained weights loaded automatically via HuggingFace `transformers` |

---

## Acknowledgments

This project was completed as the final project for CS 5330 Pattern Recognition and Computer Vision at Northeastern University. The Stanford Cars dataset was originally published by Krause et al. (2013). Pre-trained model weights are provided by the `timm` and HuggingFace `transformers` libraries.
