# SimSearch

### Self-Supervised Image Embedding Search Engine

SimSearch is a **self-supervised learning based image retrieval system** that learns visual representations from **unlabeled images** and performs **semantic image similarity search** using vector embeddings.
Instead of training on labeled categories, the model learns to understand visual structure by using **contrastive learning**. After training, images can be converted into embeddings and stored in a vector index to enable **fast image similarity search**.

---

# Project Overview

Traditional image classification requires labeled datasets. However, labeling large datasets is expensive and time-consuming.
SimSearch uses **self-supervised learning**, where the model learns directly from raw images without manual labels.
The system trains an encoder using **contrastive learning**, where two augmented views of the same image are treated as a positive pair while other images act as negative samples.
Once trained, the encoder produces **embedding vectors** that represent visual meaning. Similar images produce similar vectors.
These embeddings are then indexed using **vector search**, enabling fast retrieval of visually similar images.

---

# System Pipeline

```
Raw Images
    ↓
Data Augmentation
    ↓
Two Augmented Views of Same Image
    ↓
Encoder Network (ResNet18)
    ↓
Projection Head
    ↓
NT-Xent Contrastive Loss
    ↓
Trained Encoder
    ↓
Image Embeddings
    ↓
Vector Index (FAISS)
    ↓
Image Similarity Search
```

---

# Key Features

Self-supervised learning (no manual labels required)
Contrastive representation learning
ResNet-based visual encoder
NT-Xent contrastive loss
Image embedding generation
Vector similarity search using FAISS
Modular PyTorch implementation

---


# Model Architecture

The core component of the system is the **SimSearchEncoder**.

```
Input Image (224x224)
      ↓
ResNet18 Backbone
      ↓
Global Average Pooling
      ↓
512 Feature Vector
      ↓
Linear Projection Layer
      ↓
128 Dimensional Embedding
```

During contrastive training a **projection head** is added to improve representation learning.

---

# Augmentations Used

The augmentation pipeline follows techniques used in contrastive learning frameworks.

Random resized crop

Horizontal flip

Color jitter

Random grayscale

Gaussian blur

These augmentations ensure the model learns **robust visual representations**.

---

# Dependencies

PyTorch

Torchvision

NumPy

FAISS

scikit-learn

Pillow

tqdm

---

# Future Improvements

Use larger backbone networks (ResNet50, Vision Transformers)

Train on larger unlabeled datasets

Add web interface for interactive image search

Implement approximate nearest neighbor indexing

Add embedding visualization (t-SNE / UMAP)

Extend to multimodal retrieval (image + text)

---

# Applications

Content-based image retrieval

Visual search engines

Duplicate image detection

Dataset exploration

Representation learning research

---

# Acknowledgements

The implementation is inspired by ideas from contrastive learning frameworks such as:

SimCLR

MoCo

BYOL

DINO

---


