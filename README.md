# SimSearch  
### Self-Supervised Image Representation Learning & Retrieval

SimSearch is a deep learning project focused on **self-supervised learning** for image representation and similarity-based retrieval. The goal is to learn meaningful feature embeddings **without explicit labels**, enabling clustering and efficient search across visual data.

---

## Project Overview

Traditional supervised learning relies heavily on labeled datasets. In contrast, **SimSearch leverages self-supervised learning** to extract patterns and structure directly from raw images.

The model learns to:
- Understand visual similarity
- Separate different object categories
- Form meaningful clusters in embedding space

---

## Dataset

The dataset consists of **5 subcategories**:

- 👜 Bags  
- 🚗 Cars  
- 🐶 Dogs  
- 📱 Phones  
- 👟 Shoes  

Even without labels during training, the model gradually learns to distinguish between these categories.

---

## Methodology

- Self-supervised learning approach (contrastive / representation learning)
- Feature embedding generation
- Dimensionality reduction for visualization
- Clustering in latent space

---

## Results & Visualization

After training, the model learns to **separate datapoints and form clusters** based on semantic similarity.

### 2D Embedding Visualization

<img width="844" height="625" alt="2D" src="https://github.com/user-attachments/assets/815d556e-e64b-4e9a-97a4-08fd9602a407" />

### 3D Embedding Visualization

<img width="643" height="658" alt="3D" src="https://github.com/user-attachments/assets/7732eef1-9fce-4ae7-9684-5fefa26b43cc" />

### 3D Interactive Embedding Visualization

<img width="998" height="450" alt="newplot" src="https://github.com/user-attachments/assets/9862cfec-0db2-4d7b-a576-84acb48d2923" />

### Tech Stack

* PyTorch
* NumPy, Pandas, Matplotlib
* Scikit-learn

### Many Thanks

Abhinandan
