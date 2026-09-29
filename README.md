# Image Dimensionality Reduction with a Convolutional Autoencoder

Unsupervised dimensionality reduction of overhead images (**car** & **plane**, 28×28 grayscale) using a convolutional autoencoder. Each 784-pixel image is compressed into a **128-dimensional latent vector** and reconstructed. Starting from a simple baseline, the architecture is improved with an extra convolutional layer and Batch Normalization, then tuned on learning rate. Reconstruction quality is measured with SSIM.

**Result:** mean SSIM improved from **0.7017 to 0.8543** (+21.7%), with the biggest gain on the harder `plane` class (+29.3%).

## Key Objectives

- **Data Preparation:** Load the `car` and `plane` classes from OverheadMNIST, convert to 28×28 grayscale, and check class balance (8,012 images per class).
- **Preprocessing & Scaling:** Normalize pixel values to [0, 1] to match the decoder's `sigmoid` output, and reshape to `(28, 28, 1)` for convolutional layers.
- **Data Splitting:** Stratified 80/10/10 train/validation/test split. Labels are used only for stratification and per-class evaluation, never for training.
- **Baseline Autoencoder:** A single-Conv encoder and decoder with a 128-d bottleneck.
- **Improved Autoencoder:** Add a second Conv layer to the encoder, a Conv layer before upsampling in the decoder, and Batch Normalization after each Conv layer.
- **Hyperparameter Tuning:** Compare learning rates 1e-3, 5e-4, and 1e-4.
- **Quantitative Evaluation:** Measure reconstruction quality with SSIM (overall and per class).

## Dataset

[OverheadMNIST](https://www.kaggle.com/datasets/datamunge/overheadmnist) on Kaggle, using only the `car` and `plane` classes. The original train and test folders are merged (16,024 images) and re-split with stratification.

| Split | Car | Plane | Total |
|-------|-----|-------|-------|
| Train | 6,409 | 6,409 | 12,818 |
| Validation | 801 | 802 | 1,603 |
| Test | 802 | 801 | 1,603 |

## Models

### 1. Baseline
- **Encoder:** Conv2D 32 (3×3, ReLU) → MaxPooling → Flatten → Dense 128 (latent)
- **Decoder:** Dense (14×14×32) → Reshape → UpSampling → Conv2D 32 → Conv2D 1 (`sigmoid`)
- **Parameters:** 1,621,889

### 2. Modified
- **Encoder:** [Conv2D 32 → BatchNorm → ReLU] → MaxPooling → [Conv2D 32 → BatchNorm → ReLU] → Flatten → Dense 128 (latent)
- **Decoder:** Dense (14×14×32) → Reshape → [Conv2D 32 → BatchNorm → ReLU] → UpSampling → [Conv2D 32 → BatchNorm → ReLU] → Conv2D 1 (`sigmoid`)
- **Parameters:** 1,640,897

### 3. Modified + Tuned
Same architecture as the modified model, trained with the best learning rate from the sweep.

## Training Setup

- Loss: Mean Squared Error (MSE), trained to reconstruct the input (`fit(X, X)`)
- Optimizer: Adam
- Batch size 128, latent dimension 128
- Early stopping on validation loss with best-weights restore (patience 15 for baseline with up to 150 epochs, patience 8 for modified models with up to 80 epochs)
- Learning rates tested: 1e-3, 5e-4, 1e-4

## Results

| Model | SSIM (Overall) | Car | Plane |
|-------|----------------|-----|-------|
| Baseline | 0.7017 | 0.8079 | 0.5955 |
| Modified (lr = 1e-3) | 0.7726 | 0.8849 | 0.6603 |
| **Modified + Tuned (lr = 1e-4)** | **0.8543** | **0.9383** | **0.7702** |

**Learning rate sweep (modified architecture)**

| Learning rate | SSIM (Overall) | Car | Plane |
|---------------|----------------|-----|-------|
| 1e-3 | 0.7726 | 0.8849 | 0.6603 |
| 5e-4 | 0.7856 | 0.8950 | 0.6760 |
| **1e-4** | **0.8543** | **0.9383** | **0.7702** |

## Key Findings

- **Baseline:** reconstructs `car` reasonably well (SSIM 0.81) but struggles with `plane` (0.60). Planes are thin and diagonal, which is hard for an encoder with only one Conv layer, and their SSIM varies much more from image to image.
- **Architecture change:** the extra Conv layers and Batch Normalization improved SSIM by about 0.07 and made training more stable, with train and validation loss staying close together.
- **Learning rate:** the default 1e-3 was too large for this task. Lowering it to 1e-4 gave the largest single gain (SSIM +0.08 over the modified model), and the per-class gap between `car` and `plane` shrank noticeably.

## Conclusion

A small convolutional autoencoder can compress 28×28 overhead images into 128 dimensions (about 6× fewer values) while keeping most of their structure. Adding convolutional depth and Batch Normalization helped, but tuning the learning rate mattered even more: together they raised mean SSIM from 0.70 to 0.85 and improved the previously weak `plane` class the most. This shows that both architecture and optimization settings strongly affect how informative the latent representation is.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Metrics:** scikit-image (SSIM), scikit-learn (stratified splitting)
- **Data Handling & Visualization:** Pandas, NumPy, Matplotlib, Seaborn, Pillow
- **Environment:** Kaggle Notebooks (Tesla T4 GPU)
