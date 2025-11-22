# FACT++: Multi-Stage Hand Keypoint Fusion

This repository implements the FACT++ framework, which integrates hand keypoint data into the FACT architecture for enhanced egocentric action segmentation.

## Features

- **Multi-Stage Fusion**: Keypoints are fused at the Input Block, Update Block, and TDU Block.
- **Configurable Fusion Methods**: Supports 4 different fusion strategies described in the paper.

## Model Architecture

<!-- TODO: Add your model architecture diagram here -->
<!-- You can add an image using: ![Model Architecture](path/to/architecture.png) -->

### Overview

The FACT++ framework extends the original FACT architecture by integrating hand keypoint information at multiple stages:

1. **Input Block**: Initial fusion of hand keypoints with video features
2. **Update Block**: Temporal refinement with keypoint context
3. **TDU Block**: Final integration for action segmentation

### Architecture Diagram

```
[Add your architecture diagram here]

Suggested format:
- Create a diagram showing the flow from input to output
- Highlight the keypoint fusion points
- Show the different fusion methods (A, B, C, D)
- Save as: architecture.png or architecture.jpg
- Place in: /home/theflash/Documents/Documents/FACT_actseg_mask_arr/
- Reference here: ![FACT++ Architecture](architecture.png)
```

### Key Components

- **Keypoint Encoder**: Transforms raw hand keypoints (126-dim) to feature embeddings (512-dim)
- **Fusion Modules**: Four different fusion strategies (direct, CNN, linear, layernorm)
- **Temporal Modeling**: FACT backbone for temporal action segmentation
- **Multi-Stage Integration**: Keypoints fused at input, update, and TDU blocks

## Quick Start

```bash
# 1. Download GTEA dataset from Zenodo and extract to data/gtea/
# See Data Preparation section below for details

# 2. Install dependencies
pip install -r src/requirements.txt

# 3. Generate hand keypoints (if using raw frames)
python3 generate_hand_keypoints.py \
    --frames_root /path/to/gtea/frames \
    --keypoints_root data/gtea/keypoints_output_arr

# 4. Train with keypoint fusion
python3 -m src.train \
    --cfg src/configs/gtea.yaml \
    --set FACTPP.use_keypoints True FACTPP.fusion_method direct split "split1"
```

## Data Preparation

### Download GTEA Dataset

The GTEA (Georgia Tech Egocentric Activities) dataset is required for training and evaluation.

**Download Link**: [GTEA Dataset on Zenodo](https://zenodo.org/records/3625992#.Xiv9jGhKhPY)

### Expected Data Structure

After downloading and extracting the GTEA dataset, organize your data directory as follows:

```
FACT-plusplus-hand-keypoint-fusion/
├── data/
│   └── gtea/
│       ├── features/           # Pre-extracted video features
│       ├── groundTruth/         # Frame-level action labels
│       ├── splits/              # Train/test split files
│       ├── mapping.txt          # Action class mapping
│       └── keypoints_output_arr/  # Hand keypoints (generated in next step)
```

**Important Notes:**
- The `data/` folder is not included in the GitHub repository due to size constraints
- You must download the GTEA dataset separately and place it in the `data/gtea/` directory
- The `keypoints_output_arr/` folder will be created when you generate hand keypoints (see step 3 below)

### Data Folder Contents

The GTEA dataset should contain:

1. **features/**: Pre-extracted I3D features for each video
   - Format: `.npy` files, one per video
   - Shape: `(num_frames, feature_dim)`

2. **groundTruth/**: Frame-level action annotations
   - Format: `.txt` files, one per video
   - Each line represents the action label for that frame

3. **splits/**: Cross-validation split files
   - Files like `train.split1.bundle`, `test.split1.bundle`, etc.
   - Defines which videos are used for training/testing in each fold

4. **mapping.txt**: Maps action class names to integer labels

5. **keypoints_output_arr/**: Hand keypoint data (you will generate this)
   - Format: `.npy` files organized by video folders
   - Each file contains 126 values (left hand + right hand keypoints)

## Setup

### 1. Download and Prepare Data

Follow the [Data Preparation](#data-preparation) section above to download and organize the GTEA dataset.

### 2. Install Dependencies

Install all required Python packages:

```bash
pip install -r src/requirements.txt
```

This will install:
- PyTorch and torchvision
- MediaPipe (for hand keypoint extraction)
- OpenCV, NumPy, pandas, scipy
- tqdm, wandb, yacs, lmdb

### 3. Generate Hand Keypoints

Before training, you need to generate hand keypoints from your video frames using MediaPipe:

```bash
python3 generate_hand_keypoints.py \
    --frames_root /path/to/your/frames \
    --keypoints_root data/gtea/keypoints_output_arr
```

**Arguments:**
- `--frames_root`: Path to the directory containing video frame folders (required)
- `--keypoints_root`: Path where keypoint .npy files will be saved (default: `keypoints_output_arr`)
- `--use_cuda`: Optional flag to attempt CUDA acceleration if available

**Output Format:**
- Each frame generates a `.npy` file with shape `(126,)`
- Left hand: 21 keypoints × 3 coordinates (x, y, z) = 63 values
- Right hand: 21 keypoints × 3 coordinates (x, y, z) = 63 values
- If a hand is not detected, zeros are used for that hand's keypoints

**Example:**
```bash
# For GTEA dataset
python3 generate_hand_keypoints.py \
    --frames_root data/gtea/frames \
    --keypoints_root data/gtea/keypoints_output_arr
```

## Fusion Methods

You can select the fusion method using the `FACTPP.fusion_method` configuration option:

1.  **Direct Integration (`direct`)**:
    - Corresponds to **Method B** in the paper (Table I).
    - Uses a simple fully-connected encoder (`SimpleFC`) and adds the embedding directly to frame features.
    - **Best Performance** in the paper.

2.  **CNN-based Integration (`cnn`)**:
    - Corresponds to **Method A** in the paper (Table I).
    - Uses a 1D CNN to encode keypoints before addition.

3.  **Linear Projection Integration (`linear`)**:
    - Corresponds to **Method C** in the paper (Table I).
    - Uses a linear projection layer at each fusion point.

4.  **LayerNorm-Enhanced Integration (`layernorm`)**:
    - Corresponds to **Method D** in the paper (Table I).
    - Applies Layer Normalization after each fusion operation.

## Training

To enable keypoints and select a fusion method, use the command line arguments:

```bash
python3 -m src.train --cfg src/configs/gtea.yaml --set FACTPP.use_keypoints True FACTPP.fusion_method direct split "split1"
```

### Configuration Options

- `FACTPP.use_keypoints`: Set to `True` to enable keypoint loading and fusion.
- `FACTPP.fusion_method`: One of `direct`, `cnn`, `linear`, `layernorm`.
- `FACTPP.keypoint_dim`: Dimension of input keypoints (default 126).
- `FACTPP.embed_dim`: Dimension of keypoint embedding (default 512).

## Requirements

- The `gtea` dataset must have keypoint files in `data/gtea/keypoints_output_arr`.
- Python 3.7+
- CUDA-capable GPU recommended for training
