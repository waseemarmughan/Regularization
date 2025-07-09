# Lab09 Regularization

## Overview
CNN regularization techniques comparison using Dogs vs. Cats dataset. Six different approaches tested to prevent overfitting.

## Dataset
- **Dogs vs. Cats**: 25,000 images (20,000 train / 5,000 validation)
- **Size**: 150×150 RGB images
- **Classes**: Binary classification (Dogs/Cats)

## Results Summary

| Model | Training Acc | Validation Acc | Improvement |
|-------|-------------|----------------|-------------|
| Baseline | 98.66% | 82.08% | -- |
| Dropout | 91.69% | 85.54% | +3.46% |
| L2 Regularization | 90.13% | 84.36% | +2.28% |
| Combined (Drop+L2) | 85.79% | 85.28% | +3.20% |
| Early Stopping | 92.99% | 85.54% | +3.46% |
| Data Augmentation | 79.86% | **86.24%** | **+4.16%** |

## Quick Start

### 1. Setup
```bash
pip install tensorflow matplotlib seaborn pandas scikit-learn gdown
```

### 2. Download Data
```python
!gdown https://drive.google.com/uc?id=12WhCCpKTWpeBztLegcoYx2gMo2KbaxDG
```

### 3. Run
```python
python regularization_lab.py
```

## Key Models

### Baseline (No Regularization)
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### Best Model (Combined Regularization)
```python
model = Sequential([
    # ... conv layers ...
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))
])
```

### Data Augmentation
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
```

## Key Findings
- **Baseline**: Severe overfitting (16.58% gap)
- **Data Augmentation**: Best validation accuracy (86.24%)
- **Combined Regularization**: Most balanced performance
- **All regularization methods** significantly improved generalization

## Requirements
- **GPU**: 8GB+ VRAM recommended
- **Runtime**: ~45-60 minutes total
- **Memory**: 16GB RAM minimum

## File Structure
```
Lab09_Regularization/
├── regularization_lab.py
├── data/
│   ├── train_split/
│   └── val_split/
└── results/
    └── comparison_plots.png
```

## Author
**Mir Armughan Waseem** (12503165)  
Embedded Systems | Prof. Tobias Schaffer | July 9, 2025
