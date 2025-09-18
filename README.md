# Jade Classification Project

This project classifies jade artifacts into four categories:
- Half-ring (半环)
- Half-bi (半璧)
- Semicircle (半圆)
- Bridge (桥)

## Project Structure
jade-classification/
├── data/
│ ├── raw/ # Original data
│ │ ├── sample/ # Image directory
│ │ └── label.xlsx # Label file
│ └── processed/ # Processed data
│ └── test_results.xlsx
├── models/
│ └── model.pth # Trained model
├── src/
│ ├── baseline/ # Baseline implementation
│ │ └── train.py # Main training script
│ └── utils/ # Utility functions
│ ├── dataset.py # CustomDataset class
│ ├── preprocessing.py # Data preprocessing
│ └── model.py # Model definition
├── requirements.txt
├── README.md
└── .gitignore

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Place your images in `data/raw/sample/`
3. Place your label file as `data/raw/label.xlsx`

## Usage
- To train the model: Uncomment the `train_model()` call in `src/baseline/train.py`
- To evaluate the model: Run `python src/baseline/train.py`

