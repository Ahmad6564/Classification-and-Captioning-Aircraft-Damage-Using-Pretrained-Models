# Aircraft Damage Classification and Captioning Using Pretrained Models

A comprehensive machine learning project that combines computer vision classification and natural language processing to analyze aircraft damage images. The project classifies aircraft defects as either 'dent' or 'crack' and generates descriptive captions and summaries using state-of-the-art pretrained models.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project demonstrates the application of deep learning techniques for aircraft maintenance and inspection. It consists of two main components:

1. **Binary Classification**: Distinguishes between 'dent' and 'crack' damage types using a fine-tuned VGG16 model
2. **Image Captioning & Summarization**: Generates descriptive text for aircraft damage images using the BLIP (Bootstrapped Language-Image Pre-training) model

## ✨ Features

- **Robust Image Classification**: VGG16-based classifier with data augmentation
- **Automated Image Captioning**: Generate descriptive captions for aircraft damage
- **Image Summarization**: Create detailed summaries of aircraft condition
- **Visualization Tools**: Training progress visualization and prediction results
- **Custom Keras Layers**: Implementation of specialized neural network components
- **Comprehensive Evaluation**: Model performance metrics and validation

## 📁 Project Structure

```
aircraft-damage-analysis/
│
├── aircraft_damage_dataset_v1/
│   ├── train/
│   │   ├── dent/
│   │   └── crack/
│   ├── valid/
│   │   ├── dent/
│   │   └── crack/
│   └── test/
│       ├── dent/
│       └── crack/
│
├── models/
│   ├── vgg16_classifier.h5
│   └── blip_model/
│
├── notebooks/
│   └── aircraft_damage_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── visualization.py
│
├── requirements.txt
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ccmuhammadahmad/Readme.md.git
cd aircraft-damage-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Required Libraries

```txt
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.21.0
matplotlib>=3.5.0
pillow>=8.3.0
transformers>=4.15.0
torch>=1.10.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
```

## 📊 Usage

### 1. Data Preparation

Organize your aircraft damage images in the following structure:
```
aircraft_damage_dataset_v1/
├── train/ (training images)
├── valid/ (validation images)
└── test/ (testing images)
```

### 2. Training the Classification Model

```python
# Configuration
batch_size = 32
epochs = 5
img_rows, img_cols = 224, 224

# Train the model
python src/model_training.py --batch_size 32 --epochs 5
```

### 3. Generate Image Captions

```python
from src.captioning import generate_text

# Generate caption
image_path = "path/to/your/image.jpg"
caption = generate_text(image_path, "caption")
print(f"Caption: {caption}")

# Generate summary
summary = generate_text(image_path, "summary")
print(f"Summary: {summary}")
```

### 4. Evaluate Model Performance

```python
python src/evaluation.py --model_path models/vgg16_classifier.h5
```

## 🏗️ Model Architecture

### Part 1: VGG16 Classification Model

- **Base Model**: VGG16 pretrained on ImageNet
- **Input Shape**: (224, 224, 3)
- **Output**: Binary classification (dent/crack)
- **Optimization**: Adam optimizer with learning rate scheduling
- **Data Augmentation**: Rotation, zoom, horizontal flip

### Part 2: BLIP Model for Captioning

- **Architecture**: Vision Transformer + BERT
- **Tasks**: Image captioning and summarization
- **Input**: RGB images (any size, automatically resized)
- **Output**: Natural language descriptions

## 📈 Dataset

The aircraft damage dataset contains:
- **Training Set**: Images for model training with data augmentation
- **Validation Set**: Images for hyperparameter tuning and model selection
- **Test Set**: Images for final model evaluation

### Data Preprocessing Pipeline

1. **Image Resizing**: All images resized to 224×224 pixels
2. **Normalization**: Pixel values normalized to [0, 1]
3. **Data Augmentation**: Applied to training set only
4. **Batch Processing**: Images processed in batches of 32

## 📊 Results

### Classification Performance

```
Training Results (5 epochs):
- Final Training Accuracy: 87.0%
- Final Validation Accuracy: 68.75%
- Training Loss: 0.321
- Validation Loss: 0.508
```

### Training Progress

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 1     | 53.7%     | 60.4%   | 0.720      | 0.634    |
| 2     | 72.3%     | 69.8%   | 0.554      | 0.582    |
| 3     | 77.7%     | 71.9%   | 0.474      | 0.551    |
| 4     | 81.0%     | 66.7%   | 0.414      | 0.649    |
| 5     | 87.0%     | 68.8%   | 0.321      | 0.508    |

### Sample Outputs

**Image Captioning Example:**
- Input: Aircraft engine damage image
- Caption: "this is a picture of a plane"
- Summary: "this is a detailed photo showing the engine of a boeing 747"

## 🛠️ Key Components

### Task Implementation

1. **Data Generator Setup**: Custom ImageDataGenerator with augmentation
2. **Model Architecture**: VGG16 with custom classification head
3. **Training Pipeline**: Automated training with validation monitoring
4. **Visualization Tools**: Training curves and prediction visualization
5. **BLIP Integration**: Pretrained model for text generation

### Custom Features

- **Robust Data Pipeline**: Handles large datasets efficiently
- **Model Checkpointing**: Automatic saving of best models
- **Comprehensive Logging**: Detailed training and evaluation logs
- **Error Handling**: Graceful handling of edge cases

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Ensure backward compatibility

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **VGG16 Model**: ImageNet pretrained weights from Keras Applications
- **BLIP Model**: Salesforce Research BLIP implementation
- **Dataset**: Aircraft damage dataset contributors
- **Libraries**: TensorFlow, PyTorch, Transformers, and the open-source community

## 📞 Contact

**Muhammad Ahmad**
- GitHub: [@Ahmad6564](https://github.com/Ahmad6564)
- Email: ahmadkhalid6564@gmail.com


## 🔮 Future Enhancements

- [ ] Multi-class damage classification (scratch, corrosion, etc.)
- [ ] Real-time damage detection API
- [ ] Integration with drone inspection systems
- [ ] Advanced visualization dashboards
- [ ] Mobile application development
- [ ] Severity assessment scoring

---

⭐ **Star this repository if you found it helpful!**
