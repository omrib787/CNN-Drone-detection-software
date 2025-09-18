# Image Classification Using ResNet Features + Random Forest

This project implements an image classification pipeline that combines deep feature extraction using a pretrained ResNet50 model with traditional machine learning classifiers. After extensive testing, the Random Forest classifier outperformed other models (SVM, Decision Tree, etc.) in both accuracy and stability.

---

## Objective

To design and evaluate a hybrid image classification system that leverages pretrained deep learning models for feature extraction and classical machine learning methods for final classification. The goal was to achieve high accuracy on a custom multi-class image dataset.

---

## Technologies Used

- Python 3.x
- PyTorch (torch, torchvision)
- Scikit-learn
- NumPy, Matplotlib, Pillow
- Tkinter (for GUI)
- Joblib (for model serialization)

---

## Project Structure

| File | Description |
|------|-------------|
| `forrest.py` | Main model – trains and evaluates a Random Forest using ResNet50 features |
| `forrest_analyzer.py` | GUI to visualize predictions and top-5 probabilities |
| `RESnet.py` | ResNet50-based feature extractor |
| `SVMImageClassifier.py` | Early version using SVM (underperformed) |
| `resnet_image_analyzer.py` | Prototype GUI version (not used in final system) |
| `final report.docx.pdf` | Full project writeup and results summary |

---

## Models Tried

| Model | Status | Notes |
|-------|--------|-------|
| Random Forest | Best results | Stable, accurate, robust |
| SVM | Poor generalization | Lower accuracy |
| Decision Tree | Overfit quickly | Unstable |
| ResNet direct classification | Not used | Used only for feature extraction |

Random Forest consistently outperformed other models.  
Final accuracy on test data was approximately 90% (see report for detailed results).

---

## Dataset

- Custom image dataset with 10 balanced classes
- Preprocessed to 224x224 resolution
- Split: 80% train, 20% validation/test
- Feature vectors extracted from ResNet50 (2048D)

---

## How to Run

### 1. Install requirements

```bash
pip install torch torchvision scikit-learn pillow matplotlib joblib
```

### 2. Train the model

```bash
python forrest.py
```

- Loads dataset
- Extracts ResNet features
- Trains Random Forest
- Saves `.pkl` model

### 3. Visualize with GUI

```bash
python forrest_analyzer.py
```

- Choose a test image folder and model `.pkl` file
- GUI will show predictions and top-5 class probabilities

---

## Results Summary

- ResNet50 feature extraction + Random Forest achieved approximately 90% accuracy
- SVM and Decision Tree models were not competitive
- Random Forest had better stability, recall, and generalization
- Full experiment results are available in the final report

---

## License

This project is for academic and educational use only.