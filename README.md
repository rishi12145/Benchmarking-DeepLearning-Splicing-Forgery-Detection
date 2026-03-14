# Benchmarking-DeepLearning-Splicing-Forgery-Detection
Official implementation of our ICVGIP research paper: "Benchmarking Deep Learning Approaches for Splicing Forgery Detection in Document Image Preprocessing Pipelines", comparing ResNet50, InceptionV3, and DenseNet201 on the CASIA 2.0 dataset.

# Splicing Forgery Detection Benchmark

Official implementation of our ICVGIP research paper:

**"Benchmarking Deep Learning Approaches for Splicing Forgery Detection in Document Image Preprocessing Pipelines."**

This project evaluates three deep learning architectures for detecting image splicing forgeries:

- ResNet50
- InceptionV3
- DenseNet201

The models are trained and evaluated on the **CASIA 2.0 Image Tampering Detection Dataset**.

---

## Dataset

CASIA 2.0 Dataset Statistics

- Total Images: 7000
- Authentic Images: 5000
- Tampered Images: 2000
- Image Resolution: 512×512
- Format: JPEG

Dataset split:

- Training: 80%
- Testing: 20%

---

## Models Evaluated

### ResNet50
Residual learning architecture designed to improve gradient flow in deep networks.

### InceptionV3
Multi-scale convolutional architecture using parallel filters.

### DenseNet201
Dense connectivity network allowing feature reuse and better gradient propagation.

---

## Evaluation Metrics

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|------|------|------|------|------|------|
| ResNet50 | 99.40% | 88.80% | 89.80% | 89.30% | 0.90 |
| InceptionV3 | 96.20% | 97.80% | 94.40% | 96.10% | 0.94 |
| DenseNet201 | 99.42% | 97.48% | 99.02% | 98.24% | 0.99 |

DenseNet201 achieved the best overall performance due to its dense feature reuse and improved gradient flow.

---

## Installation

```bash
pip install -r requirements.txt
