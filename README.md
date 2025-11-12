# 🛡️ Credit Card Fraud Detection using 1D CNN

This project uses a 1D Convolutional Neural Network (CNN) to detect fraudulent credit card transactions from a highly imbalanced dataset.

## 📁 Dataset
- Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions
- 492 frauds (Class = 1), rest are normal (Class = 0)

## ⚙️ Workflow
1. **Data Preprocessing**
   - Balanced dataset using undersampling
   - Standardized features using `StandardScaler`
   - Reshaped input for CNN

2. **Model Architecture**
   - 2 Conv1D layers with BatchNorm and Dropout
   - Flatten + Dense layers
   - Sigmoid output for binary classification

3. **Training**
   - Optimizer: Adam
   - Loss: Binary Crossentropy
   - Epochs: 20

4. **Evaluation**
   - Accuracy and loss plotted for training and validation sets

## Results
- Final validation accuracy: ~95%
- Model generalizes well despite class imbalance

## Requirements
```bash
pip install tensorflow pandas scikit-learn matplotlib