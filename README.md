# Churn Prediction Model

This project aims to build a **churn prediction** model using a **neural network** in PyTorch. The goal of the project is to predict which customers are likely to stop using a company's services, based on historical data. The project utilizes **imbalanced learning techniques**, **feature scaling**, **neural network modeling**, and **early stopping** during training.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Training the Model](#2-training-the-model)
  - [3. Evaluation](#3-evaluation)
  - [4. Prediction on Test Data](#4-prediction-on-test-data)
- [Model Architecture](#model-architecture)
- [Early Stopping](#early-stopping)
- [Hyperparameter Optimization](#hyperparameter-optimization)

## Project Overview

Churn prediction helps businesses identify customers who are likely to churn (i.e., stop using their services). By using customer data (features), the project predicts a binary outcome: whether a customer will churn or not. The model is built using a neural network implemented with **PyTorch**, and training is conducted with a strategy to handle class imbalance using **SMOTE**.

## Features

- **Preprocessing pipeline**: Includes scaling, imputation, and oversampling for class imbalance.
- **Neural network model**: Multi-layer neural network using ReLU activations and Dropout for regularization.
- **Early stopping**: Prevents overfitting by monitoring the validation set performance.
- **Binary Classification**: Output is binary (`1` for churn and `0` for no churn).
- **Matthews Correlation Coefficient (MCC)**: Evaluation metric for model performance.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/karimorozova13/churn-rate-prediction.git
   cd churn-rate-prediction
   ```

2. **Install dependencies**:

   First, make sure you have `python3` and `pip` installed. Then install the dependencies using:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch**:

   Install PyTorch using the official installation guide based on your operating system and whether you have CUDA enabled GPU:

   ```bash
   pip install torch torchvision torchaudio
   ```

## Dataset

The dataset consists of customer records with various features. The target variable (`target_class`) indicates whether a customer has churned (`1`) or not (`0`). You should have two CSV files, `train.csv` and `test.csv`, in a `datasets` folder.

- `train.csv`: Used for training the model.
- `test.csv`: Used for making predictions.

Ensure the dataset follows the same format as in the example `train.csv` with a target column `target_class`.

## Usage

### 1. Data Preprocessing

1. **Imputation and Scaling**: The dataset is imputed and scaled using `StandardScaler` from `sklearn`.
2. **Handling Imbalance**: We use `SMOTE` to oversample the minority class to balance the dataset.

```python
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Handling imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y)
```

### 2. Training the Model

The model is a 3-layer neural network built using PyTorch. It is trained on the resampled training data using **Binary Cross Entropy Loss** as the loss function and **Adam optimizer** for optimization.

```python
model = ChurnNet(input_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
```

### 3. Evaluation

We evaluate the model using **Matthews Correlation Coefficient (MCC)**, which is suitable for imbalanced datasets.

```python
from sklearn.metrics import matthews_corrcoef

val_preds = model(X_val_tensor).squeeze().round().numpy()
mcc_val_score = matthews_corrcoef(y_val_tensor.numpy(), val_preds)
print(f'MCC on validation set: {mcc_val_score:.4f}')
```

### 4. Prediction on Test Data

Once the model is trained, we use it to predict the target variable for the test dataset and save the results as a CSV file.

```python
with torch.no_grad():
    test_preds = model(X_test_tensor).squeeze().round().numpy()

results = pd.DataFrame({'ID': client_ids, 'target': test_preds})
results.to_csv('./datasets/submission.csv', index=False)
```

## Model Architecture

The neural network consists of:

- **Input layer**: Number of neurons = number of features in the dataset.
- **Hidden layers**: Two fully connected layers with 256 and 128 neurons.
- **Output layer**: A single neuron with a sigmoid activation for binary classification.

The model also uses **ReLU activations** and **Dropout regularization** to prevent overfitting.

```python
class ChurnNet(nn.Module):
    def __init__(self, input_size):
        super(ChurnNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()
```

## Early Stopping

Early stopping is implemented to prevent overfitting. The model is evaluated on the validation set at each epoch, and training stops if the performance does not improve for a set number of epochs.

```python
best_mcc = -1
patience = 10
for epoch in range(num_epochs):
    # Training loop
    # Validation phase
    if mcc_val_score > best_mcc:
        best_mcc = mcc_val_score
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print('Early stopping')
            break
```

## Hyperparameter Optimization

The current model uses the following hyperparameters:

- **Learning rate**: 0.001
- **Batch size**: 64
- **Dropout rate**: 0.3
- **Patience**: 10 (for early stopping)

You can tune these hyperparameters based on your hardware and dataset.
