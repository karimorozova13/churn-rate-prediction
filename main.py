import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import matthews_corrcoef
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.validation import check_is_fitted

# Load the data
X_train = pd.read_csv('./datasets/train.csv')
X_test = pd.read_csv('./datasets/test.csv')

# Target column
y = X_train['target_class']

X_train.drop('target_class',axis=1, inplace=True)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y)

# Split the training data into train and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_resampled, y_train_resampled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_split, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_split.values, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val_split, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_split.values, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Update the DataLoader for the new training set
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 2. Define the Neural Network Model
class ChurnNet(nn.Module):
    def __init__(self, input_size):
        super(ChurnNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # Increased to 256 neurons
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)  # Dropout regularization
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x
    
# Initialize the model
input_size = X_train.shape[1]
model = ChurnNet(input_size)

# 3. Loss Function and Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

num_epochs = 50  # Train for more epochs
best_mcc = -1
patience = 10  # Increase patience

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor).squeeze().round().numpy()
        mcc_val_score = matthews_corrcoef(y_val_tensor.numpy(), val_preds)
        print(f'MCC on validation set after epoch {epoch+1}: {mcc_val_score:.4f}')

    if mcc_val_score > best_mcc:
        best_mcc = mcc_val_score
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Step the scheduler
    scheduler.step(mcc_val_score)

model.load_state_dict(torch.load('best_model.pth')) 
model.eval()

with torch.no_grad():
    val_preds = model(X_val_tensor).squeeze().round().numpy()
    mcc_val_score = matthews_corrcoef(y_val_tensor.numpy(), val_preds)
    print(f'Best Matthews Correlation Coefficient (MCC) on validation data: {mcc_val_score:.4f}')

# 6. Make Predictions on Test Data
with torch.no_grad():
    test_preds = model(X_test_tensor).squeeze().round().numpy()

client_ids = X_test.index 
results = pd.DataFrame({
    'ID': client_ids,
    'target': test_preds
})
results.to_csv('./datasets/submission.csv', index=False)