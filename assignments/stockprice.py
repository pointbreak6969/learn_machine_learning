import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
df = pd.read_csv('data/tesla_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Use 'Close' price as target
data = df[['Close']].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 60
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Split data (80% train, 20% test)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to tensors - X already has shape (batch, seq_len), add feature dim
X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # (batch, seq_len, 1)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test).unsqueeze(-1)
y_test = torch.FloatTensor(y_test)

# Custom Dataset
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = StockDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    return losses

# Evaluation function
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_device = X_test.to(device)
        predictions = model(X_test_device).cpu().numpy()

        # Inverse transform
        predictions_original = scaler.inverse_transform(predictions)
        y_test_original = scaler.inverse_transform(y_test.numpy())

        # Calculate RMSE
        rmse = np.sqrt(np.mean((predictions_original - y_test_original) ** 2))

        return predictions_original, y_test_original, rmse

# Train and evaluate all models
print("\n" + "="*50)
print("Training RNN Model...")
print("="*50)
rnn_model = RNNModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)
rnn_losses = train_model(rnn_model, train_loader, criterion, optimizer)
rnn_preds, rnn_actual, rnn_rmse = evaluate_model(rnn_model, X_test, y_test)
print(f"RNN Test RMSE: {rnn_rmse:.4f}")

print("\n" + "="*50)
print("Training GRU Model...")
print("="*50)
gru_model = GRUModel().to(device)
optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)
gru_losses = train_model(gru_model, train_loader, criterion, optimizer)
gru_preds, gru_actual, gru_rmse = evaluate_model(gru_model, X_test, y_test)
print(f"GRU Test RMSE: {gru_rmse:.4f}")

print("\n" + "="*50)
print("Training LSTM Model...")
print("="*50)
lstm_model = LSTMModel().to(device)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
lstm_losses = train_model(lstm_model, train_loader, criterion, optimizer)
lstm_preds, lstm_actual, lstm_rmse = evaluate_model(lstm_model, X_test, y_test)
print(f"LSTM Test RMSE: {lstm_rmse:.4f}")

# Summary
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
print(f"RNN  - Test RMSE: {rnn_rmse:.4f}")
print(f"GRU  - Test RMSE: {gru_rmse:.4f}")
print(f"LSTM - Test RMSE: {lstm_rmse:.4f}")

best_model = min([("RNN", rnn_rmse), ("GRU", gru_rmse), ("LSTM", lstm_rmse)],
                 key=lambda x: x[1])
print(f"\nBest Model: {best_model[0]} with RMSE: {best_model[1]:.4f}")

# Plot training losses
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rnn_losses, label='RNN', alpha=0.8)
plt.plot(gru_losses, label='GRU', alpha=0.8)
plt.plot(lstm_losses, label='LSTM', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot predictions vs actual
plt.subplot(1, 2, 2)
plt.plot(rnn_actual, label='Actual', alpha=0.8)
plt.plot(rnn_preds, label='RNN Prediction', alpha=0.8)
plt.plot(gru_preds, label='GRU Prediction', alpha=0.8)
plt.plot(lstm_preds, label='LSTM Prediction', alpha=0.8)
plt.xlabel('Time Steps')
plt.ylabel('Stock Price (USD)')
plt.title('Stock Price Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stock_prediction_results.png', dpi=150)
print("\nResults plot saved as 'stock_prediction_results.png'")

# Save models
torch.save(rnn_model.state_dict(), 'rnn_model.pth')
torch.save(gru_model.state_dict(), 'gru_model.pth')
torch.save(lstm_model.state_dict(), 'lstm_model.pth')
print("Models saved as .pth files")
