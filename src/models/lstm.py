import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BitcoinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x must be 3D: (batch, sequence_length, input_size)
        if x.dim() == 2:
            # Single sample missing batch dim: (seq, features) → (1, seq, features)
            x = x.unsqueeze(0)

        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last_hidden = attn_out[:, -1, :]
        return self.fc(last_hidden)


class LSTMPredictor:
    def __init__(self, lookback=90, epochs=100, lr=0.001):
        self.lookback = lookback
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.input_size = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_sequences(self, data: np.ndarray, target_col: int = 0):
        """
        Create sliding window sequences for LSTM.
        
        Args:
            data: 2D array of shape (num_rows, num_features)
            target_col: column index for the prediction target
            
        Returns:
            X: 3D array of shape (num_sequences, lookback, num_features)
            y: 1D array of shape (num_sequences,)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])  # shape: (lookback, num_features)
            y.append(data[i, target_col])

        X = np.array(X)  # shape: (num_sequences, lookback, num_features)
        y = np.array(y)   # shape: (num_sequences,)

        print(f"   prepare_sequences: input {data.shape} → X {X.shape}, y {y.shape}")
        return X, y

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the LSTM model.
        
        Args:
            X_train: 3D array (num_sequences, lookback, num_features)
            y_train: 1D array (num_sequences,)
        """
        assert X_train.ndim == 3, f"X_train must be 3D, got {X_train.ndim}D with shape {X_train.shape}"

        self.input_size = X_train.shape[2]
        print(f"   LSTM input_size={self.input_size}, sequences={X_train.shape[0]}")

        self.model = BitcoinLSTM(self.input_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.FloatTensor(y_train).to(self.device)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.model(X_batch).squeeze(-1)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: 3D array (num_sequences, lookback, num_features)
               
        Returns:
            1D array of predictions
        """
        self.model.eval()

        assert X.ndim == 3, f"X must be 3D, got {X.ndim}D with shape {X.shape}"
        assert X.shape[2] == self.input_size, (
            f"Feature mismatch: model expects {self.input_size} features, "
            f"got {X.shape[2]}. Shape: {X.shape}"
        )

        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            preds = self.model(X_t).squeeze(-1).cpu().numpy()
            return preds.flatten()