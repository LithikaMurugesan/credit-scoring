import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# ✅ Import from your project
from data.data_generator import FEATURE_NAMES
from privacy.dp import clip_gradients, add_dp_noise


# ═════════════════════════════════════════════════════════════════════════════
# LOCAL TRAINING (Client-side training for one bank)
# ═════════════════════════════════════════════════════════════════════════════

def local_train(model, df, local_epochs, lr, use_dp, noise_mult, max_norm, batch_size=64):
    scaler = StandardScaler()

    # Prepare data
    X = scaler.fit_transform(df[FEATURE_NAMES].values).astype(np.float32)
    y = df["default"].values.astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    model.train()
    total_loss = 0.0
    steps = 0

    # Training loop
    for _ in range(local_epochs):
        for Xb, yb in loader:
            optimizer.zero_grad()

            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()

            # Apply Differential Privacy
            if use_dp:
                clip_gradients(model, max_norm)
                add_dp_noise(model, noise_mult, max_norm, len(Xb))

            optimizer.step()

            total_loss += loss.item()
            steps += 1

    avg_loss = total_loss / max(steps, 1)

    return avg_loss, len(df), scaler


# ═════════════════════════════════════════════════════════════════════════════
# MODEL EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, df, scaler):
    # Transform data using same scaler
    X = scaler.transform(df[FEATURE_NAMES].values).astype(np.float32)
    y = df["default"].values

    model.eval()
    with torch.no_grad():
        probs = model(torch.from_numpy(X)).numpy()

    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y, preds)

    try:
        auc = roc_auc_score(y, probs)
    except:
        auc = 0.5  # fallback if AUC fails

    return acc, auc
