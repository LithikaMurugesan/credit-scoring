# Privacy-Preserving Credit Scoring
## Federated Learning + Differential Privacy — Full Demo (No Backend!)

---

## 🚀 Quick Start (3 steps)

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the app
streamlit run app.py

# Step 3: Open browser
# http://localhost:8501
```

---

## 📱 App Pages

| Page | What it shows |
|------|--------------|
| 🏠 Overview | Project architecture, bank profiles, concept explanations |
| 📊 Data Explorer | Non-IID distributions, correlation heatmaps, raw data |
| ⚙️ FL Training | **Real PyTorch training** with FedAvg + DP — live accuracy charts |
| 🔒 Privacy Analysis | ε vs accuracy curve, noise multiplier effects, Opacus explanation |
| 💳 Credit Predictor | Uses trained model to predict CIBIL score with gauge chart |

---

## 🛠️ What's Implemented (No Backend Needed!)

### Federated Learning (Flower-style, in-process)
- `CreditNet` — PyTorch neural network (3 hidden layers)
- `local_train()` — Per-bank training with DP
- `fed_avg()` — FedAvg weight aggregation
- All runs inside Streamlit session — no server needed!

### Differential Privacy (Opacus-style, manual)
- `clip_gradients()` — Per-sample gradient clipping
- `add_dp_noise()` — Gaussian noise injection
- `compute_epsilon()` — RDP-based ε approximation

### Non-IID Data
- 4 banks with different income, age, default rate distributions
- Generated synthetically — no real data needed

---

## 💡 Viva Q&A

**Q: Why no real Flower server?**
> Real `flwr.server` needs multiple processes/ports. We simulate the same FedAvg math in-process — the algorithm is identical.

**Q: Is the DP real?**
> Yes! Gradient clipping + Gaussian noise is real. The ε formula uses RDP accounting (same as Opacus internally).

**Q: What is Non-IID?**
> Each bank has a different data distribution (income range, age, default rate). This makes FL harder — the model must generalize across all.

**Q: What does FedAvg do?**
> Weighted average of all client model weights by dataset size. Simple but effective.

---

## Tech Stack
- **Python** + **Streamlit** — UI & orchestration  
- **PyTorch** — Neural network model  
- **Opacus** (manual implementation) — Differential privacy  
- **Flower** (simulated) — Federated coordination  
- **Plotly** — Interactive charts  
- **scikit-learn** — StandardScaler, metrics  
