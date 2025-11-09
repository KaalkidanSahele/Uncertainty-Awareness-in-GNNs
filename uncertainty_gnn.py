# src/uncertainty_gnn.py
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

MODEL_PATH = "model_uncert.pth"
T_MC = 30  # number of MC dropout forward passes (use 20-50 for quick runs)

# --------- Model ---------
class GCN_MC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x  # raw logits

# --------- Utilities ---------
def train_model(model, data, device, epochs=200, lr=0.01, wd=5e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.to(device)
    data = data.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == 1:
            train_acc, val_acc, test_acc = eval_model(model, data, device)
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
    return model

def eval_model(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / int(mask.sum().item()))
    return accs  # train, val, test

def mc_dropout_predict(model, data, T=30, device=None):
    """Run T stochastic forward passes with dropout active. Return mean probs, var, epistemic, entropy."""
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    model.train()  # enable dropout during inference
    preds = []
    # We disable grad to reduce memory & speed; dropout still works in train() mode
    with torch.no_grad():
        for _ in trange(T, desc="MC Dropout"):
            logits = model(data.x.to(device), data.edge_index.to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()  # [N, C]
            preds.append(probs)
    preds = np.stack(preds, axis=0)  # [T, N, C]
    mean = preds.mean(axis=0)       # [N, C]
    var = preds.var(axis=0)         # [N, C]
    # simple epistemic summary per node: mean of class-wise variances
    epistemic = var.mean(axis=1)    # [N]
    # predictive entropy
    entropy = -np.sum(mean * np.log(mean + 1e-12), axis=1)  # [N]
    return mean, var, epistemic, entropy

# --------- Main script ---------
def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # dataset
    dataset = Planetoid(root="data/Cora", name="Cora")
    data = dataset[0]
    N = data.num_nodes
    print("Loaded Cora: N nodes =", N, "num features =", dataset.num_node_features, "num classes =", dataset.num_classes)

    # model
    model = GCN_MC(dataset.num_node_features, 64, dataset.num_classes, dropout=0.5)
    # load or train
    if os.path.exists(MODEL_PATH):
        print("Loading model from", MODEL_PATH)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("No checkpoint found. Training model (this may take a minute)...")
        model = train_model(model, data, device, epochs=200)
        torch.save(model.state_dict(), MODEL_PATH)
        print("Saved model to", MODEL_PATH)

    # Quick eval
    train_acc, val_acc, test_acc = eval_model(model, data, device)
    print(f"Final eval -> Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    # MC Dropout inference
    mean, var, epistemic, entropy = mc_dropout_predict(model, data, T=T_MC, device=device)
    preds_mean = mean.argmax(axis=1)  # [N] predicted labels by mean probs
    true = data.y.cpu().numpy()

    # Which nodes are test nodes?
    test_mask = data.test_mask.cpu().numpy().astype(bool)
    test_idx = np.where(test_mask)[0]

    # Per-node correctness
    correct = (preds_mean == true)
    correct_test = correct[test_idx]
    epistemic_test = epistemic[test_idx]
    entropy_test = entropy[test_idx]

    # Print simple stats
    print("MC Dropout: T =", T_MC)
    print("Epistemic (test) mean for CORRECT nodes: {:.6f}".format(epistemic_test[correct_test].mean() if correct_test.any() else float('nan')))
    print("Epistemic (test) mean for INCORRECT nodes: {:.6f}".format(epistemic_test[~correct_test].mean() if (~correct_test).any() else float('nan')))

    # AUROC: can uncertainty detect misclassification? (higher uncertainty => misclassified)
    y_true = (~correct_test).astype(int)  # 1 = misclassified
    try:
        auroc_epi = roc_auc_score(y_true, epistemic_test)
        auroc_ent = roc_auc_score(y_true, entropy_test)
        print(f"AUROC (epistemic for misclassification detection): {auroc_epi:.4f}")
        print(f"AUROC (predictive entropy for misclassification detection): {auroc_ent:.4f}")
    except ValueError:
        print("AUROC could not be computed (maybe only one class present in test labels).")

    # Print top uncertain test nodes
    top_k = 10
    order = np.argsort(-epistemic_test)  # descending
    top_nodes = test_idx[order[:top_k]]
    print("Top uncertain test nodes (node_id, true_label, pred_label, epistemic):")
    for nid in top_nodes:
        print(nid, int(true[nid]), int(preds_mean[nid]), float(epistemic[nid]))

    # Plots
    os.makedirs("plots", exist_ok=True)

    # 1) Histogram epistemic: correct vs incorrect
    plt.figure(figsize=(6,4))
    plt.hist(epistemic_test[correct_test], bins=25, alpha=0.7, label="correct")
    plt.hist(epistemic_test[~correct_test], bins=25, alpha=0.7, label="incorrect")
    plt.xlabel("Epistemic uncertainty (mean class var)")
    plt.ylabel("Count (test nodes)")
    plt.legend()
    plt.title("Epistemic uncertainty: correct vs incorrect (test nodes)")
    plt.tight_layout()
    plt.savefig("plots/epistemic_hist.png", dpi=150)
    print("Saved plots/epistemic_hist.png")

    # 2) Scatter: predictive entropy vs node index (colored by correctness)
    plt.figure(figsize=(8,4))
    x = np.arange(len(test_idx))
    colors = ['tab:green' if c else 'tab:red' for c in correct_test]
    plt.scatter(x, entropy_test, c=colors, s=12)
    plt.xlabel("Test node (index in test set)")
    plt.ylabel("Predictive entropy")
    plt.title("Predictive entropy for test nodes (green=correct, red=incorrect)")
    plt.tight_layout()
    plt.savefig("plots/entropy_scatter.png", dpi=150)
    print("Saved plots/entropy_scatter.png")

    print("Done. Inspect plots/ and adjust T_MC for smoother estimates (increase T_MC).")

if __name__ == "__main__":
    main()
