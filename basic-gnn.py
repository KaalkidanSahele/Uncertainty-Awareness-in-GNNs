import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from models import GCN_MC

def main():
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN_MC(dataset.num_node_features, 64, dataset.num_classes, dropout=0.5).to(device)
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train():
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
        return loss.item()

    def test():
        model.eval()
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            correct = pred[mask].eq(data.y[mask]).sum().item()
            accs.append(correct / int(mask.sum().item()))
        return accs

    for epoch in range(1, 301):
        loss = train()
        if epoch % 50 == 0:
            train_acc, val_acc, test_acc = test()
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    torch.save(model.state_dict(), 'model.pth')
    print("Final test:", test()[2])

if __name__ == '__main__':
    main()
