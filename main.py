import torch
import torch.nn.functional as F

from src.data_loader import load_elliptic_data
from src.models.hybrid import HybridModel
from src.utils import (
    set_seed,
    create_masks,
    compute_class_weights,
    evaluate_model,
    get_risk_scores,
    generate_alerts
)


def train(model, data, optimizer, class_weights):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    loss = F.nll_loss(
        out[data.train_mask],
        data.y[data.train_mask],
        weight=class_weights
    )

    loss.backward()
    optimizer.step()

    return loss.item()


def validate(model, data):
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum().item()
    total = data.val_mask.sum().item()

    return correct / total


def main():
    print("Starting Bitcoin AML GNN Project")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = load_elliptic_data()
    data = create_masks(data)
    data = data.to(device)

    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")

    # Model
    model = HybridModel(input_dim=data.x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # Class weights
    class_weights = compute_class_weights(data.y).to(device)

    best_val_acc = 0

    # Training
    for epoch in range(1, 51):
        loss = train(model, data, optimizer, class_weights)
        val_acc = validate(model, data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

    print("\nTraining Complete")

    # Load best model
    model.load_state_dict(torch.load("best_model.pth"))

    # Evaluation
    print("\nTest Set Evaluation:")
    evaluate_model(model, data, data.test_mask)

    # Simulation
    print("\nRunning Fraud Detection Simulation...")

    scores = get_risk_scores(model, data)
    alerts = generate_alerts(scores, threshold=0.9)

    print(f"Flagged Transactions: {alerts.sum().item()}")

    print("\nProject Execution Complete")


if __name__ == "__main__":
    main()