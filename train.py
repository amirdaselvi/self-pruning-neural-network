import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import SelfPruningNet
from utils import get_cifar10_loaders, save_results


# ── Train One Model ─────────────────────────────────────────────

def train_model(lambda_val, train_loader, test_loader, device, epochs=30):
    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)

            # Classification loss
            cls_loss = criterion(outputs, y)

           
            sparsity_loss = 0
            for layer in model.prunable_layers():
                sparsity_loss += torch.sum(torch.abs(layer.get_gates()))

            loss = cls_loss + lambda_val * sparsity_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        acc = 100. * correct / total

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss:.4f} | Acc: {acc:.2f}%")

    # ── Evaluation ─────────────────────────────────────────────

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)

            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    test_acc = 100. * correct / total

    # Sparsity
    sparsity = model.overall_sparsity() * 100
    gates = model.all_gates().cpu().numpy()

    print(f"\n✓ Final Accuracy : {test_acc:.2f}%")
    print(f"✓ Sparsity Level : {sparsity:.2f}%\n")

    return {
        "lambda": lambda_val,
        "accuracy": test_acc,
        "sparsity": sparsity,
        "gate_values": gates
    }


# ── MAIN ───────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)

    train_loader, test_loader = get_cifar10_loaders()

    # 🔥 Stronger lambdas (important for pruning)
    lambdas = [0.01, 0.05, 0.1]

    results = []

    for lam in lambdas:
        print("\n" + "="*50)
        print(f" λ = {lam:.5f}")
        print("="*50)

        res = train_model(lam, train_loader, test_loader, device)
        results.append(res)

    save_results(results)


# ── RUN ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("STARTING TRAINING...\n")
    main()