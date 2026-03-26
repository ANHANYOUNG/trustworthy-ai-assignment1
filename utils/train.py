import torch
from tqdm import tqdm

def eval_accuracy(model, data_loader, *, device, desc):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        eval_bar = tqdm(data_loader, desc=desc)
        for images, labels in eval_bar:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)            # (64, 10)
            preds = logits.argmax(dim=1)      # (64,)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            eval_bar.set_postfix(acc=f"{correct / total:.4f}")

    # accuracy = correct / total
    return correct / total


def train_classifier(model, train_loader, test_loader, *, device, epochs=3, lr=1e-3):
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # log
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        train_bar = tqdm(train_loader, desc=f"train {epoch + 1}/{epochs}")
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            # loss.item: batch avg loss
            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

            train_bar.set_postfix(
                loss=f"{running_loss / running_total:.4f}",
                acc=f"{running_correct / running_total:.4f}",
            )

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        test_acc = eval_accuracy(
            model,
            test_loader,
            device=device,
            desc=f"eval {epoch + 1}/{epochs}",
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(
            f"epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"test_acc={test_acc:.4f}"
        )

    return model, history
