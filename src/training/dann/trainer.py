import torch
import time
from src.loaders.config import ConfigModel
from torch.optim import Adam
from torch.utils.data import DataLoader

class DANNTrainer:

    def __init__(
        self,
        config: ConfigModel,
        model: torch.nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
    ):

        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
        )
        self.label_criterion = torch.nn.BCEWithLogitsLoss()
        self.domain_criterion = torch.nn.CrossEntropyLoss()

    def run(self):
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(device)

        for epoch in range(self.config.training.epochs):
            epoch_start = time.perf_counter()
            self.model.train()

            total_loss = 0.0
            correct = 0
            total = 0

            for batch in self.train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                domains = batch["domain"].to(device)

                self.optimizer.zero_grad()

                label_logits, domain_logits = self.model(images)
                label_loss = self.label_criterion(label_logits, labels)
                domain_loss = self.domain_criterion(domain_logits, domains)
                loss = label_loss + self.config.domain_adaptation.domain_loss_weight * domain_loss

                loss.backward()
                self.optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size

                probabilities = torch.sigmoid(label_logits)
                predictions = (probabilities >= 0.5).float()

                correct += (predictions == labels).sum().item()
                total += batch_size
                
            average_loss = total_loss / total
            accuracy = correct / total
            epoch_time = time.perf_counter() - epoch_start
            
            print(
                f"Epoch {epoch + 1}/{self.config.training.epochs}, "
                f"Loss: {average_loss:.4f}, "
                f"Accuracy: {accuracy:.4f}, "
                f"Time: {epoch_time:.1f}s",
            )

            if (epoch + 1) % self.config.training.validation_interval == 0:
                metrics = self._evaluate(device, self.validation_loader, include_domain_loss=True)
                print(
                    f"Validation Total Loss: {metrics['total_loss']:.4f}, "
                    f"Validation Label Loss: {metrics['label_loss']:.4f}, "
                    f"Validation Domain Loss: {metrics['domain_loss']:.4f}, "
                    f"Validation Accuracy: {metrics['label_accuracy']:.4f}",
                )
        metrics = self._evaluate(device, self.test_loader, include_domain_loss=False)
        print(
            f"Target Test Label Loss: {metrics['label_loss']:.4f}, "
            f"Target Test Accuracy: {metrics['label_accuracy']:.4f}",
        )

    def _evaluate(
        self,
        device: torch.device,
        loader: DataLoader,
        include_domain_loss: bool,
    ):

        self.model.eval()

        total_loss = 0.0
        total_label_loss = 0.0
        total_domain_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                domains = batch["domain"].to(device)

                label_logits, domain_logits = self.model(images)
                label_loss = self.label_criterion(label_logits, labels)
                domain_loss = self.domain_criterion(domain_logits, domains)

                if include_domain_loss:
                    loss = label_loss + self.config.domain_adaptation.domain_loss_weight * domain_loss
                else:
                    loss = label_loss

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size

                total_label_loss += label_loss.item() * batch_size
                total_domain_loss += domain_loss.item() * batch_size

                probabilities = torch.sigmoid(label_logits)
                predictions = (probabilities >= 0.5).float()

                correct += (predictions == labels).sum().item()
                total += batch_size

        if total == 0:
            raise ValueError("Cannot evaluate an empty data loader.")

        return {
            "total_loss": total_loss / total,
            "label_loss": total_label_loss / total,
            "domain_loss": total_domain_loss / total,
            "label_accuracy": correct / total,
        }