import time
import torch
from torch import nn
from torch.optim import Adam
from src.loaders.config import ConfigModel
from torch.utils.data import DataLoader

class BaselineTrainer:

    def __init__(
        self,
        config: ConfigModel,
        model: nn.Module,
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
        self.criterion = nn.BCEWithLogitsLoss()


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

                self.optimizer.zero_grad()

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                loss.backward()
                self.optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size

                probabilities = torch.sigmoid(logits)
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
                average_loss, accuracy = self._evaluate(device, self.validation_loader)
                print(
                    f"Validation Loss: {average_loss:.4f}, "
                    f"Validation Accuracy: {accuracy:.4f}",
                )
        
        average_loss, accuracy = self._evaluate(device, self.test_loader)
        print(
            f"Test Loss: {average_loss:.4f}, "
            f"Test Accuracy: {accuracy:.4f}",
        )

    def _evaluate(self, device: torch.device, loader: DataLoader):
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= 0.5).float()
                correct += (predictions == labels).sum().item()
                total += batch_size

        if total == 0:
            raise ValueError("Cannot evaluate an empty data loader.")

        average_loss = total_loss / total
        accuracy = correct / total
        return average_loss, accuracy