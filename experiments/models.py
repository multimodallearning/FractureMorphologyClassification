
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from torchmetrics import ConfusionMatrix
from torchvision import models
from timm.scheduler import CosineLRScheduler
import matplotlib.pyplot as plt
import numpy as np

"""
Model for binary fracture classification
"""
class ResNetClassifier_BinaryFractureClassification(pl.LightningModule):
    def __init__(self, num_classes, class_weights, save_folder, p=0.1, pretrained=True):
        """Init for the model

        Args:
            num_classes (int): Number of classes
            class_weights (list): Weights for each class for the loss
            save_folder (str): Saving folder
            p (float, optional): Dropout rate. Defaults to 0.1.
            pretrained (bool, optional): If a pretrained ResNet should be used. Defaults to True.
        """
        super().__init__()
        self.save_hyperparameters()

        # Load pre-trained ResNet model
        self.model = models.resnet18(pretrained=pretrained)
        self.model.conv1.stride = (1,1)
        self.model.maxpool = nn.Identity()
        
        # Replace the final fully connected layer to match the number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(p), nn.Linear(in_features, num_classes))

        # Define loss function and accuracy metric
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.f1score = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        self.all_preds = []
        self.all_labels = []

        self.train_loss_epoch = []
        self.val_loss_epoch = []

        self.val_acc_epoch = []
        self.val_f1_epoch = []

        self.class_weights = class_weights
        self.save_folder = save_folder


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        f1 = self.f1score(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        f1 = self.f1score(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.all_preds.append(preds)
        self.all_labels.append(y)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        f1 = self.f1score(logits, y)
        precision = self.precision(logits, y)
        recall = self.recall(logits, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_precision", precision, prog_bar=True)
        self.log("test_recall", recall, prog_bar=True)

    def on_train_epoch_end(self):
        # Save epoch loss
        self.train_loss_epoch.append(self.trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self):
        # Save epoch loss
        self.val_loss_epoch.append(self.trainer.callback_metrics["val_loss"].item())
        self.val_acc_epoch.append(self.trainer.callback_metrics["val_acc"].item())
        self.val_f1_epoch.append(self.trainer.callback_metrics["val_f1"].item())

    def on_test_end(self):
        preds = torch.cat(self.all_preds).cpu().numpy()
        labels = torch.cat(self.all_labels).cpu().numpy()

        import numpy as np
        np.save(f"{self.save_folder}/preds.npy", preds)
        np.save(f"{self.save_folder}/labels.npy", labels)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs,
                                        
                                    warmup_prefix=True)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, metric) -> None:
        scheduler.step(self.current_epoch)
        return optim.Adam(self.parameters(), lr=1e-4)


    def plot_loss(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_loss_epoch, label="Train Loss")
        plt.plot(self.val_loss_epoch, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training & Validation Loss")
        plt.savefig(self.save_folder+"/lossplot.pdf") 
        plt.close()

"""
Model for multi label classification on the full-image
"""
class ResNetClassifier_MultiLabel(pl.LightningModule):
    def __init__(self, num_classes, class_weights, save_folder, p=0.1, pretrained=True):
        """Init for the model

        Args:
            num_classes (int): Number of classes
            class_weights (list): Weights for each class for the loss
            save_folder (str): Saving folder
            p (float, optional): Dropout rate. Defaults to 0.1.
            pretrained (bool, optional): If a pretrained ResNet should be used. Defaults to True.
        """
        super().__init__()
        self.save_hyperparameters()

        # Load pre-trained ResNet model
        self.model = models.resnet18(pretrained=pretrained)
        
        # Replace the final fully connected layer to match the number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(p), nn.Linear(in_features, num_classes))

        # Define loss function and accuracy metric
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

        
        self.accuracy = Accuracy(task="multilabel", num_labels=num_classes, average="macro")
        self.f1score = F1Score(task="multilabel", num_labels=num_classes, average="macro")
        self.precision = Precision(task="multilabel", num_labels=num_classes, average="macro")
        self.recall = Recall(task="multilabel", num_labels=num_classes, average="macro")

        self.all_preds = []
        self.all_labels = []

        self.train_loss_epoch = []
        self.val_loss_epoch = []

        self.val_acc_epoch = []
        self.val_f1_epoch = []

        self.save_folder = save_folder


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        f1 = self.f1score(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        f1 = self.f1score(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch    
        logits = self(x)        
        
        probs = torch.sigmoid(logits) # convert logits to probabilities
        preds = (probs >= 0.5).int() # binarize predictions
        
        self.all_preds.append(preds.cpu())
        self.all_labels.append(y.cpu())
        
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y.int())
        f1 = self.f1score(logits, y.int())
        precision = self.precision(logits, y.int())
        recall = self.recall(logits, y.int())
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_precision", precision, prog_bar=True)
        self.log("test_recall", recall, prog_bar=True)

    def on_train_epoch_end(self):
        # Save epoch loss
        self.train_loss_epoch.append(self.trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self):
        # Save epoch loss
        self.val_loss_epoch.append(self.trainer.callback_metrics["val_loss"].item())
        self.val_acc_epoch.append(self.trainer.callback_metrics["val_acc"].item())
        self.val_f1_epoch.append(self.trainer.callback_metrics["val_f1"].item())

    def on_test_end(self):
        preds = torch.cat(self.all_preds).cpu().numpy()
        labels = torch.cat(self.all_labels).cpu().numpy()

        import numpy as np
        np.save(f"{self.save_folder}/preds.npy", preds)
        np.save(f"{self.save_folder}/labels.npy", labels)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs,
                                        
                                    warmup_prefix=True)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, metric) -> None:
        scheduler.step(self.current_epoch)
        return optim.Adam(self.parameters(), lr=1e-4)

    def plot_loss(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_loss_epoch, label="Train Loss")
        plt.plot(self.val_loss_epoch, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training & Validation Loss")
        plt.savefig(self.save_folder+"/lossplot.pdf") 
        plt.close()

"""
Model for multi-class classification on patches
"""
class ResNetClassifier_MultiClass(pl.LightningModule):
    def __init__(self, num_classes, class_weights, save_folder, p=0.1, pretrained=True):
        """Init for the model

        Args:
            num_classes (int): Number of classes
            class_weights (list): Weights for each class for the loss
            save_folder (str): Saving folder
            p (float, optional): Dropout rate. Defaults to 0.1.
            pretrained (bool, optional): If a pretrained ResNet should be used. Defaults to True.
        """
        super().__init__()
        self.save_hyperparameters()

        # Load pre-trained ResNet model
        self.model = models.resnet18(pretrained=pretrained)
        self.model.conv1.stride = (1,1)
        self.model.maxpool = nn.Identity()
        
        # Replace the final fully connected layer to match the number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(p), nn.Linear(in_features, num_classes))

        # Define loss function and accuracy metric
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.f1score = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)  

        self.all_preds = []
        self.all_labels = []

        self.train_loss_epoch = []
        self.val_loss_epoch = []

        self.val_acc_epoch = []
        self.val_f1_epoch = []

        self.save_folder = save_folder


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        f1 = self.f1score(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        f1 = self.f1score(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.all_preds.append(preds)
        self.all_labels.append(y)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        f1 = self.f1score(logits, y)
        precision = self.precision(logits, y)
        recall = self.recall(logits, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_precision", precision, prog_bar=True)
        self.log("test_recall", recall, prog_bar=True)

    def on_train_epoch_end(self):
        self.train_loss_epoch.append(self.trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self):
        self.val_loss_epoch.append(self.trainer.callback_metrics["val_loss"].item())
        self.val_acc_epoch.append(self.trainer.callback_metrics["val_acc"].item())
        self.val_f1_epoch.append(self.trainer.callback_metrics["val_f1"].item())

    def on_test_end(self):
        preds = torch.cat(self.all_preds).cpu().numpy()
        labels = torch.cat(self.all_labels).cpu().numpy()

        np.save(f"{self.save_folder}/preds.npy", preds)
        np.save(f"{self.save_folder}/labels.npy", labels)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs,
                                        
                                    warmup_prefix=True)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, metric) -> None:
        scheduler.step(self.current_epoch)
        return optim.Adam(self.parameters(), lr=1e-4)


        
    def plot_loss(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_loss_epoch, label="Train Loss")
        plt.plot(self.val_loss_epoch, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training & Validation Loss")
        plt.savefig(self.save_folder+"/lossplot.pdf") 
        plt.close()