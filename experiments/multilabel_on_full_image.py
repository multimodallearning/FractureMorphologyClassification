import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import GrazerFracturesImageDataset_MultiLabelWholeImage
from models import ResNetClassifier_MultiLabel
from utils import MORPHOLOGY_AO

os.environ["CUDA_VISIBLE_DEVICES"]="0"
data_folder = "data"
image_path = data_folder + "/preprocessed_images"
label_path = data_folder + "/dataset.csv"
save_folder = "data/output"
os.makedirs(save_folder, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 224)),
    transforms.Normalize(mean=[0.3505533917353781], std=[0.22763733675869177])  
])

# Create dataset
dataset = GrazerFracturesImageDataset_MultiLabelWholeImage(path_to_images=image_path, path_to_labels=label_path, transform=transform)

torch.save(dataset, f"{save_folder}/dataset.pt")
print("finished dataset")

stacked = torch.stack(dataset.labels, dim=0)
counts = stacked.sum(dim=0)

with open(save_folder+"/classcounts.txt", "w") as f:
    for i, count in enumerate(counts):
        f.write(f"Class {i}: {count.item()}\n")

# Extract indices and labels
indices = torch.arange(len(dataset))
lab = dataset.labels

# First, split into train (80%) and test (20%)
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42
)

# Extract labels for the test set
test_labels = [lab[i] for i in test_idx]

# Second, split test set into validation (50%) and test (50%)
val_idx, test_idx = train_test_split(
    test_idx, test_size=0.5, random_state=42
)

# Create Subset datasets
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

num_workers = 8

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers)

# Create DataLoaders
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers)

torch.save(train_idx, os.path.join(save_folder, "train_idx.pt"))
torch.save(val_idx, os.path.join(save_folder, "val_idx.pt"))
torch.save(test_idx, os.path.join(save_folder, "test_idx.pt"))

labels = torch.stack(train_dataset.dataset.labels)
pos_counts = labels.sum(dim=0)
neg_counts = labels.shape[0] - pos_counts

# pos_weight = neg / pos 
class_weights = neg_counts / (pos_counts + 1e-6) 


# Checkpoint-Callback: speichert das Modell mit bestem val_f1
checkpoint_callback = ModelCheckpoint(monitor="val_f1", mode="max", save_top_k=1,filename="best-f1-{epoch:02d}-{val_f1:.4f}-baseline")

num_classes = len(dataset.labels[0])
# Initialize the model
model = ResNetClassifier_MultiLabel(num_classes=num_classes, class_weights=class_weights, save_folder=save_folder, pretrained=False)

# Initialize trainer
trainer = pl.Trainer(
    max_epochs=200,
    log_every_n_steps=20,
    accelerator="auto",
    devices=1,  # Use multiple GPUs if available
    callbacks=[checkpoint_callback]
)

# Train the model
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


best_model_path = checkpoint_callback.best_model_path
print(f"Best model saved at: {best_model_path}")

# load best weights
best_model = ResNetClassifier_MultiLabel.load_from_checkpoint(best_model_path, num_classes=num_classes)
torch.save(best_model.state_dict(), f"{save_folder}/bestmodelweights.pth")

# Test the model
results = trainer.test(model, dataloaders=test_loader)

with open(save_folder+"/test_results.txt", "w") as f:
    for r in results:
        for k, v in r.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")


model.plot_loss()

plt.plot(model.val_acc_epoch, label="Validation Accuracy")
plt.plot(model.val_f1_epoch, label="Validation F1")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.savefig(save_folder+"/validationplot.pdf") 
plt.close()


#plot confusion matrix for test data
from sklearn.metrics import confusion_matrix

labels = torch.cat(model.all_labels)#.cpu()
preds = torch.cat(model.all_preds)#.cpu()


acc_per_class = Accuracy(task="multilabel", num_labels=num_classes, average=None).to(labels.device)(preds, labels).cpu().numpy()
f1_per_class = F1Score(task="multilabel", num_labels=num_classes, average=None).to(labels.device)(preds, labels).cpu().numpy()
precision_per_class = Precision(task="multilabel", num_labels=num_classes, average=None).to(labels.device)(preds, labels).cpu().numpy()
recall_per_class = Recall(task="multilabel", num_labels=num_classes, average=None).to(labels.device)(preds, labels).cpu().numpy()

acc_macro = Accuracy(task="multilabel", num_labels=num_classes, average="macro").to(labels.device)(preds, labels).cpu().numpy()
f1_macro = F1Score(task="multilabel", num_labels=num_classes, average="macro").to(labels.device)(preds, labels).cpu().numpy()
precision_macro = Precision(task="multilabel", num_labels=num_classes, average="macro").to(labels.device)(preds, labels).cpu().numpy()
recall_macro = Recall(task="multilabel", num_labels=num_classes, average="macro").to(labels.device)(preds, labels).cpu().numpy()

morphology_classes = list(MORPHOLOGY_AO.keys())

lines = []
lines.append("Metriken pro Klasse:\n")
for i, class_name in enumerate(morphology_classes):
    lines.append(f"Klasse: {class_name}")
    lines.append(f"  Accuracy : {acc_per_class[i]:.4f}")
    lines.append(f"  F1 Score : {f1_per_class[i]:.4f}")
    lines.append(f"  Precision: {precision_per_class[i]:.4f}")
    lines.append(f"  Recall   : {recall_per_class[i]:.4f}")
    lines.append("-" * 30)

lines.append("\nMean over all classes:\n")
lines.append(f"Accuracy: {acc_macro:.4f}")
lines.append(f"F1-Score: {f1_macro:.4f}")
lines.append(f"Precision: {precision_macro:.4f}")
lines.append(f"Recall: {recall_macro:.4f}")

with open(save_folder+"/metrics_results.txt", "w") as f:
    f.write("\n".join(lines))

# Compute confusion matrix
for i in range(num_classes):
    cm = confusion_matrix(labels[:, i], preds[:, i])
    print(f"Confusion matrix for class {i}:\n{cm}")

    # Heatmap
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title(f"Confusion Matrix for Class {i}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_folder+f"/confmatrix{i}.pdf") 
    plt.close()
