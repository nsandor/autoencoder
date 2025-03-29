import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score

# Hyperparameters
val_split = 0.1
batch_size = 1
training_epochs = 150

# training model on my own pc
if torch.cuda.is_available:
    device = "cuda:0"
else:
    device = "cpu"
print("pytorch using device: ", device)
device = torch.device(device)

transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Argument passing
parser = argparse.ArgumentParser(
    description="Run with arg train to train and or detect to detect"
)
parser.add_argument("args", nargs="*")
parsed_args = parser.parse_args()
train = "train" in parsed_args.args
detect = "detect" in parsed_args.args


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=0, output_padding=1
            ),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(
    model, train_loader, val_loader, num_epochs=20, lr=0.001, model_name="autoencoder"
):

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        val_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        for images, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            val_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        losses.append(avg_loss)
        validation_losses.append(avg_val_loss)
        print(
            "Epoch [{}/{}], Avg Loss: {}, Validation loss: {}".format(
                epoch + 1, num_epochs, avg_loss, avg_val_loss
            )
        )

    epochs = range(num_epochs)
    fig, ax = plt.subplots()
    ax.plot(epochs, losses, label="Training loss")
    ax.plot(epochs, validation_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    ax.set_yscale("log")
    plt.title("Loss by epoch for " + model_name)
    plt.legend()
    plt.savefig("trainingcurve_" + model_name)
    plt.clf()

    torch.save(model.state_dict(), model_name + ".pth")
    print(f"model saved to {model_name}" + ".pth")

    return losses


def prep_training_data(path):

    data_path = os.path.join("./ece471_data/dataset/", path)
    dataset = datasets.ImageFolder(root=data_path, transform=transforms)

    # Validation split (fixed seed for now)
    generator = torch.Generator().manual_seed(42)
    val_set, train_set = random_split(
        dataset, [val_split, (1 - val_split)], generator=generator
    )
    print(
        f"Samples from {path} split into {len(train_set)} training images and {len(val_set)} validation images"
    )
    return val_set, train_set


def plot_detector(loss_dict, threshold, name):
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.plot(loss_dict["good"], "o", label="Normal")
    ax.plot(loss_dict["bad"], "x", color="red", label="Anomaly")
    plt.title("True classes and decision boundary for " + name)
    plt.ylabel("Reconstruction loss")
    plt.axhline(threshold, 0, 1, linestyle="-", color="green", label="Threshold")
    plt.legend()
    plt.savefig("detect_" + name)
    plt.clf()


def check_performance(loss_dict, threshold):
    # Returns a confusion matrix array, precision, recall, and F1
    return True


def detect_anomalties(model, dataset):
    model.cuda()
    model.eval()
    criterion = nn.MSELoss()
    TrueLabels = []
    Loss = []

    with torch.no_grad():
        for images, label in dataset:
            images = images.cuda()
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images).item()
            Loss.append(loss)
            TrueLabels.append(label)

    return Loss,TrueLabels


if __name__ == "__main__":
    if train:
        print("Preparing data")
        validation_data_pasta, training_data_pasta = prep_training_data(
            "pasta/train/good"
        )
        validation_data_screws, training_data_screws = prep_training_data(
            "screws/train/good"
        )

        Autoencoder = Autoencoder()

        print("Training model for pasta")
        train_loader_pasta = DataLoader(
            training_data_pasta, batch_size=batch_size, shuffle=True
        )
        val_loader_pasta = DataLoader(
            validation_data_pasta, batch_size=1, shuffle=False
        )
        loss_pasta = train_autoencoder(
            Autoencoder,
            train_loader_pasta,
            val_loader_pasta,
            num_epochs=training_epochs,
            model_name="autoencoder_pasta",
        )

        print("Training model for screws")
        train_loader_screws = DataLoader(
            training_data_screws, batch_size=batch_size, shuffle=True
        )
        val_loader_screws = DataLoader(
            validation_data_screws, batch_size=1, shuffle=False
        )
        loss_screws = train_autoencoder(
            Autoencoder,
            train_loader_screws,
            val_loader_screws,
            num_epochs=training_epochs,
            model_name="autoencoder_screws",
        )

    if detect:
        print("Loading Models")
        model_screws = Autoencoder()
        model_pasta = Autoencoder()
        model_screws.load_state_dict(torch.load("autoencoder_screws.pth"))
        model_pasta.load_state_dict(torch.load("autoencoder_pasta.pth"))

        print("Loading test data")
        test_data_pasta = datasets.ImageFolder(
            root="./ece471_data/dataset/pasta/test/", transform=transforms
        )
        test_data_screws = datasets.ImageFolder(
            root="./ece471_data/dataset/screws/test/", transform=transforms
        )

        losses_pasta,Pasta_labels = detect_anomalties(model_pasta, test_data_pasta)
        losses_screws,Screw_labels = detect_anomalties(model_screws, test_data_screws)
        print(Pasta_labels)
        # Getting a threshold for anomalies
        #screw_loss_list_good = list(losses_screws["good"])
        #pasta_loss_list_good = list(losses_pasta["good"])
        #screw_loss_threshold = max(screw_loss_list_good)
        #pasta_loss_threshold = max(pasta_loss_list_good)
        print("Auroc Score, Pasta:", roc_auc_score(Pasta_labels,losses_pasta))
        print("Auroc Score, screws:", roc_auc_score(Screw_labels,losses_screws))
        #plot_detector(losses_pasta, pasta_loss_threshold, "Pasta")
        #plot_detector(losses_screws, screw_loss_threshold, "screw")
