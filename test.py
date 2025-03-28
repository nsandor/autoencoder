import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import os
import argparse

# training model on my own pc
device = torch.device("cpu")

transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Argument passing
parser = argparse.ArgumentParser(description="Run with arg train to train and or detect to detect")
parser.add_argument("args", nargs="*")
parsed_args = parser.parse_args()
train = "train" in parsed_args.args 
detect = "detect" in parsed_args.args


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1028, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), 
 
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1028, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),   
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    
            nn.Sigmoid()  
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(model, train_loader, num_epochs=20, lr=0.001, model_name="autoencoder"):

    model.to(device)
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_dict = {}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_dict[epoch + 1] = avg_loss

    torch.save(model.state_dict(), model_name)
    print(f"model saved to {model_name}")

    return loss_dict

def prep_training_data(path):

    data_path = os.path.join('./ece471_data/dataset/', path)
    good_dataset = datasets.ImageFolder(root=data_path, transform=transforms)

    print(f"Total samples from {path} {len(good_dataset)}")
    return good_dataset


def detect_anomalties(model, dataset):
    model.eval()
    criterion = nn.MSELoss()
    reconstruction_losses = {"good": [], "bad": []}

    with torch.no_grad():
        for images, label in dataset:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images).item()

            if label == 1:
                reconstruction_losses["good"].append(loss)
            if label == 0:
                reconstruction_losses["bad"].append(loss)
    
    return reconstruction_losses

if __name__ == "__main__":


    if train:
        print("Preping models")
        training_data_pasta = prep_training_data("pasta/train/good")
        training_data_screws = prep_training_data("screws/train/good")
        Autoencoder = Autoencoder()

        print("Training model for pasta")
        train_loader_pasta = DataLoader(training_data_pasta, batch_size=32, shuffle=True)
        loss_pasta = train_autoencoder(Autoencoder, train_loader_pasta, num_epochs=300,model_name="autoencoder_pasta.pth")

        print("Training model for screws")
        train_loader_screws = DataLoader(training_data_screws, batch_size=32, shuffle=True)
        loss_screws = train_autoencoder(Autoencoder, train_loader_pasta, num_epochs=300,model_name="autoencoder_screws.pth")

        for epoch, loss in loss_pasta.items():
            print(f"Epoch {epoch}: Loss {loss:.4f}")  

        for epoch, loss in loss_screws.items():
            print(f"Epoch {epoch}: Loss {loss:.4f}")

    if detect:
        print("Loading Models")
        model_screws = Autoencoder()
        model_pasta = Autoencoder()
        model_screws.load_state_dict(torch.load("autoencoder_screws.pth"))
        model_pasta.load_state_dict(torch.load("autoencoder_pasta.pth"))

        print("Loading test data")
        test_data_pasta = datasets.ImageFolder(root='./ece471_data/dataset/pasta/test', transform=transforms)
        test_data_screws = datasets.ImageFolder(root='./ece471_data/dataset/screws/test', transform=transforms)

        test_loader_pasta = DataLoader(test_data_pasta, batch_size=32, shuffle=False)
        test_loader_screw = DataLoader(test_data_screws, batch_size=32, shuffle=False)

        losses_pasta = detect_anomalties(model_pasta, test_data_pasta)
        losses_screws = detect_anomalties(model_screws, test_data_screws)
        #print("Screw losses:", losses_screws)
        #print("Pasta losses:", losses_pasta)

        # Getting a threshold for anomalies
        screw_loss_list_good = list(losses_screws["good"])
        pasta_loss_list_good = list(losses_screws["good"])
        screw_loss_threshold = max(screw_loss_list_good)
        pasta_loss_threshold = max(pasta_loss_list_good)

        # Detecting anomalies
        pasta_loss_list_bad = list(losses_screws["bad"])
        screw_loss_list_bad = list(losses_pasta["bad"])

        true_positives = 0
        for element in screw_loss_list_bad:
            if element > screw_loss_threshold:
                true_positives += 1
        false_negatives = len(screw_loss_list_bad) - true_positives
        print(f'For screws:, True Positives:{true_positives}, False Negatives:{false_negatives}')

        true_positives = 0
        for element in pasta_loss_list_bad:
            if element > pasta_loss_threshold:
                true_positives += 1
        false_negatives = len(pasta_loss_list_bad) - true_positives
        print(f'For pasta:, True Positives:{true_positives}, False Negatives:{false_negatives}')
