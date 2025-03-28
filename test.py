import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import os

# training model on my own pc
device = torch.device("cpu")

transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), 
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),   
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=1, output_padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=1, output_padding=2),    
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




if __name__ == "__main__":
    print("Preping model")
    training_data_pasta = prep_training_data("pasta/train/good")
    training_data_screws = prep_training_data("screws/train/good")
    Autoencoder = Autoencoder()

    print("Training model for pasta")

    train_loader_pasta = DataLoader(training_data_pasta, batch_size=32, shuffle=True)
    loss_pasta = train_autoencoder(Autoencoder, train_loader_pasta, num_epochs=100,model_name="autoencoder_pasta.pth")
    train_loader_screws = DataLoader(training_data_screws, batch_size=32, shuffle=True)
    loss_screws = train_autoencoder(Autoencoder, train_loader_pasta, num_epochs=100,model_name="autoencoder_screws.pth")

    for epoch, loss in loss_pasta.items():
        print(f"Epoch {epoch}: Loss {loss:.4f}")  

    for epoch, loss in loss_screws.items():
        print(f"Epoch {epoch}: Loss {loss:.4f}")  

  