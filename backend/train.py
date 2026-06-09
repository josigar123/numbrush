from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from model import NumBrush

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data',
                               train=True,
                               download=True,
                               transform=transform)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              download=True,
                              transform=transform)

train_loader = DataLoader(train_dataset,
                          batch_size=64,
                          shuffle=True)

test_loader = DataLoader(test_dataset,
                         batch_size=64,
                         shuffle=False)

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

model = NumBrush()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

def train(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    
    best_acc = 0.0
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                predicted = torch.argmax(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {acc:.2f}%')
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with accuracy: {best_acc:.2f}%')
    
    print(f'Best Accuracy: {best_acc:.2f}%')
    

if __name__ == "__main__":
    train(model, train_loader, test_loader,
          criterion, optimizer, num_epochs)
