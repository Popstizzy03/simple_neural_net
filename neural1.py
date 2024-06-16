import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
 # 10 digits + 26 letters
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 36) 
 # Flatten the input
    def forward(self, x):
        x = x.view(-1, 28*28) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the network, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Initialize Pygame
pygame.init()

# Set screen dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Handwriting Recognition")

# Colors
white = (255, 255, 255)
black = (0, 0, 0)

# Font
font = pygame.font.Font(None, 36)
def predict_image(model, image):
    # Transform and normalize the image
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    image = transform(image).unsqueeze(0)
    
    # Predict using the neural network
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    
    return predicted.item()

def draw_text(text, x, y):
    text_surface = font.render(text, True, white)
    screen.blit(text_surface, (x, y))

def main():
    drawing = False
    last_pos = None
    image = pygame.Surface((28, 28))
    image.fill(white)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
                last_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                # Predict the drawn image
                prediction = predict_image(model, image)
                draw_text(f"Prediction: {chr(prediction + 48) if prediction < 10 else chr(prediction + 55)}", 10, 10)
            elif event.type == pygame.MOUSEMOTION and drawing:
                if last_pos is not None:
                    pygame.draw.line(screen, black, last_pos, event.pos, 5)
                    pygame.draw.line(image, black, last_pos, event.pos, 5)
                last_pos = event.pos
        
        pygame.display.flip()
        screen.fill(black)

    pygame.quit()

if __name__ == "__main__":
    main()
from torchvision import datasets

# Load datasets
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Training loop
for epoch in range(10):  # Number of epochs
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
