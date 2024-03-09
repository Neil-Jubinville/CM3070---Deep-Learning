
import torch
from torchvision import transforms
from PIL import Image
from Siamese512 import SiameseNetwork
from SiameseDataset import SiameseDataset
from torchvision.datasets import ImageFolder
from jobloss import ContrastiveLoss
from torchvision import transforms
from torch.utils.data import DataLoader
import random
# Instantiate the model
model = SiameseNetwork()

# Load the weights
model.load_state_dict(torch.load('checkpoints/test_siamese_network_weights.pth'))

# Switch to evaluation mode
model.eval()

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the images
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and transform the images
image1 = Image.open("../dataset/0-50/Warehouse Worker.png").convert("RGB")
image2 = Image.open("../dataset/23-40/First Officer.png").convert("RGB")

image1 = transform(image1).unsqueeze(0).to(device)  # Add batch dimension
image2 = transform(image2).unsqueeze(0).to(device)

with torch.no_grad():  
    output1, output2 = model(image1, image2)
    similarity = torch.pairwise_distance(output1, output2)
    print("Similarity:", similarity.item())

    euclidean_distance = torch.norm(output1 - output2).item()

    # Additionally, compute Cosine similarity
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_similarity = cos(output1, output2).item()

    print(f"Euclidean Distance: {euclidean_distance}")
    print(f"Cosine Similarity: {cosine_similarity}")

    
