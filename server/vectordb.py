import faiss
import numpy as np
import torch
from Siamese512 import SiameseNetwork
import os
from torchvision import transforms
from PIL import Image
from random import randint 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
model = SiameseNetwork().to('cuda')
# Load the weights
model.load_state_dict(torch.load('checkpoints/test_siamese_network_weights.pth'))

model.eval()  # Set the model to evaluation mode

def extract_feature_vector(model, image):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        print('forward once')
        feature_vector = model.forward_once(image.to('cuda'))
        print('got vector')
    return feature_vector.cpu()



preprocessed_images = []

# load the images path list 
class_directories = os.listdir('../dataset/')  # the classes.

for directory in class_directories:
    for filename in os.listdir('../dataset/'+directory):
        preprocessed_images.append('../dataset/'+directory+'/'+filename) 


#print(preprocessed_images[0:1000])

#preprocessed_images =preprocessed_images[1:100]

# Dimension of the vectors stored in FAISS
dim = 128  

print('create index')
cpu_index = faiss.IndexFlatL2(dim)  # Create a flat (brute-force) index
print('done creating index')

print('build transform')
# Build the transform
transform=transforms.Compose(
    [transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Assuming you have a list of images and a model
for image_name in preprocessed_images:
    print('processing '+  image_name)
    image = Image.open(image_name).convert("RGB")
    image = transform(image).unsqueeze(0).to(device) 
    print('about to extract feature '+  image_name)
    vector = extract_feature_vector(model, image)  
    # Convert tensor to numpy array and reshape to (1, -1) as FAISS expects
    vector_np = vector.cpu().numpy().reshape(1, -1)
    # Add the vector to the index
    cpu_index.add(vector_np)

# Save the FAISS index
faiss.write_index(cpu_index, "siamese_network_vectors.index")
# Load the index
index = faiss.read_index("siamese_network_vectors.index")


print('------------------------------ index built --------------------------------------')

# Path to a test image
test_image_path = preprocessed_images[randint(0,len(preprocessed_images))]  # Update this to your test image path
print('testing for ' + test_image_path)
# Load and preprocess the test image
test_image = Image.open(test_image_path).convert("RGB")
test_image = transform(test_image).unsqueeze(0).to(device)  # Apply the same transformations

test_vector = extract_feature_vector(model, test_image)
test_vector_np = test_vector.cpu().numpy().reshape(1, -1)  # Prepare it for FAISS

k = 3  # Closest 3 to return
D, I = index.search(test_vector_np, k)  # D is the distances, I is the indices of the closest images

print("Closest matches:")

for i, idx in enumerate(I[0]):
    closest_image_name = preprocessed_images[idx]
    print(f"{i+1}: {closest_image_name} with distance {D[0][i]}")

torch.cuda.empty_cache()