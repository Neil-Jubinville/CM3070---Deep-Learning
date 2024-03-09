import faiss
import numpy as np
import torch
from Siamese512 import SiameseNetwork
import os
from torchvision import transforms
from PIL import Image
from random import randint 
from time import sleep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
model = SiameseNetwork().to('cuda')
# Load the weights
model.load_state_dict(torch.load('checkpoints/test_siamese_network_weights.pth'))

model.eval()  # Set the model to evaluation mode

def extract_feature_vector(model, image):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  
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




# Build the transform
transform=transforms.Compose(
    [transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# Load the index
index = faiss.read_index("siamese_network_vectors.index")


print('------------------------------ index built --------------------------------------')

def random_search():

    output = '------------------------------ Search --------------------------------------\n'

    print('------------------------------ Search --------------------------------------')
    
    # Path to a test image
    test_image_path = preprocessed_images[randint(0,len(preprocessed_images))]  # Update this to your test image path
    print('TEST:  ' + test_image_path)
    output +='TEST:  ' + test_image_path +'\n'
    # Load and preprocess the test image
    test_image = Image.open(test_image_path).convert("RGB")
    test_image = transform(test_image).unsqueeze(0).to(device)  # Apply the same transformations

    test_vector = extract_feature_vector(model, test_image)
    test_vector_np = test_vector.cpu().numpy().reshape(1, -1)  # Prepare it for FAISS

    k = 5  # Closest 3 to return
    D, I = index.search(test_vector_np, k)  # D is the distances, I is the indices of the closest images

    print("Closest matches:")
    output +='Closest matches: ' + '\n'

    for i, idx in enumerate(I[0]):
        closest_image_name = preprocessed_images[idx]
        print(f"{i+1}: {closest_image_name} with distance {D[0][i]}")
        output +=f"{i+1}: {closest_image_name} with distance {D[0][i]}"+ '\n'

    torch.cuda.empty_cache()
    return output


with open('similarity_results.txt','w') as  logfile: 
    while True:
        output = random_search()
        logfile.write(output)
        sleep(1)
