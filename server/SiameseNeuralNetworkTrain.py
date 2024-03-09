import torch
from torchvision import transforms
from PIL import Image
from Siamese512 import SiameseNetwork
from SiameseDataset import SiameseDataset
from torchvision.datasets import ImageFolder
from jobloss import ContrastiveLoss
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchsummary import summary
import random

torch.cuda.empty_cache()

# wrap in main due to multi-threading issue
def main():
    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and move it to the gpu
    model = SiameseNetwork().to(device)

    # Build the transform fo rthe dataset
    transform=transforms.Compose(
        [transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
 
    dataset = ImageFolder(root='../dataset')
    siamese_dataset = SiameseDataset(imageFolderDataset=dataset,
                        transform=transform)

    from torch.optim.lr_scheduler import StepLR

    data_loader = DataLoader(siamese_dataset, shuffle=True, batch_size=4, num_workers=4)
    loss_function = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)


    if device.type == 'cuda':
        summary(model, input_size=[(3, 128, 128), (3, 128, 128)])
    else:
        print("CUDA is not available. Model has been moved to CPU.")

    for epoch in range(0, 20):
        for i, data in enumerate(data_loader, 0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            optimizer.zero_grad()
            output1, output2 = model(img0, img1)
            loss = loss_function(output1, output2, label)
            loss.backward()
            optimizer.step()
        print("Epoch {}\n Current loss {}\n".format(epoch, loss.item()))
        scheduler.step()

    print("Training Finished")
    torch.save(model.state_dict(), 'checkpoints/test_siamese_network_weights.pth')





if __name__ == '__main__':
    main()









# a simple test to make surre it works
def testing():

    transform = transforms.Compose([
        transforms.Resize((128 , 128)),  # Resize the image to 512x512
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        # Normalize the image 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Preprocess the images using the defined transform
    image1 = transform(image1).unsqueeze(0)  # Add a batch dimension
    image2 = transform(image2).unsqueeze(0)  # Add a batch dimension

    # Assuming your model and images are on the same device (GPU or CPU)
    # Move images to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image1, image2 = image1.to(device), image2.to(device)
    model = model.to(device)

    # Pass the images through the model
    similarity = model(image1, image2)

    # Print the similarity
    print("Similarity:", similarity)