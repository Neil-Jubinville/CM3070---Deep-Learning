import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),  # Adjusted for 3 input channels
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 29 * 29, 512),  # Adjusted for the output size of the convolutional layers
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)  # Flatten the output
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return (output1,output2)
    
        # similarity = F.pairwise_distance(output1, output2)
        # return similarity

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and move it to the selected device
model = SiameseNetwork().to(device)

import torch
print(torch.cuda.is_available())


# Assuming your input size is (3, 512, 512) for an RGB image of 512x512 pixels
# Note: torchsummary summary might need to be called differently if running outside a notebook
# if device.type == 'cuda':
#     summary(model, input_size=[(3, 512, 512), (3, 512, 512)])
# else:
#     print("CUDA is not available. Model has been moved to CPU.")