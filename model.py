import cv2
import skimage
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


class UBCOCEANNet1(nn.Module):
    def __init__(self, label_count):
        '''Define the neural network architecture. This is where you can experiment with different architectures.
        params:
            label_count (int): The number of unique labels to predict
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP layer
        self.fc = nn.Linear(64, label_count) # Fully connected layer
        # self.output = nn.LogSoftmax(dim=1)  # Log softmax compatible with negative loss likelihood optimizer for best numerical stability

    # x represents our data, pass it through the entire neural network
    # Here, we define actions such as the activation function and max pooling, defining how data passes through the neural network
    # This forward pass is implicitly called when we call model(x)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.global_avg_pool(x)  # Apply GAP layer
        x = x.view(-1, 64)  # Flatten for fully connected layer
        x = self.fc(x)

        # Apply softmax to x to get output distribution amongst classes
        # Log softmax compatible with negative loss likelihood optimizer for best numerical stability
        # outputs = F.log_softmax(x, dim=1)
        # outputs = self.output(x)

        # In this case, x represents the logits (raw output) of the network
        # The CrossEntropyLoss criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
        return x
    

class JPEGCompressTransform():
    def __init__(self, jpeg_quality=80):
        self.jpeg_quality = jpeg_quality

    def __call__(self, sample):
        # Encode the image as a JPEG with specified quality
        _, jpeg_image = cv2.imencode('.jpg', sample, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])

        # `jpeg_image` is a numpy array containing the compressed JPEG image
        # Convert to bytes if you want to use the binary data
        jpeg_bytes = jpeg_image.tobytes()
        return jpeg_bytes
    

class RescaleWithAspectRatioTransform():
    def __init__(self, longest_edge):
        self.longest_edge = longest_edge

    def __call__(self, sample):
        return self.resize_with_aspect_ratio(sample, self.longest_edge)

    def resize_with_aspect_ratio(self, image, longest_edge):
        """
        Resize image while preserving its aspect ratio

        Parameters
        ----------
        image: numpy.ndarray of shape (height, width, 3)
            Image array

        longest_edge: int
            Desired number of pixels on the longest edge

        Returns
        -------
        image: numpy.ndarray of shape (resized_height, resized_width, 3)
            Resized image array
        """

        height, width = image.shape[:2]
        scale = longest_edge / max(height, width)
        image = cv2.resize(image, dsize=(int(np.ceil(width * scale)), int(np.ceil(height * scale))), interpolation=cv2.INTER_AREA)
        return image

class UCBOCEANTestDataset(Dataset):
    def __init__(self, data, data_dir, transform=None, target_transform=None):
        # Don't open images that allocate more than 20GB RAM
        import PIL
        PIL.Image.MAX_IMAGE_PIXELS = 20971520000
        self.data = data
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load the data at the specified index
        x = self.data.iloc[index]

        # Preprocess the data as needed
        x, label = self.preprocess(x)

        # Return the preprocessed data and its label (if applicable)
        return x, label
    
    def preprocess(self, x):
        # Apply any preprocessing transformations to the data row
        label = x['label']
        
        # Load the data from the zip file
        image_id = x['image_id']
        image = self.load_test_image(image_id)

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def load_test_image(self, image_id):
        image_file_name = f'{image_id}.png'
        image_file_path = self.data_dir / image_file_name
        return skimage.io.imread(image_file_path)