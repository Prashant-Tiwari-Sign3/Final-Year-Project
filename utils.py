import os
import torch
import numpy as np
import pandas as pd
import librosa as lr
from PIL import Image
from tqdm import tqdm
import noisereduce as nr
from tqdm.auto import tqdm
import moviepy.editor as mp
import librosa.display as ld
from sklearn.metrics import *
from sklearn.ensemble import *
import matplotlib.pyplot as plt
from PIL.ImageOps import invert
from timeit import default_timer as timer
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from nltk.corpus import stopwords, wordnet
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split

def build_spectogram(audio_path, plot_path, bar = False):
    """
    Build spectrograms from audio files and save them as PNG images.

    Parameters:
        audio_path (str): The path to the directory containing the audio files.
        plot_path (str): The path to the directory where the spectrogram images will be saved.
        bar (bool, optional): Whether to include a colorbar in the spectrogram images. Default is False.

    Returns:
        None

    Raises:
        OSError: If there is an error while creating directories or loading audio files.

    """
    folders = []
    for item in os.listdir(audio_path):
        item_path = os.path.join(audio_path, item)
        if os.path.isdir(item_path):
            folders.append(item)

    for folder in folders:
        item_list = os.listdir(audio_path + folder)
        os.makedirs(plot_path+'/'+folder)
        for item in item_list:
            music, rate = lr.load(audio_path+folder+'/'+item)
            stft = lr.feature.melspectrogram(y=music, sr=rate, n_mels=256)
            db = lr.amplitude_to_db(stft)
            fig, ax = plt.subplots()
            img = ld.specshow(db, x_axis='time', y_axis='log', ax=ax)
            plt.axis(False)
            if bar == True:
                fig.colorbar(img, ax=ax, format='%0.2f')
            a = item.replace('.wav', '.png')
            plt.savefig(plot_path+'/'+folder+'/'+a)

def performance(model, x_test, y_test):
    """
    Calculates and displays the performance metrics of a trained model.

    Parameters:
    -----------
    model : object
        The trained machine learning model.

    x_test : array-like of shape (n_samples, n_features)
        The input test data.

    y_test : array-like of shape (n_samples,)
        The target test data.

    Returns:
    --------
    None

    Prints:
    -------
    Model Performance:
        Classification report containing precision, recall, F1-score, and support for each class.
    Accuracy:
        The accuracy of the model on the test data.
    Confusion Matrix:
        A plot of the confusion matrix, showing the true and predicted labels for the test data.

    Example:
    --------
    >>> performance(model, x_test, y_test)
    """

    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    print("                 Model Performance")
    print(report)
    print(f"Accuracy = {round(accuracy*100, 2)}%")
    matrix = confusion_matrix(y_test, preds)
    matrix_disp = ConfusionMatrixDisplay(matrix)
    matrix_disp.plot(cmap='Reds')
    plt.show()
    
class CustomDataset_CSVlabels(Dataset):
    """
    A PyTorch dataset for loading spectrogram images and their corresponding labels from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing the image file names and labels.
        img_dir (str): Root directory where the image files are stored.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. 
            E.g, ``transforms.RandomCrop`` for randomly cropping an image.

    Attributes:
        img_labels (DataFrame): A pandas dataframe containing the image file names and labels.
        img_dir (str): Root directory where the image files are stored.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. 
            E.g, ``transforms.RandomCrop`` for randomly cropping an image.
    
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the image and label at the given index.

    Returns:
        A PyTorch dataset object that can be passed to a DataLoader for batch processing.
    """
    def __init__(self,csv_file, img_dir, transform=None) -> None:
        super().__init__()
        self.img_labels = pd.read_csv(csv_file)
        self.img_labels.drop(['Unnamed: 0'], axis=1, inplace=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.img_labels)
    
    def __getitem__(self, index):
        """
        Returns the image and label at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and label.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index,0])
        image = Image.open(img_path)
        image = image.convert("RGB")
        y_label = torch.tensor(int(self.img_labels.iloc[index,1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

def Train_Loop(
        num_epochs:int,
        train_dataloader:torch.utils.data.DataLoader,
        test_dataloader:torch.utils.data.DataLoader,
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        loss_function:torch.nn.Module,
        device:str
):
    """
    Trains a PyTorch model using the given train and test dataloaders for the specified number of epochs.

    Parameters:
    -----------
    num_epochs : int
        The number of epochs to train the model for.
    train_dataloader : torch.utils.data.DataLoader
        The dataloader for the training data.
    test_dataloader : torch.utils.data.DataLoader
        The dataloader for the test/validation data.
    model : torch.nn.Module
        The PyTorch model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer to be used during training.
    loss_function : torch.nn.Module
        The loss function to be used during training.

    Returns:
    --------
    None

    Raises:
    -------
    None

    Notes:
    ------
    This function loops over the specified number of epochs and for each epoch, it trains the model on the training
    data and evaluates the performance on the test/validation data. During each epoch, it prints the training loss
    and the test loss and accuracy. At the end of training, it prints the total time taken for training.
    """
    model.to(device)
    start_time = timer()
    
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch: {epoch}\n-----------")
        train_loss = 0
        for batch, (x,y) in enumerate(train_dataloader):
            x,y = x.to(device), y.to(device)
            y=y.float().squeeze()
            model.train()
            y_logits = model(x).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits))
            loss = loss_function(y_logits, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if batch % 10 == 0:
            #     print(f"Looked at {batch * len(x)}/{len(train_dataloader.dataset)} samples")

        train_loss /= len(train_dataloader)
        
        test_loss, test_acc = 0, 0 
        test_log_loss = 0
        model.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                X,y = X.to(device), y.to(device)
                y = y.float().squeeze()
                test_logits = model(X).squeeze()
                test_pred = torch.round(torch.sigmoid(test_logits))
                test_loss += loss_function(test_logits, y)
                test_acc += accuracy_score(y_true=y, y_pred=test_pred)
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc*100:.2f}%\n")

    end_time = timer()
    print(f"Time taken = {end_time-start_time}")

class CustomDataset_FolderLabels:
    """
    CustomDataset class for loading and splitting a dataset into training, validation, and testing sets.

    Args:
        data_path (str): Path to the main folder containing subfolders for each class.
        train_ratio (float): Ratio of data allocated for the training set (0.0 to 1.0).
        val_ratio (float): Ratio of data allocated for the validation set (0.0 to 1.0).
        test_ratio (float): Ratio of data allocated for the testing set (0.0 to 1.0).
        batch_size (int): Number of samples per batch in the data loaders.
        transform (torchvision.transforms.transforms.Compose): Transformations to be applied on the image

    Attributes:
        train_loader (torch.utils.data.DataLoader): Data loader for the training set.
        val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
        test_loader (torch.utils.data.DataLoader): Data loader for the testing set.

    """
    def __init__(self, data_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=32, transform=None):
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        if transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        self._load_dataset()

    def _load_dataset(self):
        """
        Loads the dataset and splits it into training, validation, and testing sets.

        """
        dataset = ImageFolder(root=self.data_path, transform=self.transform)
        num_samples = len(dataset)

        train_size = int(self.train_ratio * num_samples)
        val_size = int(self.val_ratio * num_samples)
        test_size = num_samples - train_size - val_size

        self.train_set, self.val_set, self.test_set = random_split(dataset, [train_size, val_size, test_size])

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)

    def get_train_loader(self):
        """
        Get the data loader for the training set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the training set.

        """
        return self.train_loader

    def get_val_loader(self):
        """
        Get the data loader for the validation set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the validation set.

        """
        return self.val_loader

    def get_test_loader(self):
        """
        Get the data loader for the testing set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the testing set.

        """
        return self.test_loader

    
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin: int = 1) -> None:
        """
        Contrastive Loss function for training a neural network with pairwise distance-based contrastive loss.

        Args:
            margin (int, optional): The margin for the contrastive loss. Default is 1.

        Note:
            The contrastive loss function aims to minimize the distance between embeddings of similar pairs
            and maximize the distance between embeddings of dissimilar pairs.

            The formula for the contrastive loss is:
            loss = y * (dist^2) + (1 - y) * max(margin - dist, 0)^2

            where:
            - dist: The Euclidean distance between two embeddings.
            - y: The binary label (1 for similar pairs, 0 for dissimilar pairs).
            - margin: The margin for the contrastive loss.

        Example:
            loss = ContrastiveLoss(margin=1)
            embedding1 = torch.tensor([1.0, 2.0])
            embedding2 = torch.tensor([3.0, 4.0])
            similarity_label = 1  # Similar pair
            output = loss(torch.norm(embedding1 - embedding2), similarity_label)
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, y):
        """
        Forward pass of the contrastive loss function.

        Args:
            dist (torch.Tensor): The Euclidean distance between two embeddings.
            y (torch.Tensor): The binary label indicating the similarity between the embeddings (1 for similar pairs, 0 for dissimilar pairs).

        Returns:
            torch.Tensor: The contrastive loss value.

        Note:
            This function computes the contrastive loss for a pair of embeddings based on their distance and similarity
            label. The loss aims to pull similar pairs closer together and push dissimilar pairs apart in the embedding space.
        """
        loss = y * torch.pow(dist, 2) + (1 - y) * torch.pow(torch.clamp(self.margin - dist, min=0), 2)
        return loss.mean()
    
class SiameseDataset(Dataset):
    """
    Custom PyTorch Dataset for Siamese neural network training on image pairs.

    This dataset loads pairs of grayscale images and their corresponding labels from a CSV file. 
    The CSV file should contain image file names of two files in each row along with a label 
    (1 if the images are similar, 0 if they are not). The images are loaded from the specified 
    'image_dir' directory.

    Parameters:
        labels_csv (str): Path to the CSV file containing image pairs and their labels.
        image_dir (str): Path to the directory containing the images.
        transforms (transforms.Compose, optional): A composition of PyTorch transforms to be applied 
            to the images. Default is None.

    Returns:
        tuple: A tuple containing two images (image1 and image2) and their corresponding label.
            - image1 (torch.Tensor): The first grayscale image, converted to a PyTorch tensor.
            - image2 (torch.Tensor): The second grayscale image, converted to a PyTorch tensor.
            - label (torch.Tensor): The label indicating whether the images are similar (1) or not (0).
    
    Note:
        - The CSV file should have three columns: 'image1', 'image2', and 'label'.
        - The 'transforms' argument is an optional parameter that allows applying transformations 
          to the images. The transformations should be a composition of transforms from the 
          'torchvision.transforms' module. If not provided, the images will be returned as PIL Images.
        - The images are loaded as grayscale ('L') images.

    Example:
        # Assuming you have a CSV file 'data.csv' and a folder 'images' containing the images
        # Initialize the dataset with default transforms
        data = SiameseDataset("data.csv", "images")

        # Or, initialize the dataset with custom transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        data = SiameseDataset("data.csv", "images", transform=transform)
    """
    def __init__(self, labels_csv:str, image_dir:str, transforms:transforms=None) -> None:
        super().__init__()
        self.labels = pd.read_csv(labels_csv, index_col=False)
        self.image_dir = image_dir
        self.transform = transforms

    def __getitem__(self, index):
        image1_path = os.path.join(self.image_dir, str(self.labels.iat[index, 1]))
        image2_path = os.path.join(self.image_dir, str(self.labels.iat[index, 2]))
        label = torch.tensor(self.labels.iat[index,3])
        
        image1 = Image.open(image1_path).convert("L")
        image2 = Image.open(image2_path).convert("L")

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)        

        return image1, image2, label
    
    def __len__(self):
        return len(self.labels)
    
def Siamese_TrainLoop(
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        loss_function:torch.nn.Module,
        num_epochs:int,
        scheduler:torch.optim.lr_scheduler.StepLR,
        train_dataloader:torch.utils.data.DataLoader,
        test_dataloader:torch.utils.data.DataLoader,
        early_stopping_rounds:int,
        val_dataloader:torch.utils.data.DataLoader=None,
        device:str='cpu'
):
    """
    TrainLoop function for training a PyTorch neural network model using the specified data and settings.

    Parameters:
        model (torch.nn.Module): The PyTorch neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters during training.
        loss_function (torch.nn.Module): The loss function used to compute the training and validation loss.
        num_epochs (int): The number of epochs to train the model for.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler for the optimizer.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset (used for early stopping).
        early_stopping_rounds (int): Number of epochs to wait before stopping training if there is no improvement in validation loss.
        val_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the validation dataset. Default is None.
        device (str, optional): Device to use for training (e.g., 'cpu' or 'cuda'). Default is 'cpu'.

    Returns:
        None: This function does not return any value. It trains the model and prints the progress.

    Note:
        - The model, optimizer, loss function, and scheduler should be properly initialized before calling this function.
        - The train and validation dataloaders should provide batches of data in the format (x1, x2, y), where x1 and x2 are input tensors and y is the target tensor.
        - The test dataloader is used for early stopping based on validation loss. If early stopping is not required, set `test_dataloader` to None.
        - The model will be moved to the specified `device` before training.
        - This function uses tqdm for displaying the training progress.
    """
    model.to(device)
    start_time = timer()
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_loss = 0
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch : {epoch}\n----------------------")
        for batch, (x1, x2, y) in enumerate(train_dataloader):
            x1.to(device)
            x2.to(device)
            y.to(device)
            optimizer.zero_grad()
            distance = model(x1, x2)
            loss = loss_function(distance, y)
            loss.backward()
            optimizer.step()
            print(f"Loss for batch: {batch} = {loss}")
            train_loss += loss

        print(f"Training Loss = {train_loss}")

        if val_dataloader is not None:
            model.eval()
            validation_loss = 0
            with torch.inference_mode():
                for x1, x2, y in val_dataloader:
                    distance = model(x1,x2)
                    loss = loss_function(distance, y)
                    validation_loss+=loss

                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement+=1

                print(f"Current Validation Loss = {validation_loss}")
                print(f"Best Validation Loss = {best_val_loss}")
                print(f"Epochs without Improvement = {epochs_without_improvement}")

                if epochs_without_improvement > early_stopping_rounds:
                    print("Early Stoppping Triggered")
                    break

        scheduler.step()

    end_time = timer()
    print(f"Training Time = {end_time-start_time} seconds")