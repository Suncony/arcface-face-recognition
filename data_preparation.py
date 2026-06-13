import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torchvision import models, transforms, datasets
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid
    
def prepare_data():
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.RandomCrop(224),     # Random crop for better generalization
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)  # Randomly erase parts of the image
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the LFW dataset with different transformations for training and validation sets
    lfw_dataset_train = datasets.ImageFolder(root="data", transform=train_transforms)
    lfw_dataset_val = datasets.ImageFolder(root="data", transform=val_transforms)

    # Extract labels from the dataset
    labels = [label for _, label in lfw_dataset_train]

    # Organize data by person ID
    person_to_indices = {}
    for idx, label in enumerate(labels):
        if label not in person_to_indices:
            person_to_indices[label] = []
        person_to_indices[label].append(idx)

    # Split each person's images into train/val sets
    train_indices = []
    val_indices = []
    for person, indices in person_to_indices.items():
        if len(indices) > 1:
            # If a person has more than one image, split them between training and validation
            train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
        else:
            # If a person has only one image, put it in the training set
            train_indices.extend(indices)

    # Create Subsets for training and validation
    train_dataset = torch.utils.data.Subset(lfw_dataset_train, train_indices)
    val_dataset = torch.utils.data.Subset(lfw_dataset_val, val_indices)

    # Create DataLoader objects
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=2,  
        pin_memory=True
    )
    
    # Show the sample data with improved formatting
    show_sample_data(train_loader, lfw_dataset_train)

    return train_loader, val_loader, len(lfw_dataset_train.class_to_idx)

def show_sample_data(loader, dataset, num_images=8, rows=2):

    data_iter = iter(loader)
    images, labels = next(data_iter)
    
    # Calculate columns based on number of images and rows
    cols = num_images // rows
    
    # Create image grid with increased padding
    img_grid = make_grid(images[:num_images], nrow=cols, normalize=True, padding=20)
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create the main axis for images
    ax = plt.gca()
    ax.imshow(np.transpose(img_grid.numpy(), (1, 2, 0)))
    ax.axis('off')
    
    # Create reverse mapping of idx to class name
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    
    # Calculate grid dimensions
    grid_width = img_grid.size(2) / cols
    grid_height = img_grid.size(1) / rows
    
    # Style configurations
    label_font_size = 10
    background_color = '#2C3E50'  # Dark blue background for labels
    text_color = 'white'
    label_height = 25
    
    for i, label in enumerate(labels[:num_images]):
        person_name = idx_to_class[label.item()].replace('_', ' ') 
        
        # Calculate positions
        col = i % cols
        row = i // cols
        
        # Calculate center positions for the label
        x_center = col * grid_width + grid_width/2
        y_position = (row + 1) * grid_height - label_height/2
        
        # Add background rectangle for the label
        rect = Rectangle((col * grid_width + 10, y_position - label_height/2),
                        grid_width - 20, label_height,
                        facecolor=background_color,
                        alpha=0.9,
                        edgecolor='none',
                        zorder=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x_center, y_position,
                person_name,
                color=text_color,
                fontsize=label_font_size,
                fontweight='bold',
                ha='center',
                va='center',
                zorder=3)
    
    # Add title with proper styling
    plt.title("Sample Training Data with Names",
              pad=20,
              fontsize=16,
              fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    plt.close()