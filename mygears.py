#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 17:25:41 2025

@author: kasikritdamkliang
"""
import tensorflow as tf
from tensorflow import keras
import os, platform
import time
import cv2
from datetime import datetime
from tqdm import tqdm
import glob
# import itertools
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from PIL import Image 
import seaborn as sns 
from collections import defaultdict
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import random
import tensorflow_addons as tfa

class CustomTQDMProgressBar(tfa.callbacks.TQDMProgressBar):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Safely extract the learning rate as a scalar for logging
        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        
        # Add the learning rate to the logs for display
        logs['learning_rate'] = lr
        super().on_epoch_end(epoch, logs)

# def cfg_to_dict(cfg_node):
#     """
#     Recursively convert a YACS CfgNode to a nested dictionary.
#     """
#     if isinstance(cfg_node, CN):
#         return {k: v for k, v in cfg_node.items()}
#     return cfg_node

def cfg_to_dict(cfg_node):
    """
    Recursively convert a YACS CfgNode to a nested dictionary.
    """
    # if isinstance(cfg_node, config):
    #     return {k: v for k, v in cfg_node.items()}
    return {k: v for k, v in cfg_node.items()}

def filter_hidden_files(file_paths):
    """
    Filter out paths that start with a '.' indicating they are hidden.
    This only considers the filename, not part of the path, as hidden.
    """
    return [path for path in file_paths if not os.path.basename(path).startswith('.')]

def make_pair(imgs, labels):
    pairs = []
    for img, mask in zip(imgs, labels):
        pairs.append( (img, mask) )
    
    return pairs

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def convertBinary(image, debug=False):
    h, w, _ = image.shape
    # Convert the RGB image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply a binary threshold
    _, binary_image = cv2.threshold(gray_image, 160, 255,
                    cv2.THRESH_BINARY)
    
    pixels = cv2.countNonZero(binary_image)
    pixel_ratio = (pixels/(h * w)) * 100
    title = 'Pixel ratio: {:.2f}%'.format(pixel_ratio)
    if debug==True:
        plt.figure()
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(binary_image, cmap='gray')
        plt.title(title)
        print(title)
    
    # Convert back to 3 channels (to ensure it has RGB channels)
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    
    return binary_image
        

def preprocess_for_alexnet(image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image_resized = cv2.resize(image_bgr, (227, 227))
    image_resized = image_bgr.astype(np.float32)

    # 4. Subtract ImageNet mean values for each channel (RGB)
    mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    image_normalized = image_resized - mean

    return image_normalized

def create_balanced_dataframe(patchPaths_train, class_list_train):
    """
    Creates a balanced DataFrame by oversampling the minority classes.

    Args:
    patchPaths_train (list): List of file paths to training images.
    class_list_train (list): List of class labels corresponding to the training images.

    Returns:
    pd.DataFrame: A DataFrame containing balanced image paths and their respective class labels.
    """
    
    # Initialize RandomOverSampler
    ros = RandomOverSampler()

    # Convert class list to NumPy array for compatibility with RandomOverSampler
    y = np.array(class_list_train)
    X = np.array(patchPaths_train).reshape(-1, 1)  # Reshape to 2D array for compatibility

    # Print class distribution before resampling
    print("Class distribution before resampling:", Counter(y))

    # Perform the oversampling
    X_Ros, y_Ros = ros.fit_resample(X, y)

    # Print class distribution after resampling
    print("Class distribution after resampling:", Counter(y_Ros))

    # Create a balanced DataFrame
    balanced_df = pd.DataFrame({
        'path': X_Ros.flatten(),  # Flatten to 1D array
        'class': y_Ros            # Class labels
    })
    
    return balanced_df

def get_png_files_and_classes(dataset_root,
    patient_set,
    aug=False):
    """
    This function takes the root directory of the dataset and a set of patient data (test_set or train_set),
    and returns three items:
    1. png_list: A list of all PNG file paths.
    2. class_list: A list of corresponding class IDs for each PNG file.
    3. class_counts: A dictionary with the total number of PNG files for each class.
    
    Parameters:
    - dataset_root (str): The root directory of the dataset.
    - patient_set (list of tuples): A list where each entry is a tuple (patient_id, class_id).
    
    Returns:
    - png_list (list of str): List of all PNG file paths.
    - class_list (list of str): List of corresponding class IDs for each PNG file.
    - class_counts (dict): A dictionary with the total number of PNG files for each class.
    """
    png_list = []    # To store the paths of all PNG files
    class_list = []  # To store the corresponding class IDs
    class_counts = defaultdict(int)  # To store the count of PNG files for each class
    
    # Iterate through the patient_set to read *.png files
    for (patient_path, class_id) in patient_set:
        print(patient_path, class_id)
        # break
        # Construct the patch path
        if aug:
            patch_path = os.path.join(dataset_root,
                # class_id,
                patient_path,
                'cells',
                '*',
                '*.png')
            print(f"{patch_path=}")    
            png_files = glob.glob(patch_path)

            
            patch_path_aug = os.path.join(dataset_root,
                    # class_id,
                    patient_path,
                    'aug',
                    '*.png')
            print(f"{patch_path_aug=}")  
            # Use glob to find all png files in the constructed path
            png_files_cell = glob.glob(patch_path)
            png_files_aug = glob.glob(patch_path_aug)
            png_files = png_files_cell + png_files_aug
        else:
            patch_path = os.path.join(dataset_root,
                #class_id,
                patient_path,
                'cells',
                '*',
                '*.png')  
            print(f"{patch_path=}")    
            png_files = glob.glob(patch_path)
        
        # patient_id = patient_path.split(os.sep)[2]
        print(f"Patient {patient_path} (Class {class_id}): Found {len(png_files)} PNG files.")
        
        # Append png_files to png_list and class_list
        png_list.extend(png_files)        # Add all PNG file paths to png_list
        class_list.extend([class_id] * len(png_files))  # Add the class_id for each PNG file
        
        # Update the count of PNG files for the current class_id
        class_counts[class_id] += len(png_files)
    
    # Return both lists and the class counts
    return png_list, class_list, class_counts

#%%
def count_plot(df, title, save_dir, save=False, ):
    fig = plt.figure(dpi=300)   
    # Set the desired class order: first '0' then '1'
    class_order = ['0', '1']
    
    # Create a countplot, explicitly setting the order of classes
    palette = sns.color_palette("Set2", 3)
    ax = sns.countplot(x=df['class'],
                       palette=palette,
                       order=class_order)  # Specify the order of classes
    # Add class count labels on each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    # Add labels and title
    plt.xlabel("Class")
    plt.title(title)   
    if save==True:
        plot_file = os.path.join(save_dir, f"{title}.png")
        fig.savefig(plot_file)  
        print(f"Saved {plot_file}")
    
    plt.show()

def plot_images_with_labels(images, 
    labels,
    train_log,
    set_label,
    # save=False,
    config,
    ):
    """
    Plots a grid of images with their corresponding labels and saves the figure.
    
    Args:
    - images (list or np.array): List of images to be displayed.
    - labels (np.array): Array of one-hot encoded labels.
    - train_log (str): The directory or log name to save the plot.
    - set_label (str): The specific set label (e.g., "train", "validation", etc.).
    - config (dict): Configuration dictionary containing image dimensions.
    
    Returns:
    - None: Displays and saves the plot.
    """
    # Convert one-hot encoded labels to class indices
    class_indices = np.argmax(labels, axis=1)

    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), dpi=120)
    axes = axes.flatten()

    for img, label_idx, ax in zip(images, class_indices, axes):
        # Undo any preprocessing (like rescaling for pre-trained models)
        if np.max(img) == 1:
            img = img * 255

        # Display the image
        img_shape = (config['DATA']['W'], config['DATA']['H'], config['DATA']['C'])
        ax.imshow(img.reshape(img_shape).astype("uint8"))

        # Set the title to the corresponding class label
        ax.set_title(f"Label: {label_idx}", fontsize=18)
        ax.axis('on')

    # Set the overall title for the figure
    plot_title = f"{train_log}-{set_label}"
    plt.suptitle(plot_title, fontsize=18)

    # Show the plot
    plt.show()

    # Save the figure
    if config['SAVE']:
        plot_file = os.path.join(train_log, f"{plot_title}.png")
        fig.savefig(plot_file, dpi=120)
        print(f'Saved {plot_file}')

def plot_images_by_class(df, class_label, savefile):
    """
    Plots a grid of images for a specific class (IDA or Thalassemia).

    Args:
    - df (DataFrame): DataFrame containing image paths and labels.
    - class_label (int): Class to filter (0 = IDA, 1 = Thalassemia).
    - num_images (int): Number of images to display.

    Returns:
    - None: Displays the images in a grid.
    """
    num_images=12
    # Filter dataframe based on the selected class
    class_df = df[df['class'] == class_label]

    # Check if we have enough images
    num_images = min(num_images, len(class_df))
    if num_images == 0:
        print(f"No images found for class {class_label}")
        return

    # Sample images randomly
    # sample_df = class_df.sample(n=num_images, random_state=42).reset_index(drop=True)
    sample_df = class_df.sample(n=num_images).reset_index(drop=True)

    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), dpi=120)
    axes = axes.flatten()

    for i in range(num_images):
        img_path = sample_df.iloc[i]['path']  # Use .iloc[i] instead of .loc[i]
        label = sample_df.iloc[i]['class']

        # Load the image using OpenCV (convert to RGB)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the image
        axes[i].imshow(img)
        # axes[i].set_title(f"Label: {'IDA' if label == '0' else 'Thalassemia'}", fontsize=12)
        axes[i].axis("off")

    # Hide unused subplots if there are fewer than 12 images
    for j in range(num_images, len(axes)):
        axes[j].axis("off")

    # Set the overall title for the figure
    # plt.suptitle(f"Sample Images for {'IDA' if class_label == '0' else 'Thalassemia'}", fontsize=18)
    plt.show()
    fig.savefig(savefile)

#%%
def model_mapping_function(model_name, config):
    model_root = config['DATA']['SAVEPATH']  # Assuming config contains the save path

    # Select appropriate model paths based on color mode
    if config['DATA']['COLOR'] == 'RGB':
        model_mapping = {
            'EfficientNetV2S': [
                os.path.join(model_root, 'Linux-Fold-0-EfficientNetV2S-rgb-bal-aug-20250302-2215.hdf5'),
                os.path.join(model_root, 'Linux-Fold-1-EfficientNetV2S-rgb-bal-aug-20250303-1001.hdf5'),
                os.path.join(model_root, 'Linux-Fold-2-EfficientNetV2S-rgb-bal-aug-20250303-1001.hdf5'),
                os.path.join(model_root, 'Linux-Fold-3-EfficientNetV2S-rgb-bal-aug-20250303-1001.hdf5'),
                os.path.join(model_root, 'Linux-Fold-4-EfficientNetV2S-rgb-bal-aug-20250303-1001.hdf5'),
            ]
        }
    else:  # Assuming BGR color mode
        model_mapping = {
            'EfficientNetV2S': [
                os.path.join(model_root, 'Linux-Fold-0-EfficientNetV2S-bal-aug-20250110-1607.hdf5'),
                os.path.join(model_root, 'Linux-Fold-1-EfficientNetV2S-bal-aug-20250110-1607.hdf5'),
                os.path.join(model_root, 'Linux-Fold-2-EfficientNetV2S-bal-aug-20250110-1607.hdf5'),
                os.path.join(model_root, 'Linux-Fold-3-EfficientNetV2S-bal-aug-20250109-1903.hdf5'),
                os.path.join(model_root, 'Linux-Fold-4-EfficientNetV2S-bal-aug-20250110-1607.hdf5'),
            ]
        }

    # Check if the requested model exists in the mapping
    if model_name in model_mapping:
        return model_mapping[model_name]
    else:
        raise ValueError(f"No model mapping found for model {model_name}")

from tensorflow.keras.models import load_model
def load_ensemble_models(model_names, config):
    print("\nLoading models")
    models = []
    for model_name in tqdm(model_names):
        model_paths = model_mapping_function(model_name, config)  # Get all model paths for the name
        for model_path in model_paths:
            print(f"Loading model from: {model_path}")
            model = load_model(model_path)
            models.append(model)
    return models

def predict_for_model(model, images, verbose=0):
    """
    Predicts the class probabilities for a single model.
    
    Args:
    model: The model to use for prediction.
    images (numpy array): Batch of images for inference.

    Returns:
    numpy array: Class probabilities from the model.
    """
    prediction = model.predict(images, verbose=verbose)
    return prediction

def print_model_summary(model):
    total_params = model.count_params()  # Total parameters (trainable + non-trainable)
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])  # Trainable parameters
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])  # Non-trainable parameters

    # Print the model summary information
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")


def plot_training_metrics(history, train_log, save_plots=True):
    """
    Function to plot training metrics including accuracy, validation accuracy, loss, 
    validation loss, and learning rate (LR).

    Parameters:
    - history: The history object returned by the model training.
    - train_log: A directory path where the plots should be saved.
    - save_plots: Boolean, if True, the plots will be saved as PNG files.
    """

    # Extract data from history
    epochs = range(1, len(history.history['loss']) + 1)
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    lr = history.history.get('lr')

    # Accuracy and Validation Accuracy Plot
    fig1 = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
    plt.title('Training and Validation Accuracy ' + train_log)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Save accuracy plot if save_plots is True
    if save_plots:
        acc_file = f'{train_log}-accuracy.png'
        acc_file_path = os.path.join(train_log, acc_file)
        fig1.savefig(acc_file_path)

    # Create subplots: 1 row, 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

    # Plot 1: Loss and Validation Loss
    axs[0].plot(epochs, loss, 'b', label='Training loss')
    axs[0].plot(epochs, val_loss, 'orange', label='Validation loss')
    axs[0].set_title('Training and Validation Loss ' + train_log)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot 2: Learning Rate (LR)
    axs[1].plot(epochs, lr, 'g', label='Learning rate')
    axs[1].set_title('Learning Rate ' + train_log)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Learning Rate')
    axs[1].legend()

    # Display the subplots
    plt.tight_layout()
    plt.show()

    # Save plots if save_plots is True
    if save_plots:
        plot_file = f'{train_log}-loss-lr-subplot.png'
        plot_file_path = os.path.join(train_log, plot_file)
        fig.savefig(plot_file_path)
            
def plot_training_metrics_from_df(history_df, train_log, save_plots=True):
    # Extract metrics from the DataFrame
    epochs = range(1, len(history_df) + 1)
    acc = history_df['accuracy']
    val_acc = history_df['val_accuracy']
    loss = history_df['loss']
    val_loss = history_df['val_loss']
    lr = history_df['lr']

    # Accuracy and Validation Accuracy Plot
    fig1 = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
    plt.title('Training and Validation Accuracy ' + train_log)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Save accuracy plot if save_plots is True
    if save_plots:
        os.makedirs(train_log, exist_ok=True)
        acc_file = f'{train_log}-accuracy.png'
        acc_file_path = os.path.join(train_log, acc_file)
        fig1.savefig(acc_file_path)

    # Create subplots: 1 row, 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

    # Plot 1: Loss and Validation Loss
    axs[0].plot(epochs, loss, 'b', label='Training loss')
    axs[0].plot(epochs, val_loss, 'orange', label='Validation loss')
    axs[0].set_title('Training and Validation Loss ' + train_log)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot 2: Learning Rate (LR)
    axs[1].plot(epochs, lr, 'g', label='Learning rate')
    axs[1].set_title('Learning Rate ' + train_log)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Learning Rate')
    axs[1].legend()

    # Display the subplots
    plt.tight_layout()
    plt.show()

    # Save plots if save_plots is True
    if save_plots:
        os.makedirs(train_log, exist_ok=True)
        plot_file = f'{train_log}-loss-lr-subplot.png'
        plot_file_path = os.path.join(train_log, plot_file)
        fig.savefig(plot_file_path)

def format_p_measures(p_measures):
    for key, value in p_measures.items():
        if key in ['TP', 'TN', 'FP', 'FN']:  # Keep these values as integers
            if isinstance(value, list):  # If value is a list, convert each item to int
                p_measures[key] = [int(v) for v in value]
            else:
                p_measures[key] = int(value)
        else:  # For other metrics, format to two decimal points
            if isinstance(value, list):  # If it's a list, round each item to 2 decimal places
                p_measures[key] = [round(v, 3) for v in value]
            elif isinstance(value, (float, int)):  # If it's a scalar, round it to 2 decimals
                p_measures[key] = round(value, 3)
    return p_measures

def verify_images(df, image_column):
    """
    Verifies the images in the DataFrame. Removes rows with corrupted or unidentifiable images.

    Parameters:
    - df: The pandas DataFrame containing image paths.
    - image_column: The name of the column in the DataFrame containing the image paths.

    Returns:
    - A DataFrame with only valid images.
    """
    valid_image_paths = []

    for idx, row in df.iterrows():
        image_path = row[image_column]
        try:
            # Try to open the image
            img = Image.open(image_path)
            img.verify()  # Verify the image (this catches image corruption)
            valid_image_paths.append(image_path)
        except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
            print(f"Skipping corrupted or invalid image: {image_path}, {e}")
    
    # Filter the DataFrame to keep only valid images
    return df[df[image_column].isin(valid_image_paths)]

     
#% Horizontal and vertical flip  
def augment_images_and_update_df(df, 
                                 #dest_dir,
                                 save=False):
    """
    Function to perform horizontal and vertical flips on images and append augmented images to the DataFrame.
    
    Args:
    - df (pd.DataFrame): DataFrame containing 'path' and 'class' columns.
    - dest_dir (str): Destination directory to save augmented images.
    
    Returns:
    - df (pd.DataFrame): Updated DataFrame with augmented image paths and their respective classes.
    """
    
    # List to store new augmented file paths and classes
    augmented_data = []

    # Loop through the DataFrame
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        file = row['path']
        class_id = str(row['class'])

        # Open the image
        try:
            img = Image.open(file)
        except FileNotFoundError:
            print(f"File not found: {file}")
            continue
        
        # Extract patient_id from the file path
        parts = file.split('/')
        patient_id = parts[3]

        # Horizontal flip
        img_h_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
       
        # Vertical flip
        img_v_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Save the flipped images
        base_dir, filename = os.path.split(file)
        filename_without_extension = os.path.splitext(filename)[0]
        
        # Destination paths for augmented images
        patient_dir = os.path.join(
            # dest_dir,
            #base_dir
            #class_id,
            #patient_id,
            base_dir.split(os.sep)[0],
            base_dir.split(os.sep)[1],
            base_dir.split(os.sep)[2],
            base_dir.split(os.sep)[3],
            'aug')
        # Horizontal flip path
        h_flip_path = os.path.join(patient_dir, f'{filename_without_extension}_h_flip.png')
        # Vertical flip path
        v_flip_path = os.path.join(patient_dir, f'{filename_without_extension}_v_flip.png')
        
        if save:
            # Ensure the destination directory exists
            os.makedirs(patient_dir, exist_ok=True)
            img_h_flip.save(h_flip_path)
            img_v_flip.save(v_flip_path)
        
        # Append the new augmented paths and classes to the list
        augmented_data.append({'path': h_flip_path, 'class': class_id})
        augmented_data.append({'path': v_flip_path, 'class': class_id})

    # Convert augmented data to DataFrame and append it to the original DataFrame
    augmented_df = pd.DataFrame(augmented_data)
    df = pd.concat([df, augmented_df], ignore_index=True)

    return df



















