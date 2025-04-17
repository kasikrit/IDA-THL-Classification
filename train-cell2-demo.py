# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:29:07 2023

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
from PIL import Image, UnidentifiedImageError

from tensorflow.keras import models, layers, optimizers, Model
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
    array_to_img, img_to_array, load_img)
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import backend as K
# from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical

import tensorflow_addons as tfa
tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)

class CustomTQDMProgressBar(tfa.callbacks.TQDMProgressBar):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Safely extract the learning rate as a scalar for logging
        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        
        # Add the learning rate to the logs for display
        logs['learning_rate'] = lr
        super().on_epoch_end(epoch, logs)
# from tensorflow_addons.callbacks import TQDMProgressBar
# tqdm_callback = CustomTQDMProgressBar()

# from livelossplot import PlotLossesKeras

from tensorflow.keras.callbacks import ModelCheckpoint

# import tensorflow.keras.layers as L
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
# from imblearn.over_sampling import RandomOverSampler
# from imutils import paths

# from keras.applications import ResNet50
# from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, GlobalAveragePooling2D,
    BatchNormalization, Dropout, concatenate)
# from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
# from keras_applications.resnet import ResNet50

from tensorflow.keras.applications.efficientnet_v2 import (
    EfficientNetV2B3,
    EfficientNetV2S,
    EfficientNetV2M,
    EfficientNetV2L,
    preprocess_input)

from tensorflow.keras.applications.convnext import(
    ConvNeXtTiny,
    ConvNeXtSmall,
    ConvNeXtBase,
    ConvNeXtLarge,
    ConvNeXtXLarge,
    preprocess_input)

from tensorflow.keras.applications.efficientnet import (
    EfficientNetB7, preprocess_input)
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2, preprocess_input)
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3, preprocess_input)
from tensorflow.keras.applications.densenet import (
    DenseNet121, DenseNet201, preprocess_input)

from tensorflow.keras.applications.resnet import (
    ResNet50, ResNet101, ResNet152, preprocess_input)
# from tensorflow.keras.applications.resnet_v2 import (
#     ResNet101V2, ResNet50V2, preprocess_input)

from tensorflow.keras.applications.mobilenet import (
    MobileNet, preprocess_input)

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input)

from tensorflow.keras.applications.vgg16 import (
    VGG16, preprocess_input)

from tensorflow.keras.applications.vgg19 import (
    VGG19, preprocess_input)

from tensorflow.keras.optimizers import Adadelta, Nadam, Adam
from tensorflow.keras.models import load_model

# from tqdm import tqdm
from vit_keras import vit, utils, visualize

import random
import json
from yacs.config import CfgNode as CN
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
 
from tensorflow.keras.applications import (vgg16, vgg19,
    EfficientNetB7, xception, efficientnet, inception_resnet_v2,
    inception_v3, densenet, resnet, resnet_v2, 
    mobilenet, mobilenet_v2)

def cfg_to_dict(cfg_node):
    """
    Recursively convert a YACS CfgNode to a nested dictionary.
    """
    if isinstance(cfg_node, CN):
        return {k: v for k, v in cfg_node.items()}
    return cfg_node

def filter_hidden_files(file_paths):
    """
    Filter out paths that start with a '.' indicating they are hidden.
    This only considers the filename, not part of the path, as hidden.
    """
    return [path for path in file_paths if not os.path.basename(path).startswith('.')]

from imblearn.over_sampling import RandomOverSampler
from collections import Counter

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

# def seed_everything(seed = seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     os.environ['TF_DETERMINISTIC_OPS'] = '1'

# seed_everything()

# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
def make_pair(imgs,labels):
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
    
def count_plot(df, title):
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
    plt.show()

     
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

def print_model_summary(model):
    total_params = model.count_params()  # Total parameters (trainable + non-trainable)
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])  # Trainable parameters
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])  # Non-trainable parameters

    # Print the model summary information
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")

from collections import defaultdict
def get_png_files_and_classes(
        # dataset_root,
        patient_set,
        aug=False
    ):
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
        # print(class_id, patient_id)
        # break
        # Construct the patch path
        if aug:
            patch_path = os.path.join(
                # dataset_root,
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
            patch_path = os.path.join(
                # dataset_root,
                #class_id,
                patient_path,
                'cells',
                '*',
                '*.png')  
            print(f"{patch_path=}")    
            png_files = glob.glob(patch_path)
        
        patient_id = patient_path.split(os.sep)[2]
        print(f"Patient {patient_id} (Class {class_id}): Found {len(png_files)} PNG files.")
        
        # Append png_files to png_list and class_list
        png_list.extend(png_files)        # Add all PNG file paths to png_list
        class_list.extend([class_id] * len(png_files))  # Add the class_id for each PNG file
        
        # Update the count of PNG files for the current class_id
        class_counts[class_id] += len(png_files)
    
    # Return both lists and the class counts
    return png_list, class_list, class_counts

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

def plot_images_with_labels(images, labels, train_log, set_label, save=False):
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
    if save:
        plot_file = os.path.join(train_log, f"{plot_title}.png")
        fig.savefig(plot_file, dpi=120)
        print(f'Saved {plot_file}')
        
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
        # patient_id = parts[3]

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
        
import configsCell
config = configsCell.get_config()

from ModelBuilderDemo import ModelBuilder, MetricsTracker

print(f"{tf.__version__=}") #2.11.0, python = 3.9.16
#2.3.0, python = 3.6
print(f"{tf.test.gpu_device_name()=}")

print(f"{keras.__version__=}") #2.11.0, python = 3.9.16
#2.4.0, python = 3.6

#%%
t_begin = datetime.now()
# Sanity check
dataset_root = config['DATA']['SET']
class_labels = config['DATA']['CLASS_LABELS']
print(f"{dataset_root=}, {class_labels=}")

dir_0_list = [
    os.path.join(config['DATA']['SET'], '0'),
]

dir_1_list = [
    os.path.join(config['DATA']['SET'], '1'),
]
# Get the directory names
class_0_list = list()
for dir_path in dir_0_list:
    #directories_0 = [f for f in os.listdir(dir_class) if not f.startswith('.')]
    dir_class_list = []
    for f in os.listdir(dir_path):
        if not f.startswith('.'):
            dir_class_list.append(os.path.join(dir_path,
                            f))              
    class_0_list.extend(dir_class_list)
    
class_1_list = list()
for dir_path in dir_1_list:
    #directories_0 = [f for f in os.listdir(dir_class) if not f.startswith('.')]
    dir_class_list = []
    for f in os.listdir(dir_path):
        if not f.startswith('.'):
            dir_class_list.append(os.path.join(dir_path,
                            f))              
    class_1_list.extend(dir_class_list)
    
# Combine directory names from both '0' and '1'
all_directories = [ (d, '0') for d in class_0_list] + \
                  [ (d, '1') for d in class_1_list]

print(f"{len(all_directories)=}")

#%%
# test_set_ano = []
# patient_maps = pd.read_csv('anonymized_patient_mapping.csv')
# print(patient_maps.columns)

# lookup = {
#     (str(row["patient_id"]), str(row["class"])): row["anonymized_id"]
#     for _, row in patient_maps.iterrows()
# }

# lookup.get(('20051-54', '1')) # 'THL-041'
# #('IDA-THL-Dataset-Phase-I/1/THL-064',

# # Convert to anonymized form
# test_set_ano = [
#     (lookup.get((pid, cls), f"{pid}_{cls}_NOT_FOUND"), cls)
#     for pid, cls in test_set
# ]

# print(f"{test_set_ano=}")
# test_set = test_set_ano

test_set = [
        ('IDA-016', '0'),
        ('IDA-023', '0'),
        ('IDA-031', '0'),
        ('IDA-032', '0'),
        ('IDA-033', '0'),
        ('IDA-044', '0'),
        
        ('THL-040', '1'),
        ('THL-041', '1'),
        ('THL-048', '1'),
        ('THL-053', '1'),
        ('THL-065', '1'),
        ('THL-067', '1'),
]
# test_set.sort()
print(test_set)
print(f"\n{len(test_set)=}")

#%%
train_set_single = []
test_set_single = []
for (patient_path, class_id) in all_directories:
    patient_id = patient_path.split(os.sep)[-1]
    if (patient_id,  class_id) not in test_set:
        train_set_single.append((patient_path, class_id))
    else:
        test_set_single.append((patient_path, class_id))
        
train_set_single.sort()
test_set_single.sort()
print(f"{len(train_set_single)=}")   
print(f"{len(test_set_single)=}")   

#%%
# Call the function with the test set
print('\nCreate test data frames')
patchPaths_test, class_list_test, class_counts_test = get_png_files_and_classes(
    #dataset_root,
    test_set_single,
    aug=True)
print(f"Total number of PNG files: {len(patchPaths_test)}") #20072
print(f"Total number of class IDs: {len(class_list_test)}")
print(f"Class counts: {dict(class_counts_test)}")     
test_df = pd.DataFrame({
     'class': class_list_test,
     'path': patchPaths_test
 })
print(f"Test set: {len(class_list_test)}")
count_plot(test_df, f"Test set: {len(class_list_test)}")

test_df = create_balanced_dataframe(patchPaths_test, class_list_test)
print(test_df.sample(3))
print(f"Test set bal: {len(test_df['class'])}")
count_plot(test_df, f"Test set bal: {len(test_df['class'])}")

# %%
from sklearn.model_selection import KFold

# Define the number of folds
FOLD = 5

# Initialize KFold from sklearn
kf = KFold(n_splits=FOLD,
           shuffle=True,
           random_state=2024
)

# Lists to hold training and validation sets
training_sets = []
validation_sets = []

print('\nSplit the shuffled data into 5 equal-sized folds')

# Split the data into 5 folds using KFold
folds = []
for fold_idx, (train_indices, val_indices) in enumerate(kf.split(train_set_single)):
    validation_set = [train_set_single[i] for i in val_indices]  # Validation set for current fold
    training_set = [train_set_single[i] for i in train_indices]  # Training set for current fold
    folds.append((training_set, validation_set))

for i, (training_set, validation_set) in enumerate(folds):
    training_sets.append(training_set)
    validation_sets.append(validation_set)
    
    # Print the sizes of the training and validation sets for each fold
    print(f"Fold {i}: Training set size = {len(training_set)}, "
          f"Validation set size = {len(validation_set)}, "
          f"Ratio = {len(validation_set)*100/len(training_set):.2f}%")
    
#%%
print("\nInitialize counters for both classes in training_sets")
ida_count = 0
thalas_count = 0

# Assuming training_sets is a list of 5 folds, each fold being a list of tuples
for i, fold in enumerate(training_sets):
    ida_count = sum(1 for _, label in fold if label == '0')  # Count IDA subjects
    thal_count = sum(1 for _, label in fold if label == '1')  # Count Thal subjects
    
    print(f"Fold {i}: IDA = {ida_count}, Thalassemia = {thal_count}")

#%%
print("\nInitialize counters for both classes in validation_sets")
ida_count = 0
thalas_count = 0

# Assuming training_sets is a list of 5 folds, each fold being a list of tuples
for i, fold in enumerate(validation_sets):
    ida_count = sum(1 for _, label in fold if label == '0')  # Count IDA subjects
    thal_count = sum(1 for _, label in fold if label == '1')  # Count Thal subjects
    
    print(f"Fold {i}: IDA = {ida_count}, Thalassemia = {thal_count}")


#%% Optional: Print the contents of each fold (training and validation sets)
# for i in range(FOLD):
#     print(f"\nFold {i}:")
#     print(f"Training set: {training_sets[i]}")
#     print(f"Validation set: {validation_sets[i]}")

#%%
# Example usage: setting the environment for each seed in _C.DATA.SEEDS
# for seed in config['DATA']['SEEDS']:
#     print('\n\nseed: ', seed)
#     seed_everything(seed)
for fold in [
        0, 
        # 1,
        # 2,
        # 3,
        # 4,
        ]:
    print("\n\nFold:", fold)    
    train_set = training_sets[fold]
    val_set = validation_sets[fold]
    
    patchPaths_train, class_list_train, class_counts_train = get_png_files_and_classes(
        # dataset_root,
        train_set,
        aug=True)
    print(f"Total number of PNG files: {len(patchPaths_train)}") 
    print(f"Total number of class IDs: {len(class_list_train)}")
    print(f"Class counts: {dict(class_counts_train)}")
    
    print('\nCreate train data frames')      
    train_df = pd.DataFrame({
        'class': class_list_train,
        'path': patchPaths_train
    })
    # print('train: ', type(train_df), train_df.shape)
    # print(train_df.sample(3))
    count_plot(train_df, f"Train set {fold}: {len(class_list_train)}") 
    
    train_df_bal = create_balanced_dataframe(
        train_df['path'],
        train_df['class'])
    print(train_df_bal.sample(3))  # Print a few samples from the balanced dataframe
    print(f"Train set bal {fold}: {len(train_df_bal['class'])}")
    count_plot(train_df_bal, f"Train set bal {fold}: {len(train_df_bal['class'])}")
    
  
    patchPaths_val, class_list_val, class_counts_val = get_png_files_and_classes(
        # dataset_root,
        val_set,
        aug=True)
    print(f"Total number of PNG files: {len(patchPaths_val)}") 
    print(f"Total number of class IDs: {len(class_list_val)}")
    print(f"Class counts: {dict(class_counts_val)}")
    
    print('\nCreate val data frames')      
    val_df = pd.DataFrame({
        'class': class_list_val,
        'path': patchPaths_val
    })
    print('val: ', type(val_df), val_df.shape)
    # print(val_df.sample(3))
    
    count_plot(val_df, f"Validation set {fold}: {len(class_list_val)}") 
    
    val_df_bal = create_balanced_dataframe(
        val_df['path'],
        val_df['class'])
    print(val_df_bal.sample(3))  # Print a few samples from the balanced dataframe
    print(f"Validate set bal {fold}: {len(val_df_bal['class'])}")
    count_plot(val_df_bal, f"Validate set bal {fold}: {len(val_df_bal['class'])}")     

    train_df = train_df_bal.copy()
    val_df = val_df_bal.copy()
    
    # stop
    
    #%%
    #% Run only first time for each fold
    # print("\nPerforming augmentation and update the DataFrame")
    # train_df_bal_aug = augment_images_and_update_df(
    #     train_df_bal,
    #     #dataset_root,
    #     # save=False,
    #     save=True
    #     )
    # print(train_df_bal_aug.describe())
    # count_plot(train_df_bal_aug, f"Traing set bal-aug {fold}: {len(train_df_bal_aug['class'])}")
    # print(f"Traing set bal-aug {fold}: {len(train_df_bal_aug['class'])}")
     
    # val_df_bal_aug = augment_images_and_update_df(
    #     val_df_bal,
    #     #dataset_root,
    #     # save=False,
    #     save=True
    #     )
    # print(val_df_bal_aug.describe())
    # count_plot(val_df_bal_aug, f"Validate set bal-aug {fold}: {len(val_df_bal_aug['class'])}")
    # print("Augmentation completed")
    # print(f"Validate set bal-aug {fold}: {len(val_df_bal_aug['class'])}")

    # train_df = train_df_bal_aug.copy()
    # val_df = val_df_bal_aug.copy()
    
    #%% sanicheck
    # png_list = []
    # for path in train_set:
    #     patch_path = '20241020/0/*/cells/*/aug/*.png'
    #     png_files = glob.glob(patch_path)
    #     print(f"{len(png_files)=}")
    #     png_list.extend(png_files)
        
    # # Remove all files in png_list
    # for file_path in png_list:
    #     try:
    #         os.remove(file_path)  # Remove the file
    #         #print(f"Deleted: {file_path}")
    #     except FileNotFoundError:
    #         print(f"File not found: {file_path}")
    #     except PermissionError:
    #         print(f"Permission denied: {file_path}")
     
    
     #%%          
    if platform.system() == 'Linux' and config['DATA']['VERIFY']:
        print("\nVerify images in the train and validation dataframes")
        train_df = verify_images(train_df, 'path')
        val_df = verify_images(val_df, 'path')
        print("Verify completed")
      
    #%%    
    for model_name in config['MODEL']['NAMES']:
        # print(model_name)
        train_log = config['BASE'] + '-' + \
            f"Fold-{fold}" + '-' + \
            model_name + '-' + 'bal-aug' + '-' + \
            config['TRAIN']['DATETIME']        
        print(f"{train_log=}")
        
        if config['SAVE'] == True:
            os.makedirs(train_log, exist_ok=True)
            cfg_dict = cfg_to_dict(config)
            json_file_path = os.path.join(train_log,'_config.json')  
            with open(json_file_path, 'w') as json_file:    
                json.dump(cfg_dict, json_file, indent=4)
            
            print("Saved: ", json_file_path)
        
        model_builder = ModelBuilder(model_name=model_name, config=config)
        model = model_builder.create_model()
        preprocessing_function = model_builder.get_preprocessing_function()       
        # print_model_summary(model)
        print(model.summary())
         
        #%%
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
        
        def custom_preprocessing_function(image):
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # print("DEBUG: 1")
            image_pre = preprocessing_function(image_bgr)
            # print(image_pre.shape)
            # print("DEBUG: 2")
            return image_pre
        
        def preprocess_for_alexnet(image):
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # image_resized = cv2.resize(image_bgr, (227, 227))
            image_resized = image_bgr.astype(np.float32)
    
            # 4. Subtract ImageNet mean values for each channel (RGB)
            mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
            image_normalized = image_resized - mean
    
            return image_normalized
        
        #%%
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            # rescale = 1./255,
            # samplewise_center = True,
            # samplewise_std_normalization = True,
            # preprocessing_function = data_augment,
            # rotation_range=20,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # rescale=1./255,
            # shear_range=0.2,
            # zoom_range=0.2,
            # horizontal_flip=True,
            # vertical_flip=True,
            # fill_mode='nearest',
            # fill_mode='reflect',
            # preprocessing_function = preprocessing_function,
            preprocessing_function = custom_preprocessing_function
            # preprocessing_function = preprocess_for_alexnet
            )                                
        print('\nTrain_gen')
        # Convert the 'class' column to strings
        train_df['class'] = train_df['class'].astype(str)      
        train_gen  = datagen.flow_from_dataframe(
            dataframe=train_df.sample(config['TRAIN']['SAMPLE_SIZE'])
                if config['TRAIN']['SAMPLE'] else train_df,
            x_col = 'path',
            y_col = 'class',
            batch_size = config['TRAIN']['BATCH_SIZE'],
            # target_size=(227, 227), #for alexnet
            # seed = seed,
            shuffle = True,
            class_mode='categorical',
            ) 
        print(f"{train_gen.n}")
        print(f"{Counter(train_gen.classes)=}")
        
        print('\nval_gen')
        val_df['class'] = val_df['class'].astype(str)
        val_gen  = datagen.flow_from_dataframe(
            dataframe=val_df.sample(int(config['TRAIN']['SAMPLE_SIZE'] * config['TRAIN']['SAMPLE_SIZE_VAL']))
                if config['TRAIN']['SAMPLE'] else val_df,
            x_col = 'path',
            y_col = 'class',
            batch_size = config['TRAIN']['BATCH_SIZE'],
            shuffle = True,
            class_mode='categorical',
            )
        print(f"{val_gen.n}")
        print(f"{Counter(val_gen.classes)=}")
        
        print('\ntest_gen')
        test_df['class'] = test_df['class'].astype(str)
        test_gen  = datagen.flow_from_dataframe(
              dataframe=test_df.sample(int(config['TRAIN']['SAMPLE_SIZE'] * config['TRAIN']['SAMPLE_SIZE_VAL']))
                  if config['TRAIN']['SAMPLE'] else test_df,
              x_col = 'path',
              y_col = 'class',
              batch_size = config['TRAIN']['BATCH_SIZE'],
              # shuffle = False, # set to false for actual evaluation
              shuffle = True,  # set to true for demo
              class_mode='categorical',
              )
        print(f"{test_gen.n}")
        print(f"{Counter(test_gen.classes)=}")
          
        #%%
        for set_label, demo_gen in zip(['train', 'val', 'test'],
                        [train_gen, val_gen, test_gen]):       
            images, labels = demo_gen.next()
            print(images.shape, labels.shape)
            print(Counter(np.argmax(labels, axis=1)))
            print(f"{np.min(images[0])=}")
            print(f"{np.max(images[0])=}")
            plot_images_with_labels(images, labels,
                train_log, set_label,
                save=False,
            )
            
        #%%
        from tensorflow.keras.callbacks import Callback
        
        class LRTensorBoard(Callback):
            def __init__(self, log_dir, init_lr, lr_multiplier, file):
                super().__init__()
                self.log_dir = log_dir
                self.init_lr = init_lr
                self.lr_multiplier = lr_multiplier
                self.iterations = 0
                self.losses = []
                self.lrs = []
                self.file = file
            
            def on_train_batch_end(self, batch, logs=None):
                lr = self.init_lr * (self.lr_multiplier ** self.iterations)
                self.lrs.append(lr)
                self.losses.append(logs['loss'])
                self.iterations += 1
            
            def on_epoch_end(self, epoch, logs=None):
                if self.iterations == 0:
                    return
                plt.figure(figsize=(8, 6), dpi=120)
                plt.plot(self.lrs, self.losses)
                plt.xscale('log')
                plt.xlabel('Learning Rate')
                plt.ylabel('Loss')
                plt.title('LR Range Test')
                plt.grid(True)
                plt.savefig(f'{self.log_dir}/{self.file}')
                plt.close()
     
        
        #%%
        print("\nConfigure model for training")
        optimizer = tfa.optimizers.AdamW(
            learning_rate=config['TRAIN']['LR'],
            weight_decay=config['TRAIN']['LR'])                 
        print(type(optimizer))
        
        model.compile(
            optimizer=optimizer,
            loss ='binary_crossentropy',
            metrics=['accuracy']
            )
        
        backup_model_best = os.path.join(
            config['DATA']['SAVEPATH'],
            f'{train_log}.hdf5')       
        print('\nbackup_model_best: ', backup_model_best)

        reduceLROnPlat = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=config['TRAIN']['LR_FACTOR'],  # Try values like 0.5 or 0.7
            patience=config['TRAIN']['reduceLR_patience'],  # Increase patience if needed
            verbose=1,
            mode='min',  # Set to 'min' to reduce on improvement
            min_delta=config['TRAIN']['LR'],  # A smaller threshold
            cooldown=2,  # Optional, gives some cooldown period
            # min_lr=1e-7  # Set a lower limit for the learning rate
            )
        
        early_stopping_callbacks = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Monitor the same metric
            patience=config['TRAIN']['stop_patience'],  # Set a higher patience than ReduceLROnPlateau
            restore_best_weights=True,
            verbose=1
        )

        mcp2 = ModelCheckpoint(backup_model_best,
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                    mode='min'
        )
                  
        #%%    
        print("\nPerform training...")
        t1 = datetime.now()
        
        model.reset_states()
        # Prepare the list of callbacks
        callbacks_list = [
            mcp2,
            reduceLROnPlat,
            early_stopping_callbacks,
            tqdm_callback,
        ]
        if config['SAVE']:
            csv_file_path = os.path.join(train_log, f"{train_log}.csv")
            metrics_tracker = MetricsTracker(csv_file_path)
            callbacks_list.append(metrics_tracker)
            
        with tf.device('/device:GPU:0'):
            history = model.fit(
                train_gen,  
                validation_data = val_gen,
                epochs = config['TRAIN']['EPOCHS'],
                verbose = 2,
                callbacks = callbacks_list,
                )
            del model
                
        t2 = datetime.now() - t1
        print('\nTraining Time used: ', t2)
              
        #%%
        if config['SAVEHIST'] == True:
            model_history_df = pd.DataFrame(history.history) 
            
            # with open('unet_history_df.csv', mode='w') as f:
            #     unet_history_df.to_csv(f)
                
            # with open('att_unet_history_df.csv', mode='w') as f:
            #     att_unet_history_df.to_csv(f)
            
            history_file = f'{train_log}_history_df.csv'
            # history_file_path = os.path.join(model_name, history_file)
            with open(os.path.join(train_log, history_file), mode='w') as f:
                model_history_df.to_csv(f)  
            print("\nSaved: ", history_file)
                
        plot_training_metrics(history=history, train_log=train_log)  

        #%%        
        # best_model = load_model('/Users/kasikritdamkliang/Datasets/Thalas/Dataset_NT1/models/Linux-seed1337-ViT_b16-20240918-0057.hdf5')
        # print('\nLoaded best model')
        # print_model_summary(best_model)
        
        best_model = load_model(backup_model_best)
        print("\nLoaded: ", backup_model_best)
        print_model_summary(best_model)
        
        #%%
        print("\nEvaluate Val set")
        scores = best_model.evaluate(val_gen, verbose=2)
        
        for metric, value in zip(best_model.metrics_names, scores):
            print("mean {}: {:.2}".format(metric, value))
            
        print() 
    
        #%% 
        print("\nEvaluate Test set")
        scores_test = best_model.evaluate(test_gen, verbose=2)
        for metric, value in zip(best_model.metrics_names, scores_test):
            print("mean {}: {:.2}".format(metric, value))
            
        print() 
        
        
        #%%
        print("\nEvaluate the whole test set")
        t3 = datetime.now()
        y_test_pred_list = []
        pair_idx_test_list = []
           
        for batch in tqdm(range(test_gen.__len__())):
            # print('\nPredicting batch: ', batch)
            x_test, y_test = test_gen.__getitem__(batch)
            print(x_test.shape, y_test.shape) 
            # print(X_test.shape, y_test.shape, len(pair_idx_test))
            # y_test_argmax = np.argmax(y_test, axis=3).astype('uint8')
            verbose=2
            with tf.device('/device:GPU:0'):
                y_test_pred = best_model.predict(
                    # test_generator_RBF.__getitem__(image_number),
                    x_test,   
                    batch_size=config['TRAIN']['BATCH_SIZE'], 
                    verbose=verbose)
        
            y_test_pred_list.append(y_test_pred)
              
        y_test_pred = np.concatenate(y_test_pred_list, axis=0)
        print(y_test_pred.shape)
        
        y_test_pred_argmax = np.argmax(y_test_pred, axis=1)
        print(y_test_pred_argmax.shape)
        
        t4 = datetime.now() - t3
        print('Execution times: ', t4, '\n')
             
        #%%
        true_classes = test_gen.classes
        predicted_classes = y_test_pred_argmax
        print('Test set')
        print(classification_report(true_classes,
            predicted_classes,
            target_names=config['DATA']['CLASS_LABELS'])
            )
        
        class_report = classification_report(true_classes,
            predicted_classes,
            target_names=config['DATA']['CLASS_LABELS'],
            output_dict = True)
        
        #%%
        confusionmatrix = confusion_matrix(true_classes,
                                    predicted_classes)
        
        FP = confusionmatrix.sum(axis=0) - np.diag(confusionmatrix)  
        FN = confusionmatrix.sum(axis=1) - np.diag(confusionmatrix)
        TP = np.diag(confusionmatrix)
        # TN = confusionmatrix.values.sum() - (FP + FN + TP)
        TN = confusionmatrix.sum() - (FP + FN + TP)
        
        
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)
        sen = TPR.mean()
        spec = TNR.mean()
        
        print(sen, spec)
        
        p_measures = dict()
        p_measures = {
            # 'Training': t2,           
            'TP': TP, 
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'TPR': TPR,
            'TNR': TNR,
            'PPV': PPV,
            'NPV': NPV,
            'FPR': FPR,
            'FNR': FNR,
            'FDR': FDR,
            'ACC': ACC,
            
            'precision': class_report['macro avg']['precision'],
            'f1-score' : class_report['macro avg']['f1-score'],
            'MeanAcc': ACC.mean(),
            'MeanSen': sen,
            'MeanSpec': spec,           
        }
               
        # Convert numpy arrays to lists
        for key, value in p_measures.items():
            if isinstance(value, np.ndarray):
                p_measures[key] = value.tolist()
        # print(p_measures)
        p_measures = format_p_measures(p_measures)
        print(p_measures)    
        # Define file path
        file_path = os.path.join(train_log, train_log + '_p_measures.json')
        
        # Write to JSON file
        with open(file_path, 'w') as file:
            json.dump(p_measures, file, indent=4)
        
        print(file_path)
        
        #%%
        fig = plt.figure(figsize = (6, 4), dpi=300)
        sns.heatmap(confusionmatrix,
                    cmap = 'Blues',
                    annot = True,
                    cbar = True,
                    fmt="d",  # Format as integer
                    xticklabels=config['DATA']['CLASS_LABELS'],
                    yticklabels=config['DATA']['CLASS_LABELS'])
        plt.ylabel('Actual', fontsize=10)
        plt.xlabel("Predicted\n \
                   Accuracy={:0.2f}\n \
                   Sensitivity={:0.2f}\n \
                   Specificity={:0.2f}\n".format(ACC.mean(),
                                           TPR.mean(),
                                           TNR.mean())
                   ,
                  fontsize=10)
        plt.tight_layout()
        plt.show()
        
        filename = os.path.join(train_log, train_log + '-confuseMatrix.png')
        fig.savefig(filename)
        print(filename)
        
        #%%
        confusionmatrix_normalized = confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis]
        
        fig = plt.figure(figsize = (6, 4), dpi=300)
        sns.heatmap(confusionmatrix_normalized,
                    cmap = 'Blues',
                    annot = True,
                    cbar = True,
                    fmt=".2f",
                    xticklabels=config['DATA']['CLASS_LABELS'],
                    yticklabels=config['DATA']['CLASS_LABELS']
                    )
        plt.ylabel('Actual', fontsize=10)
        plt.xlabel("Predicted\n \
                   Accuracy={:0.2f}\n \
                   Sensitivity={:0.2f}\n \
                   Specificity={:0.2f}\n".format(ACC.mean(),
                                           TPR.mean(),
                                           TNR.mean())
                   ,
                  fontsize=10)
        plt.tight_layout()
        plt.show()
        filename = os.path.join(train_log, train_log + '-confuseMatrixNorm.png')
        fig.savefig(filename)
        print(filename)
        
#%%
t_end = datetime.now() - t_begin
print('\nAll execution time used: ', t_end)
 
#%%










