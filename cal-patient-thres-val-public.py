#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

# from tensorflow.keras import models, layers, optimizers, Model
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

# from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.efficientnet import EfficientNetB7
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

from tensorflow.keras.optimizers import Adadelta, Nadam, Adam

# from tqdm import tqdm
from vit_keras import vit, utils, visualize

import random
import json
from yacs.config import CfgNode as CN

# from tensorflow.keras.applications.efficientnet import (
#     EfficientNetB7, preprocess_input)
from tensorflow.keras.applications import (vgg16, vgg19,
    EfficientNetB7, xception, efficientnet, inception_resnet_v2,
    inception_v3, densenet, resnet, resnet_v2, 
    mobilenet, mobilenet_v2)

from collections import Counter

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

# def seed_everything(seed = seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     os.environ['TF_DETERMINISTIC_OPS'] = '1'

# seed_everything()

print(f"{tf.__version__ = }") #2.11.0, python = 3.9.16
#2.3.0, python = 3.6
print(tf.test.gpu_device_name())

# print(f"{keras.__version__ = }") #2.11.0, python = 3.9.16
#2.4.0, python = 3.6

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


def plot_images_without_labels(images, train_log, set_label, save=False):
    """
    Plots a grid of images without displaying labels and optionally saves the figure.

    Args:
    - images (list or np.array): List of images to be displayed.
    - train_log (str): The directory or log name to save the plot.
    - set_label (str): The specific set label (e.g., "train", "validation", etc.).
    
    Returns:
    - None: Displays and optionally saves the plot.
    """
    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), dpi=120)
    axes = axes.flatten()

    for img, ax in zip(images, axes):
        # Undo any preprocessing (like rescaling for pre-trained models)
        if np.max(img) == 1:
            img = img * 255

        # Display the image
        img_shape = (config['DATA']['W'], config['DATA']['H'], config['DATA']['C'])
        ax.imshow(img.reshape(img_shape).astype("uint8"))

        # Remove axis labels and ticks
        ax.axis('off')

    # Set the overall title for the figure
    plot_title = f"{train_log}-{set_label}"
    plt.suptitle(plot_title, fontsize=18)

    # Show the plot
    plt.show()

    # Save the figure
    if save:
        plot_file = os.path.join(train_log, f"{plot_title}.png")
        fig.savefig(plot_file, dpi=120, bbox_inches='tight')
        print(f'Saved {plot_file}')

    
# Wrapper preprocessing function
def vit_preprocess_with_conversion(img):
    if isinstance(img, Image.Image):  # If the image is a PIL image (e.g., PngImageFile)
        img = np.asarray(img)  # Convert to NumPy array
    return vit.preprocess_inputs(img)  # Apply the ViT preprocessing after conversion
    
def get_preprocessing_function(model_name):
    # Mapping model names to their respective preprocessing functions
    preprocessing_mapping = {
        'VGG16': tf.keras.applications.vgg16.preprocess_input,
        'VGG19': tf.keras.applications.vgg19.preprocess_input,
        'EfficientNetB7': tf.keras.applications.efficientnet.preprocess_input,
        'Xception': tf.keras.applications.xception.preprocess_input,
        'InceptionResNetV2': tf.keras.applications.inception_resnet_v2.preprocess_input,
        'InceptionV3': tf.keras.applications.inception_v3.preprocess_input,
        'DenseNet121': tf.keras.applications.densenet.preprocess_input,
        'DenseNet201': tf.keras.applications.densenet.preprocess_input,
        'MobileNet': tf.keras.applications.mobilenet.preprocess_input,
        'MobileNetV2': tf.keras.applications.mobilenet_v2.preprocess_input,
        
        # EfficientNetV2 models use the same preprocessing function
        'EfficientNetV2B3': tf.keras.applications.efficientnet_v2.preprocess_input,
        'EfficientNetV2S': tf.keras.applications.efficientnet_v2.preprocess_input,
        'EfficientNetV2M': tf.keras.applications.efficientnet_v2.preprocess_input,
        'EfficientNetV2L': tf.keras.applications.efficientnet_v2.preprocess_input,
        
        # ConvNeXt models use the same preprocessing function
        'ConvNeXtTiny': tf.keras.applications.convnext.preprocess_input,
        'ConvNeXtSmall': tf.keras.applications.convnext.preprocess_input,
        'ConvNeXtBase': tf.keras.applications.convnext.preprocess_input,
        'ConvNeXtLarge': tf.keras.applications.convnext.preprocess_input,
        'ConvNeXtXLarge': tf.keras.applications.convnext.preprocess_input,
    }

    if model_name in preprocessing_mapping:
        return preprocessing_mapping[model_name]
    elif model_name.startswith('ViT'):
        return _preprocess_vit
    else:
        raise ValueError(f"No preprocessing function for model {model_name}")

def _preprocess_vit(img):
    if isinstance(img, Image.Image):
        img = np.asarray(img)
    return vit.preprocess_inputs(img)
    
def print_model_summary(model):
    total_params = model.count_params()  # Total parameters (trainable + non-trainable)
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])  # Trainable parameters
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])  # Non-trainable parameters

    # Print the model summary information
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
 
#%       
from tensorflow.keras.models import load_model
def load_ensemble_models(model_names):
    print("\nLoading models")
    models = []
    for model_name in tqdm(model_names):
        model_paths = model_mapping_function(model_name)  # Get all model paths for the name
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


def model_mapping_function(model_name):
    model_root = config['DATA']['SAVEPATH']  # Assuming config contains the save path
    
    model_mapping = {
        'EfficientNetV2S': [
            os.path.join(model_root,
                'Linux-Fold-0-EfficientNetV2S-bal-aug-20250110-1607.hdf5'),
            os.path.join(model_root,
                'Linux-Fold-1-EfficientNetV2S-bal-aug-20250110-1607.hdf5'),
            os.path.join(model_root,
                'Linux-Fold-2-EfficientNetV2S-bal-aug-20250110-1607.hdf5'),
            os.path.join(model_root,
                'Linux-Fold-3-EfficientNetV2S-bal-aug-20250109-1903.hdf5'), 
            os.path.join(model_root,
                'Linux-Fold-4-EfficientNetV2S-bal-aug-20250110-1607.hdf5'), 
        ],  
        }
    
    if model_name in model_mapping:
        return model_mapping[model_name]
    else:
        raise ValueError(f"No model mapping found for model {model_name}")

def custom_preprocessing_function(image, model_name):
    preprocessing_function = get_preprocessing_function(model_name)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_pre = preprocessing_function(image_bgr)
   
    return image_pre

import configsCellDemo
config = configsCellDemo.get_config()

from ModelBuilderDemo import ModelBuilder
from mygears import (get_png_files_and_classes, count_plot, 
                     create_balanced_dataframe,
                     plot_images_with_labels,
                     model_mapping_function,
                     load_ensemble_models,
                     predict_for_model,
                     )

from sklearn.metrics import roc_curve
import pickle

print(f"{tf.__version__ = }")

#%%

#% Inference for >= 2 models
t1 = datetime.now()
#% model_names for the ensemble
model_names = config['MODEL']['NAMES'] #['EfficientNetV2S']
model_name = model_names[0]
class_labels = ['IDA', 'THL']
dataset_root = config['DATA']['SET']
print(f'{dataset_root=}')

log = config['MODEL']['EVA_FILE_PATH']
os.makedirs(log, exist_ok=True)
print(f"Created {log=}")

#%%
models = load_ensemble_models(model_names, config)
print("Models have been loaded.")

#%%
with open("validation_sets.json", "r") as f:
    validation_sets = json.load(f)
    
print(f"{validation_sets=}")
    
#%%
if platform.system() == 'Linux':
    print('\nInference for each patient')
    # List of patient prediction dictionaries for all folds
    all_patient_predictions_mean_based = []
    all_patient_predictions_argmax_based = []
    
    for val_fold, val_set in enumerate(validation_sets):   
        # Select models that exclude the current validation fold
        train_folds = list(set(range(5)) - {val_fold})  # Get folds used for training
        models_to_use = [models[i] for i in train_folds]  # Select the models
        
        print(f"Fold {val_fold}: using models from folds: {train_folds}")
    
        # Initialize prediction storage
        patient_predictions_mean_based = {}
        patient_predictions_argmax_based = {}
        
        cnt = 1
        for patient_path, class_id in val_set:
            png_files = []
            class_list = []  # To store the corresponding class IDs
            
            print(f"\n\nSeq {cnt}: {patient_path}, {class_id}")
    
            patient_id = patient_path.split(os.path.sep)[-1]
            patch_path = os.path.join(patient_path, 'cells', '*', '*.png')
            
            # Gather PNG files
            png_files = glob.glob(patch_path)
            class_list.extend([class_id] * len(png_files))
            
            print(f"Patient {patch_path} (Class {class_id}): Found {len(png_files)} PNG files.")
            
            # Create patient dataframe
            patient_df = pd.DataFrame({
                'class': class_list,
                'path': png_files
            })
            
            print(f"{len(class_list)=}")
            count_plot(patient_df, f"{len(class_list)}")
            patient_df['class'] = patient_df['class'].astype(str)
            
            y_test_pred_ensemble = []
            y_test_pred_each_argmax_list = []
            
            # Loop through selected models and predict
            for model in models_to_use:
                print(f"Predicting with model: {model_name}, {model}")
    
                model_builder = ModelBuilder(model_name=model_name, config=config)
          
                def inner_custom_preprocessing_function(image):
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    preprocessing_function = model_builder.get_preprocessing_function()
                    return preprocessing_function(image_bgr)
                
                if config['DATA']['COLOR'] == 'RGB':
                    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                        preprocessing_function=model_builder.get_preprocessing_function()
                    )
                else:
                    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                        preprocessing_function=inner_custom_preprocessing_function
                    )
                
                if config['SAMPLE']:
                    patient_df = patient_df.sample(n=config['SAMPLE_SIZE'], random_state=42)
        
                patient_gen = datagen.flow_from_dataframe(
                    dataframe=patient_df,  
                    x_col='path',
                    batch_size=config['TRAIN']['BATCH_SIZE'],
                    shuffle=False,
                    class_mode=None
                )
                
                print(f"{patient_gen.n} patches loaded.")
                
                images = patient_gen.next()
                print(images.shape)
                
                plot_images_without_labels(images, "Infer", f"{patient_id}-{class_id}", save=True)
                
                # Perform inference
                y_test_pred_list = []
                for batch in tqdm(range(patient_gen.__len__())):
                    images = patient_gen.__getitem__(batch)
                    y_test_pred = predict_for_model(model, images, verbose=0)
                    y_test_pred_list.append(y_test_pred)
                    
                    # Majority hard-voting
                    y_test_pred_each_argmax = np.argmax(y_test_pred, axis=1)
                    y_test_pred_each_argmax_list.extend(y_test_pred_each_argmax)
                
                # Store predictions from this model
                y_test_preds = np.concatenate(y_test_pred_list, axis=0)
                y_test_pred_ensemble.append(y_test_preds)
    
            # Compute final prediction for the patient
            y_test_pred_avg = np.mean(y_test_pred_ensemble, axis=0)
            P_patient_mean = np.mean(y_test_pred_avg[:, 1])  # Soft Voting (mean probability for Thalassemia)
    
            # Store probability in patient-level predictions
            patient_predictions_mean_based[patient_id] = {
                "val_fold": val_fold,
                "probability_1": P_patient_mean,
                "class": class_id
            }
            
            # Argmax-Based Patient Classification
            y_test_pred_argmax = np.argmax(y_test_pred_avg, axis=1)
            unique, counts = np.unique(y_test_pred_argmax, return_counts=True)
            
            # Compute the proportion of patches classified as class 1 (Thalassemia)
            class_1_count = counts[unique == 1][0] if 1 in unique else 0
            total_patches = np.sum(counts)
            P_patient_argmax = class_1_count / total_patches
    
            # Store probability in patient-level predictions
            patient_predictions_argmax_based[patient_id] = {
                "val_fold": val_fold,
                "probability_1": P_patient_argmax,
                "class": class_id
            }
            
            cnt += 1
        # end all patients in each valset 
        
        # Append all fold predictions into a single list
        all_patient_predictions_mean_based.extend([
            {"val_fold": val_fold, "patient_id": pid, **data} for pid, data in patient_predictions_mean_based.items()
        ])
        
        all_patient_predictions_argmax_based.extend([
            {"val_fold": val_fold, "patient_id": pid, **data} for pid, data in patient_predictions_argmax_based.items()
        ])

#%
t2 = datetime.now() - t1
print('\nExecution times used: ', t2)  

#%% 

# Define the output CSV file names
output_csv_file_list = [
    "patient_predictions_mean_based_val.csv",
    "patient_predictions_argmax_based_val.csv"
]


#%% List of all patient predictions across folds
"""
all_patient_predictions_list = [
    all_patient_predictions_mean_based,
    all_patient_predictions_argmax_based
]

# Define the column headers
fieldnames = ["val_fold", "patient_id", "probability_1", "class"]

#%
print("\nWrite all folds’ predictions to CSV")
for patient_predictions, output_csv_file in zip(all_patient_predictions_list, output_csv_file_list):
    with open(output_csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()
        
        # Write all patient-level predictions for all folds
        writer.writerows(patient_predictions)

    print(f"Patient-level predictions saved to {output_csv_file}")

"""
    
#%%
def compute_optimal_patient_threshold(y_true_patient, y_score_patient):
    """Compute the optimal patient-level threshold using Youden's Index."""
    if len(np.unique(y_true_patient)) < 2:
        raise ValueError("ROC computation requires at least one positive and one negative case.")
    
    fpr, tpr, thresholds = roc_curve(y_true_patient, y_score_patient)
    youden_index = tpr - fpr  # Compute Youden's Index for each threshold
    optimal_threshold = thresholds[np.argmax(youden_index)]  # Get the best threshold
    return optimal_threshold

#%
print("\nCompute optimal thresholds across folds")
def compute_optimal_thresholds_across_folds(patient_predictions_df,
                                            method_name):
    patient_predictions_df = patient_predictions_df.dropna()
    optimal_thresholds_per_fold = []

    for val_fold in sorted(patient_predictions_df["val_fold"].unique()):
        print(f"[{method_name}] Processing Fold {val_fold}...")

        fold_df = patient_predictions_df[patient_predictions_df["val_fold"] == val_fold]
        y_true_patient = fold_df["class"].values.astype(int)
        y_score_patient = fold_df["probability_1"].values.astype(float)

        # Compute threshold using Youden’s Index
        T_upper_fval = compute_optimal_patient_threshold(y_true_patient, y_score_patient)
        optimal_thresholds_per_fold.append(T_upper_fval)

        print(f"[{method_name}] Optimal Threshold for Fold {val_fold}: {T_upper_fval:.3f}")

    T_upper = np.mean(optimal_thresholds_per_fold)
    print(f"\n[{method_name}] Averaged Optimal Threshold (T_upper): {T_upper:.3f}\n")
    return T_upper

#%%
print("\nLoad both fusion strategy CSVs")
df_mean = pd.read_csv("patient_predictions_mean_based_val.csv")
df_argmax = pd.read_csv("patient_predictions_argmax_based_val.csv")

# Compute thresholds
T_upper_mean = compute_optimal_thresholds_across_folds(df_mean, method_name="Mean-Based")
T_upper_argmax = compute_optimal_thresholds_across_folds(df_argmax, method_name="Argmax-Based")

"""
Compute optimal thresholds across folds
[Mean-Based] Processing Fold 0...
[Mean-Based] Optimal Threshold for Fold 0: 0.682
[Mean-Based] Processing Fold 1...
[Mean-Based] Optimal Threshold for Fold 1: 0.461
[Mean-Based] Processing Fold 2...
[Mean-Based] Optimal Threshold for Fold 2: 0.647
[Mean-Based] Processing Fold 3...
[Mean-Based] Optimal Threshold for Fold 3: 0.614
[Mean-Based] Processing Fold 4...
[Mean-Based] Optimal Threshold for Fold 4: 0.560

[Mean-Based] Averaged Optimal Threshold (T_upper): 0.593

[Argmax-Based] Processing Fold 0...
[Argmax-Based] Optimal Threshold for Fold 0: 0.784
[Argmax-Based] Processing Fold 1...
[Argmax-Based] Optimal Threshold for Fold 1: 0.428
[Argmax-Based] Processing Fold 2...
[Argmax-Based] Optimal Threshold for Fold 2: 0.717
[Argmax-Based] Processing Fold 3...
[Argmax-Based] Optimal Threshold for Fold 3: 0.663
[Argmax-Based] Processing Fold 4...
[Argmax-Based] Optimal Threshold for Fold 4: 0.578

[Argmax-Based] Averaged Optimal Threshold (T_upper): 0.634

"""

#%%
# List to store 10th percentile values per fold
lower_thresholds_per_fold = []
print("\nCompute the 10th percentile (T_lower) for each fold using Mean-Based Fusion")
for val_fold in sorted(df_mean["val_fold"].unique()):
    print(f"Processing Fold {val_fold}...")

    # Extract predicted probabilities for this fold (mean-based fusion)
    fold_df = df_mean[df_mean["val_fold"] == val_fold]
    y_score_patient = fold_df["probability_1"].values.astype(float)

    # Compute the 10th percentile (T_lower_fval)
    T_lower_fval = np.percentile(y_score_patient, 10)
    lower_thresholds_per_fold.append(T_lower_fval)

    print(f"10th Percentile (T_lower) for Fold {val_fold}: {T_lower_fval:.6f}")

# Compute the final averaged T_lower across all validation folds
T_lower = np.mean(lower_thresholds_per_fold)
print(f"\nAveraged Lower Threshold (T_lower): {T_lower:.6f}")

"""
Compute the 10th percentile (T_lower) for each fold using Mean-Based Fusion
Processing Fold 0...
10th Percentile (T_lower) for Fold 0: 0.119946
Processing Fold 1...
10th Percentile (T_lower) for Fold 1: 0.218088
Processing Fold 2...
10th Percentile (T_lower) for Fold 2: 0.603034
Processing Fold 3...
10th Percentile (T_lower) for Fold 3: 0.223927
Processing Fold 4...
10th Percentile (T_lower) for Fold 4: 0.133125

Averaged Lower Threshold (T_lower): 0.259624
"""

#%%




















