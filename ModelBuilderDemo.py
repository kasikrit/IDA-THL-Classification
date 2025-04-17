import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import (VGG16, VGG19, EfficientNetB7, Xception, 
                                           InceptionResNetV2, InceptionV3, DenseNet121, 
                                           DenseNet201, MobileNet, MobileNetV2)
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, Flatten)
from tensorflow.keras.regularizers import l2
from vit_keras import vit, utils, visualize
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import os

import configsCell
config = configsCell.get_config()

class ModelBuilder:
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        self.base_model = None
        self.model = None
        self.preprocessing_function = None
    
    def create_model(self):
        vit_models = ['ViT_b16', 'ViT_l16', 'ViT_b32', 'vit_l32']
        # convnext_models = ['ConvNeXtTiny', 'ConvNeXtSmall', 'ConvNeXtBase', 'ConvNeXtLarge', 'ConvNeXtXLarge']
        
        if self.model_name in vit_models:
            self.model = self._create_vit_model()
        # elif self.model_name in convnext_models:
        #     self.model = self._load_convnext_model()
        else:
            self.model = self._create_cnn_model()
        
        # print(self.model.summary())
        print(f'\nCreated: {self.model.name}')
        # self.base_model.trainable = True
        print(f'{self.base_model.trainable=}')
        return self.model
    
    def _create_cnn_model(self):
        eff_net_list = [
            'EfficientNetV2B3',
            'EfficientNetV2S',
            'EfficientNetV2M',
            'EfficientNetV2L',
        ]
        
        convNeX_list = [
        'ConvNeXtTiny',
        'ConvNeXtSmall',
        'ConvNeXtBase',
        'ConvNeXtLarge',
        'ConvNeXtXLarge',
        ] 
        print(f"{self.model_name=}")
        
        if self.model_name in eff_net_list:
            base_model = self._load_efficientnet_model()
        elif self.model_name in convNeX_list: 
            base_model = self._load_convnext_model()
            print(type(base_model))
        elif self.model_name == 'VGG16':
            base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
        elif self.model_name == 'VGG19':
            base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
        elif self.model_name == 'EfficientNetB7':
            base_model = tf.keras.applications.EfficientNetB7(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
        elif self.model_name == 'Xception':
            base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
        elif self.model_name == 'InceptionResNetV2':
            base_model = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
        elif self.model_name == 'InceptionV3':
            base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
        elif self.model_name == 'DenseNet121':
            base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
        elif self.model_name == 'DenseNet201':
            base_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
        elif self.model_name == 'MobileNet':
            base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
        elif self.model_name == 'MobileNetV2':
            base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=self.config['DATA']['DIMENSION'])
        else:
            raise ValueError(f"Model {self.model_name} not supported")
        
        self.base_model = base_model
        return self._add_custom_layers(base_model)

    def _create_vit_model(self):
        activation_func = tfa.activations.gelu
    
        if self.model_name == 'ViT_b16':
            base_model = vit.vit_b16(
                image_size=self.config['DATA']['IMAGE_SIZE'],
                activation=activation_func,  # GELU activation
                pretrained=True,
                include_top=False,
                pretrained_top=False,
                classes=self.config['MODEL']['NUM_CLASSES']
            )
        elif self.model_name == 'ViT_l16':
            base_model = vit.vit_l16(
                image_size=self.config['DATA']['IMAGE_SIZE'],
                activation=activation_func,  # GELU activation
                pretrained=True,
                include_top=False,
                pretrained_top=False,
                classes=self.config['MODEL']['NUM_CLASSES']
            )
        elif self.model_name == 'ViT_b32':
            base_model = vit.vit_b32(
                image_size=self.config['DATA']['IMAGE_SIZE'],
                activation=activation_func,  # GELU activation
                pretrained=True,
                include_top=False,
                pretrained_top=False,
                classes=self.config['MODEL']['NUM_CLASSES']
            )
        else:
            base_model = vit.vit_l32(
                image_size=self.config['DATA']['IMAGE_SIZE'],
                activation=activation_func,  # GELU activation
                pretrained=True,
                include_top=False,
                pretrained_top=False,
                classes=self.config['MODEL']['NUM_CLASSES']
            )
    
        self.base_model = base_model
        return self._add_custom_layers(base_model)

    def _add_custom_layers(self, base_model):
        # vit_models = ['ViT_b16', 'ViT_l16', 'ViT_b32', 'vit_l32']
        activation_func = self.config.get('ACTIVATION', 'relu')  # Default to 'relu'
    
        custom_layers = Flatten()(base_model.output)
        custom_layers = BatchNormalization()(custom_layers)
        
        L2_decay = config['TRAIN']['LR']/10.0
        
        custom_layers = Dense(1024, activation=activation_func,
                        kernel_regularizer=l2(L2_decay)
                        )(custom_layers)
        custom_layers = Dropout(self.config['TRAIN']['DROPOUT'])(custom_layers)
        custom_layers = BatchNormalization()(custom_layers)
    
        custom_layers = Dense(512, activation=activation_func,
                               # kernel_regularizer=l2
                              )(custom_layers)
        custom_layers = Dropout(self.config['TRAIN']['DROPOUT'])(custom_layers)
        custom_layers = BatchNormalization()(custom_layers)
    
        custom_layers = Dense(256, activation=activation_func,
                               # kernel_regularizer=l2
                              # kernel_regularizer=l2(L2_decay)
                              )(custom_layers)
        custom_layers = Dropout(self.config['TRAIN']['DROPOUT'])(custom_layers)
        custom_layers = BatchNormalization()(custom_layers)
    
        custom_layers = Dense(64, activation=activation_func,
                               # kernel_regularizer=l2
                              )(custom_layers)
        custom_layers = Dropout(self.config['TRAIN']['DROPOUT'])(custom_layers)
        custom_layers = BatchNormalization()(custom_layers)
    
        custom_layers = Dense(32, activation=activation_func,
                               # kernel_regularizer=l2
                              )(custom_layers)
        custom_layers = Dropout(self.config['TRAIN']['DROPOUT'])(custom_layers)
        custom_layers = BatchNormalization()(custom_layers)
    
        custom_layers = Dense(16, activation=activation_func,
                               # kernel_regularizer=l2
                              )(custom_layers)
        custom_layers = Dropout(self.config['TRAIN']['DROPOUT'])(custom_layers)
    
        custom_layers = Dense(self.config['MODEL']['NUM_CLASSES'], activation='softmax')(custom_layers)
        
        model = Model(base_model.input, custom_layers, name=self.model_name)
        base_model.trainable = False
        return model

    def _load_convnext_model(self):
        # Mapping ConvNeXt model names to the respective TensorFlow Keras classes
        convnext_mapping = {
            'ConvNeXtTiny': tf.keras.applications.ConvNeXtTiny,
            'ConvNeXtSmall': tf.keras.applications.ConvNeXtSmall,
            'ConvNeXtBase': tf.keras.applications.ConvNeXtBase,
            'ConvNeXtLarge': tf.keras.applications.ConvNeXtLarge,
            'ConvNeXtXLarge': tf.keras.applications.ConvNeXtXLarge,
        }

        if self.model_name in convnext_mapping:
            ConvNeXtClass = convnext_mapping[self.model_name]
            return ConvNeXtClass(
                weights='imagenet',
                include_top=False,
                input_shape=self.config['DATA']['DIMENSION']
            )
        else:
            raise ValueError(f"ConvNeXt model {self.model_name} not supported")

    def _load_efficientnet_model(self):
        efficientnet_mapping = {
            'EfficientNetV2B3': tf.keras.applications.EfficientNetV2B3,
            'EfficientNetV2S': tf.keras.applications.EfficientNetV2S,
            'EfficientNetV2M': tf.keras.applications.EfficientNetV2M,
            'EfficientNetV2L': tf.keras.applications.EfficientNetV2L
        }
        
        if self.model_name in efficientnet_mapping:
            EfficientNetClass = efficientnet_mapping[self.model_name]
            return EfficientNetClass(
                weights='imagenet', 
                include_top=False, 
                input_shape=self.config['DATA']['DIMENSION']
            )
        else:
            raise ValueError(f"EfficientNet model {self.model_name} not supported")

    def get_preprocessing_function(self):
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
    
        if self.model_name in preprocessing_mapping:
            return preprocessing_mapping[self.model_name]
        elif self.model_name.startswith('ViT'):
            return self._preprocess_vit
        else:
            raise ValueError(f"No preprocessing function for model {self.model_name}")

    def _preprocess_vit(self, img):
        try:
            if isinstance(img, Image.Image):
                # img.verify()
                img = np.asarray(img)   
            # Apply ViT-specific preprocessing
            return vit.preprocess_inputs(img)
        except (IOError, UnidentifiedImageError, ValueError) as e:
            # Catch errors related to opening and reading the image
            print(f"Skipping invalid or corrupted image. Error: {e}")
            return None  # Return None for invalid images

class MetricsTracker(tf.keras.callbacks.Callback):
    def __init__(self, csv_file_path):
        super(MetricsTracker, self).__init__()
        self.csv_file_path = csv_file_path  # Path to the CSV file
        self.history = []  # Store metrics in memory as well (optional)

        # Write header to the CSV if it doesn't exist
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w') as f:
                f.write('epoch,loss,val_loss,accuracy,val_accuracy,lr\n')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Get the current learning rate from the optimizer
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            # Evaluate learning rate if it's a schedule (e.g., ExponentialDecay)
            lr_value = tf.keras.backend.get_value(lr(self.model.optimizer.iterations))
        else:
            lr_value = tf.keras.backend.get_value(lr)

        # Collect the metrics for the current epoch
        epoch_metrics = {
            'epoch': epoch + 1,
            'loss': logs.get('loss'),
            'val_loss': logs.get('val_loss'),
            'accuracy': logs.get('accuracy'),
            'val_accuracy': logs.get('val_accuracy'),
            'lr': lr_value
        }

        # Append the current epoch's metrics to the history list (optional)
        self.history.append(epoch_metrics)

        # Write the current epoch's metrics to the CSV file
        with open(self.csv_file_path, 'a') as f:
            f.write(f"{epoch_metrics['epoch']},{epoch_metrics['loss']},{epoch_metrics['val_loss']},"
                    f"{epoch_metrics['accuracy']},{epoch_metrics['val_accuracy']},{epoch_metrics['lr']}\n")

    def get_dataframe(self):
        # Convert the history list to a Pandas DataFrame (optional)
        return pd.DataFrame(self.history)