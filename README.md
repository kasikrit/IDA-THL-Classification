# Classification for Iron Deficiency Anemia (IDA) and Thalassemia (THL)

The code demonstrates the development of the hybrid AI-based patient-level classification framework—integrating soft voting with optimized probability-based thresholds.

**Manuscript Title:** An AI-Based Decision Support Framework for Clinical Screening of Iron Deficiency Anemia and Thalassemia

**Abstract:**

Iron deficiency anemia (IDA) and thalassemia (THL) are common hematological disorders that necessitate efficient and accurate screening for early diagnosis. Traditional blood smear analysis is labor-intensive, prone to subjectivity, and lacks reproducibility, highlighting the urgent need for AI-driven methods to improve diagnostic sensitivity and specificity.
This study proposes a novel hybrid AI framework for patient-level classification, integrating soft voting with optimized probability-based thresholds. The model was trained and validated using a real-world dataset from Hatyai Hospital, Thailand, and evaluated at both the patch and patient levels. The proposed approach achieved 96\% accuracy on the test set. Specifically, it yielded precision-recall values of 1.00 and 0.83 for IDA, and 0.95 and 1.00 for THL, respectively. At the patient level, sensitivity reached 1.00 for THL and 0.83 for IDA. Bayesian probability updates further confirmed prediction reliability, yielding post-test probabilities exceeding 99.99% for IDA and 80% for THL. The model explained 62.84\% of the variance in patient classifications, demonstrating strong discriminatory power.
Model interpretability, assessed using SHAP and Grad-CAM, highlighted key red blood cell morphological features. The proposed framework thus serves as a cost-effective screening tool. 
Limitations include the use of a single-center dataset and the need for adaptive threshold optimization. Future work will focus on multi-center validation and real-world clinical integration. This study thereby establishes a structured baseline for AI-assisted hematology screening, fostering early detection and improved clinical decision-making in resource-limited settings.

## Hardware and Software Specifications

All training, evaluation, and inference were performed on an NVIDIA A100-SXM4-80GB GPU running Ubuntu 20.04.5 LTS (GNU/Linux 5.15.0-50-generic x86_64). Dataset preparation, visualization, and presentation tasks were carried out on macOS 14.7.4 (23H420).

The code and data used in this study were developed using [Python 3.9.16](https://www.python.org/downloads/release/python-3916/) and [TensorFlow 2.11.0](https://www.tensorflow.org/).

For evaluation metrics such as AUROC, calibration plots, logistic regression, isotonic regression, Brier score loss, and R² score, we used [scikit-learn](https://scikit-learn.org/1.3/preface.html) (version 1.3.2), [SciPy](https://docs.scipy.org/doc/scipy-1.11.4/) (version 1.11.4) and [Statsmodels](https://www.statsmodels.org/stable/index.html) (version 0.14.1).

A complete example demonstrating model development and validation is available on Kaggle (Python 3.11.11 and TensorFlow 2.12.0):  
[Kaggle Notebook: Classification for IDA and Thalassemia](https://www.kaggle.com/code/kasikrit/classification-for-ida-and-thalassemia/)




## Creating a Python Environment

We use [Anaconda](https://anaconda.org/). In the Anaconda command prompt, use the following commands:

```bash
conda create -n py39 python=3.9.16
conda activate py39
```

## Installing Required Software Packages

```bash
pip install TensorFlow==2.11.0 scikit-learn==1.3.2 scipy==1.11.4 statsmodels==0.14.1 vit-keras==0.1.2
```

## Available Dataset
The dataset will be publicly availabled at [10.6084/m9.figshare.28779455](https://doi.org/10.6084/m9.figshare.28779455).

## Training Models

### 1. Configure Training Parameters

Modify the parameters in [`configsCellDemo.py`](https://github.com/kasikrit/IDA-THL-Classification/blob/main/configsCellDemo.py) as described below:

```python
# ===== Platform-Specific Path Configuration =====
import platform

# Set base path depending on the OS (macOS or Linux)
if platform.system() == 'Darwin' or platform.system() == 'Linux':
    _C.BASEPATH = '.'  # Use current directory as the base path

# Dataset identifier
_C.DATA.SET = 'IDA-THL-Dataset-Phase-I'  # Name of the dataset used in this phase

# ===== Dataset Configuration =====
_C.DATA.SAVEPATH = os.path.join(_C.BASEPATH, 'models')  # Directory to save trained models
_C.DATA.C = 3  # Number of image channels (e.g., RGB or BGR)
_C.DATA.W = _C.DATA.H = 256  # Image width and height
_C.DATA.IMAGE_SIZE = _C.DATA.W  # Image size used by the model input
_C.DATA.DIMENSION = _C.DATA.W, _C.DATA.H, _C.DATA.C  # Full image dimension tuple
_C.DATA.TEST_SIZE = 0.30  # Proportion of the test data split
_C.DATA.CLASS_LABELS = ['IDA', 'THL']  # Class names
_C.DATA.CLASSES = [0, 1]  # Numerical class labels
_C.DATA.CLASSDICT = [(0, "IDA"), (1, "THL")]  # Mapping between class index and label
_C.DATA.COLOR = 'BGR'  # Image color format ('BGR' or 'RGB')
# _C.DATA.COLOR = 'RGB'  # Uncomment to switch to RGB format

# ===== Model Configuration =====
_C.MODEL = CN()
_C.MODEL.NAMES = [
    'EfficientNetV2S',  # Best-performing model in this study
    # Additional models available but commented out
]
# Optional inference or evaluation paths:
# _C.MODEL.INFER_FILE_PATH = 'infer-ens-20250302-12THL.csv'
# _C.MODEL.EVA_FILE_PATH = 'Eva-ens-20250302'
_C.MODEL.NUM_CLASSES = 2  # Number of output classes (IDA, THL)

# ===== Training Configuration =====
_C.TRAIN = CN()
_C.TRAIN.Enable = True  # Enable or disable training
_C.TRAIN.DATETIME = datetime.now().strftime("%Y%m%d-%H%M")  # Timestamp for the training run
_C.TRAIN.BATCH_SIZE = 32  # Batch size for training
_C.TRAIN.LR = 1e-4  # Initial learning rate
_C.TRAIN.LR_FACTOR = 0.5  # Learning rate reduction factor on plateau
_C.TRAIN.EPOCHS = 120  # Total number of training epochs
_C.TRAIN.DROPOUT = 0.4  # Dropout rate to prevent overfitting
_C.TRAIN.reduceLR_patience = 2  # Patience before reducing LR on plateau
_C.TRAIN.stop_patience = 12  # Patience for early stopping
_C.TRAIN.VERBOSE = 2  # Verbosity level for training logs

# For Vision Transformer (ViT) models
_C.TRAIN.ACTIVATION = 'gelu'  # Activation function used in transformer-based models

_C.DATA.VERIFY = False  # Toggle dataset verification
# _C.DATA.VERIFY = True  # Enable for dataset verification

# ===== Sampling for Quick Testing =====
# _C.TRAIN.SAMPLE = False  # Full dataset training
_C.TRAIN.SAMPLE = True  # Enable sampling for quick debugging
if _C.TRAIN.SAMPLE:
    _C.TRAIN.SAMPLE_SIZE = 12  # Number of samples for test run
    _C.TRAIN.SAMPLE_SIZE_VAL = 0.2  # Validation split for the sample set
    _C.TRAIN.BATCH_SIZE = 1  # Batch size for sample training
    _C.TRAIN.EPOCHS = 2  # Fewer epochs for quicker testing

_C.TRAIN.Evaluate = True  # Evaluate the model after training
_C.SAVE = True  # Enable model checkpoint saving
# _C.SAVE = False  # Disable saving for dry runs
_C.SAVEHIST = True  # Save training history for visualization and analysis
```

### 2. Import Configuration in [train-cell2-demo.py](https://github.com/kasikrit/IDA-THL-Classification/blob/main/train-cell2-demo.py):

```python
import configsCellDemo
config = configsCellDemo.get_config()
```

### 3. Start Model Training
Run the following command to begin training:

```python
python train-cell2-demo.py
``` 

## Patient-Level Classification: Hybrid Threshold Calculation

This section explains how to calculate the optimal probability thresholds for the hybrid patient-level classification approach.

### 1. Configure Model File Paths

Modify the file [`cal-patient-thres-val-public.py`](https://github.com/kasikrit/IDA-THL-Classification/blob/main/cal-patient-thres-val-public.py) to specify the trained model files. Example configuration:

```python
def model_mapping_function(model_name):
    model_root = config['DATA']['SAVEPATH']  # Ensure 'SAVEPATH' is set correctly in your config file
    
    model_mapping = {
        'EfficientNetV2S': [
            os.path.join(model_root, 'Linux-Fold-0-EfficientNetV2S-bal-aug-20250110-1607.hdf5'),
            os.path.join(model_root, 'Linux-Fold-1-EfficientNetV2S-bal-aug-20250110-1607.hdf5'),
            os.path.join(model_root, 'Linux-Fold-2-EfficientNetV2S-bal-aug-20250110-1607.hdf5'),
            os.path.join(model_root, 'Linux-Fold-3-EfficientNetV2S-bal-aug-20250109-1903.hdf5'),
            os.path.join(model_root, 'Linux-Fold-4-EfficientNetV2S-bal-aug-20250110-1607.hdf5'),
        ],
    }
```

## 2. Run Threshold Calculation

Execute the script to calculate the optimal thresholds:

```bash
python cal-patient-thres-val-public.py
```

## 3. Output

The script will output:

- The average **T_upper** values for both:
  - Mean-based threshold
  - Argmax-based threshold
- The average **T_lower** value, calculated as the 10th percentile across all validation folds.

These threshold values are essential for applying the hybrid patient-level classification during the inference stage.

```python
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

```
