# Classification for Iron Deficiency Anemia (IDA) and Thalassemia (THL)

The code demonstrates the development of the hybrid AI-based patient-level classification framework—integrating soft voting with optimized probability-based thresholds.

**Manuscript Title:** An AI-Based Decision Support Framework for Clinical Screening of Iron Deficiency Anemia and Thalassemia

**Abstract:**

Iron Deficiency Anemia (IDA) and Thalassemia (THL) are common hematologic disorders requiring efficient and accurate screening for early diagnosis. Traditional blood smear analysis is labor-intensive and subjective, underscoring the need for AI-driven solutions to enhance Sensitivity and Specificity.
This study introduces a novel hybrid AI-based patient-level classification framework integrating soft voting with optimized probability-based thresholds. The model was trained and validated using a real-world dataset from Hatyai Hospital, Thailand, and evaluated at both the patch and patient levels. The proposed approach achieved 96\% accuracy on the test set, with precision-recall values of 1.00 and 0.83 for IDA and 0.95 and 1.00 for THL, respectively. THL sensitivity reached 1.00 at the patient level, while IDA sensitivity was 0.83. Bayesian probability updates confirmed prediction reliability, yielding post-test probabilities exceeding 99.99\% for IDA and 80\% for THL. The model explained 62.84\% of the variance in patient classifications, demonstrating strong discriminatory power.

Model interpretability was assessed using SHAP and Grad-CAM, which highlighted key red blood cell morphological features. The proposed framework serves as a cost-effective and clinically interpretable AI-assisted hematology screening tool, supporting decision-making in resource-limited settings.

Limitations include the use of a single-center dataset and the need for adaptive threshold optimization. Future work will focus on multi-center validation and real-world clinical integration. This study establishes a structured baseline for AI-assisted hematology screening, supporting early detection and improved clinical decision-making.

# Hardware and Software Specifications

All training, evaluation, and inference were performed on an NVIDIA A100-SXM4-80GB GPU running Ubuntu 20.04.5 LTS (GNU/Linux 5.15.0-50-generic x86_64). Dataset preparation, visualization, and presentation tasks were carried out on macOS 14.7.4 (23H420).

The code and data used in this study were developed using [Python 3.9.16](https://www.python.org/downloads/release/python-3916/) and [TensorFlow 2.11.0](https://www.tensorflow.org/).

For evaluation metrics such as AUROC, calibration plots, logistic regression, isotonic regression, Brier score loss, and R² score, we used [scikit-learn](https://scikit-learn.org/1.3/preface.html) (version 1.3.2) and [SciPy](https://docs.scipy.org/doc/scipy-1.11.4/) (version 1.11.4).

A complete example demonstrating model development and validation is available on Kaggle (Python 3.11.11 and TensorFlow 2.12.0):  
[Kaggle Notebook: Train and Validate the Proposed Model](https://www.kaggle.com/code/kasikrit/ida-thl-classification-phase-i)


# Creating a Python Environment

We use [Anaconda](https://anaconda.org/). In the Anaconda command prompt, use the following commands:

```bash
conda create -n py39 python=3.9.16
conda activate py39
```

## Installing Required Software Packages

```bash
pip install TensorFlow==2.11.0 scikit-learn==1.3.2 scipy==1.11.4 vit-keras==0.1.2
```

# Training Models

1. Set up parameters in configCell.py as descripbed below:

```python
# ===== Platform-Specific Path Configuration =====
import platform

# Set base path depending on the OS (macOS or Linux)
if platform.system() == 'Darwin' or platform.system() == 'Linux':
    _C.BASEPATH = '.'  # Use current directory as base path

# Dataset identifier
_C.DATA.SET = 'IDA-THL-Dataset-Phase-I'  # Name of the dataset used in this phase

# ===== Dataset Configuration =====
_C.DATA.SAVEPATH = os.path.join(_C.BASEPATH, 'models')  # Directory to save trained models
_C.DATA.C = 3  # Number of image channels (e.g., RGB or BGR)
_C.DATA.W = _C.DATA.H = 256  # Image width and height
_C.DATA.IMAGE_SIZE = _C.DATA.W  # Image size used by model input
_C.DATA.DIMENSION = _C.DATA.W, _C.DATA.H, _C.DATA.C  # Full image dimension tuple
_C.DATA.TEST_SIZE = 0.30  # Proportion of test data split
_C.DATA.CLASS_LABELS = ['IDA', 'THL']  # Class names
_C.DATA.CLASSES = [0, 1]  # Numerical class labels
_C.DATA.CLASSDICT = [(0, "IDA"), (1, "THL")]  # Dictionary mapping for class index and label
_C.DATA.COLOR = 'BGR'  # Image color format (can be changed to 'RGB' if needed)
# _C.DATA.COLOR = 'RGB'  # Alternative color format

# ===== Model Configuration =====
_C.MODEL = CN()
_C.MODEL.NAMES = [
    'EfficientNetV2S',  # Primary model used; other architectures commented for flexibility
]
# _C.MODEL.INFER_FILE_PATH = 'infer-ens-20250302-12THL.csv'  # Optional inference CSV path
# _C.MODEL.EVA_FILE_PATH = 'Eva-ens-20250302'  # Optional evaluation path
_C.MODEL.NUM_CLASSES = 2  # Number of output classes (IDA, THL)

# ===== Training Configuration =====
_C.TRAIN = CN()
_C.TRAIN.Enable = True  # Toggle to enable/disable training
_C.TRAIN.DATETIME = datetime.now().strftime("%Y%m%d-%H%M")  # Timestamp for training run
_C.TRAIN.BATCH_SIZE = 32  # Batch size for training
_C.TRAIN.LR = 1e-4  # Initial learning rate
_C.TRAIN.LR_FACTOR = 0.5  # Learning rate reduction factor on plateau
_C.TRAIN.EPOCHS = 120  # Total number of training epochs
_C.TRAIN.DROPOUT = 0.4  # Dropout rate to prevent overfitting
_C.TRAIN.reduceLR_patience = 2  # Epochs with no improvement before reducing LR
_C.TRAIN.stop_patience = 12  # Epochs with no improvement before early stopping
_C.TRAIN.VERBOSE = 2  # Verbosity level for training logs

# For ViT (Vision Transformer) architectures
_C.TRAIN.ACTIVATION = 'gelu'  # Activation function used in the transformer model

_C.DATA.VERIFY = False  # Toggle to verify dataset integrity
# _C.DATA.VERIFY = True  # Enable for dataset verification

# ===== Sampling for Quick Test Runs =====
# _C.TRAIN.SAMPLE = False  # Disable sampling for full dataset training
_C.TRAIN.SAMPLE = True  # Enable sampling for development/debugging
if _C.TRAIN.SAMPLE:
    _C.TRAIN.SAMPLE_SIZE = 12  # Number of samples for test run
    _C.TRAIN.SAMPLE_SIZE_VAL = 0.2  # Validation split for sample set
    _C.TRAIN.BATCH_SIZE = 1  # Batch size for sample training
    _C.TRAIN.EPOCHS = 2  # Fewer epochs for quicker testing

_C.TRAIN.Evaluate = True  # Toggle to evaluate model after training
_C.SAVE = True  # Enable model checkpoint saving
# _C.SAVE = False  # Disable saving (useful for dry runs)
_C.SAVEHIST = True  # Save training history for plotting or further analysis
```
Usage:

```python
import configsCellDemo
config = configsCellDemo.get_config()
```

2. Train models
```python
python train-cell2-demo.py
``` 

