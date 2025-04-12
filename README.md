# IDA-THL-Classification
The code demonstrates the development of the hybrid AI-based patient-level classification framework—integrating soft voting with optimized probability-based thresholds.

**Manuscript Title:** An AI-Based Decision Support Framework for Clinical Screening of Iron Deficiency Anemia and Thalassemia

**Abstact:**

Iron Deficiency Anemia (IDA) and Thalassemia (THL) are common hematologic disorders requiring efficient and accurate screening for early diagnosis. Traditional blood smear analysis is labor-intensive and subjective, underscoring the need for AI-driven solutions to enhance Sensitivity and Specificity.
This study introduces a novel hybrid AI-based patient-level classification framework integrating soft voting with optimized probability-based thresholds. The model was trained and validated using a real-world dataset from Hatyai Hospital, Thailand, and evaluated at both the patch and patient levels. The proposed approach achieved 96\% accuracy on the test set, with precision-recall values of 1.00 and 0.83 for IDA and 0.95 and 1.00 for THL, respectively. THL sensitivity reached 1.00 at the patient level, while IDA sensitivity was 0.83. Bayesian probability updates confirmed prediction reliability, yielding post-test probabilities exceeding 99.99\% for IDA and 80\% for THL. The model explained 62.84\% of the variance in patient classifications, demonstrating strong discriminatory power.

Model interpretability was assessed using SHAP and Grad-CAM, which highlighted key red blood cell morphological features. The proposed framework serves as a cost-effective and clinically interpretable AI-assisted hematology screening tool, supporting decision-making in resource-limited settings.

Limitations include the use of a single-center dataset and the need for adaptive threshold optimization. Future work will focus on multi-center validation and real-world clinical integration. This study establishes a structured baseline for AI-assisted hematology screening, supporting early detection and improved clinical decision-making.

# Hardware and Software Specifications

All training, evaluation, and inference were performed on an NVIDIA A100-SXM4-80GB GPU running Ubuntu 20.04.5 LTS (GNU/Linux 5.15.0-50-generic x86_64). Dataset preparation, visualization, and presentation tasks were carried out on macOS 14.7.4 (23H420).

The code and data used in this study were developed using [Python 3.9.16](https://www.python.org/downloads/release/python-3916/) and [TensorFlow 2.11.0](https://www.tensorflow.org/).

For evaluation metrics such as AUROC, calibration plots, logistic regression, isotonic regression, Brier score loss, and R² score, we used [scikit-learn](https://scikit-learn.org/1.3/preface.html) (version 1.3.2) and [SciPy](https://docs.scipy.org/doc/scipy-1.11.4/) (version 1.11.4).

A complete example demonstrating model development and validation is available on Kaggle:  
[Kaggle Notebook: Train and Validate the Proposed Model](https://www.kaggle.com/code/kasikrit/pm-analysis-train-and-validate)


## Creating a Python Environment

We use [Anaconda](https://anaconda.org/). In the Anaconda command prompt, use the following commands:

```bash
conda create -n py39 python=3.9.16
conda activate py39
