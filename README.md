# Multimodal Gait Recognition

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![TensorFlow 2.x](https://img.shields.io/badge/tensorflow-2.x-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This project develops and evaluates a suite of deep learning models for human gait classification using multimodal sensor data. The primary goal is to accurately identify six different gait patterns from 3D skeleton joint positions and plantar foot pressure data.

The analysis culminates in a top-performing multimodal model that directly fuses preprocessed skeleton data with pressure data, achieving a **test accuracy of 93.8%**.

## ✨ Key Features

* **Multimodal Data Fusion**: Combines kinematic data (skeleton joints) with kinetic data (foot pressure) to create a robust classification system.
* **Feature Strategy Comparison**: Systematically compares the effectiveness of traditional preprocessing (variance-based feature selection and scaling) against unsupervised feature learning with a 1D CNN Autoencoder.
* **Comprehensive Model Comparison**: Trains and evaluates over 10 different deep learning architectures, including various LSTMs, GRUs, Attention models, and hybrid CNN-RNNs.
* **Rigorous Validation**: Implements **K-Fold** and **Leave-One-Subject-Out (LOSO)** cross-validation to ensure the best model is robust and generalizes well to unseen individuals.
* **Interactive Demo**: Includes a walkthrough section to visualize model predictions on individual samples, complete with feature heatmaps and confidence scores.

## 📊 Results

The multimodal model using direct preprocessed features (`gru_cnn_fusion`) significantly outperformed all other architectures. The experiment showed that, in this case, unsupervised feature learning with the autoencoder resulted in a loss of critical information, leading to lower performance.

| Model | Test Accuracy |
| :--- | :--- |
| **gru_cnn_fusion** | **93.75%** |
| cnn_lstm | 89.93% |
| attention | 89.93% |
| gru | 89.24% |
| dual_lstm | 88.89% |



### Confusion Matrix for the Best Model
The confusion matrix below shows the per-class accuracy of the `gru_cnn_fusion` model.



## ⚙️ Project Pipeline

The project follows a systematic pipeline from raw data to final evaluation:

1.  **Data Ingestion & Preprocessing**: Loads data from `skeleton.csv` and `pressure.csv` files for all subjects and trials. Handles missing or infinite values.
2.  **Feature Engineering & Selection**:
    * Calculates joint velocities from skeleton positions.
    * Applies a variance-based feature selection to reduce dimensionality and keep the most informative features.
    * Normalizes features using `StandardScaler` for skeleton/velocity data and `MinMaxScaler` for pressure data.
3.  **Model Training**: Trains over 10 different models sequentially, tracking performance metrics like accuracy, loss, training time, and memory usage.
4.  **Evaluation**: The top models are further validated using 5-Fold and Leave-One-Subject-Out (LOSO) cross-validation to confirm their robustness.

## 🚀 How to Run

1.  **Prerequisites**
    * Ensure you have Python 3.10+ installed.
    * Install the required libraries:
        ```bash
        pip install pandas numpy scikit-learn tensorflow seaborn matplotlib psutil
        ```

2.  **Dataset**
    * Download the dataset from its source.
    * Place the ZIP file in a directory accessible by the notebook.
    * Update the `zip_path` variable in the second code cell.

3.  **Execution**
    * Open `gait_recognition.ipynb` in a Jupyter environment.
    * Run the cells sequentially from top to bottom. The notebook will automatically extract the data, preprocess it, train all models, and display the results.

## 📄 Dataset Citation

The dataset used in this project was created by Jun et al. and is publicly available. If you use this dataset in your research, please cite the original paper:

> K. Jun, S. Lee, D. -W. Lee and M. S. Kim, "Deep learning-based multimodal abnormal gait classification using a 3D skeleton and plantar foot pressure," *IEEE Access*, doi: 10.1109/ACCESS.2021.3131613.

## 📜 License

This project is licensed under the MIT License.
