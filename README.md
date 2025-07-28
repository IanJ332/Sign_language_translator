# A Comparative Study of Deep Learning Models for Sign Language Sentence Recognition

## 1\. Project Overview

This project presents a systematic evaluation of deep learning architectures for sign language sentence recognition using the large-scale **[How2Sign](https://how2sign.github.io/)** dataset. The primary objective is to investigate and quantify the performance progression from a simple keypoint-based baseline to a sophisticated, hyperparameter-optimized video-based model. All experiments focus exclusively on **frontal-view RGB video clips**.

\<hr\>

## 2\. Methodology

The investigation follows an iterative, three-phase approach to model development, featuring a two-stage training strategy for the advanced video-based models.

### Phase 1: Baseline Model (Keypoint-based LSTM)

A baseline was established using pre-processed 2D pose estimation keypoints to test the efficacy of abstract geometric data.

  * **Architecture**: A stacked LSTM network: `LSTM(64) -> Dropout(0.5) -> LSTM(64) -> Dropout(0.5) -> Dense(32)`.

### Phase 2: Advanced Model (CNN-LSTM)

To leverage richer visual information, an advanced model was built to process raw video frames. This architecture combines a CNN for spatial feature extraction with an LSTM for temporal modeling.

  * **CNN Base**: A pre-trained **MobileNetV2** with frozen weights.
  * **Training Strategy**: A two-stage process was employed:
    1.  **Feature Extraction Training**: Initially, only the LSTM and Dense classification layers were trained. This allows the new layers to learn how to interpret the powerful, general-purpose features from the frozen CNN base.
    2.  **Fine-Tuning**: After initial stable training, the top layers of the MobileNetV2 base were unfrozen and the entire model was trained for a few more epochs with a very low learning rate, allowing the feature extractor to adapt to the specifics of sign language data.
  * **Manual Hyperparameters**: This phase used a manually-tuned configuration of `LSTM(128)` and `Dropout(0.5)`.

### Phase 3: Bayesian Hyperparameter Optimization

To systematically discover an optimal configuration for the CNN-LSTM, Bayesian Optimization was performed using the **Optuna** framework. The search space was defined as:

  * **`learning_rate`**: Log-uniform distribution from `1e-5` to `1e-3`.
  * **`lstm_units`**: Integer from `64` to `256`.
  * **`dropout_rate`**: Uniform distribution from `0.2` to `0.5`.

\<hr\>

## 3\. Data Processing & Feature Engineering

Several key preprocessing decisions were made to ensure model compatibility and computational efficiency.

  * **Sequence Standardization**: All video clips were standardized to a fixed length of **30 frames**. Shorter videos were padded with zero-vectors, and longer videos were truncated. This is a requirement for batch processing in RNNs.
  * **Frame Resolution**: All video frames were downsampled to **64x64 pixels** using OpenCV. This drastically reduces the computational load while retaining essential visual features.
  * **Keypoint Feature Vector**: For the baseline model, a **274-dimensional** feature vector was engineered for each frame by concatenating the (X, Y) coordinates from pose, face, and hand keypoints, discarding confidence scores.
  * **CNN Feature Extraction (Transfer Learning)**: For the advanced models, the pre-trained MobileNetV2 (without its top layer) was used as a frozen feature extractor. This leverages existing knowledge of visual patterns, significantly accelerating training and improving performance.

\<hr\>

## 4\. Technical Environment

  * **Dataset**: How2Sign (official train/validation/test splits).
  * **Hardware**: **NVIDIA A100** GPU.
  * **Frameworks**: Google Colab, TensorFlow, Keras, Optuna.

\<hr\>

## 5\. Project File Structure

The project is organized into the following key notebooks, which correspond to the experimental phases. The `0_` prefixed files are utility and pipeline verification ("smoke test") scripts.

  * `1_1_Train_Baseline_LSTM.ipynb`: Implements and trains the baseline LSTM model.
  * `2_2_Train_Manual_CNN_LSTM.ipynb`: Implements and trains the manually-tuned CNN-LSTM model.
  * `3_1_Optimize_Hyperparameters_Optuna.ipynb`: Executes the Optuna hyperparameter search.
  * `3_2_Train_Final_Optimized_Model.ipynb`: Trains and evaluates the final model using the best-found hyperparameters.