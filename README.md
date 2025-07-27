# A Comparative Study of Deep Learning Models for Sign Language Sentence Recognition

## 1. Project Overview and Objectives

This project undertakes a systematic evaluation of different deep learning architectures for the complex task of sign language sentence recognition. The primary objective is to investigate the performance progression from a simple baseline model to a more sophisticated, hyperparameter-optimized model, while detailing the critical data engineering and training strategy decisions made along the way.

All experiments are conducted using the large-scale **[How2Sign](https://how2sign.github.io/)** dataset, with a specific focus on **frontal-view RGB video clips**.

## 2. Methodology and Models

The core of this investigation lies in an iterative, three-phase approach to model development, featuring a two-stage training strategy for the advanced models.

### Phase 1: Baseline Model (Keypoint-based LSTM)
To establish a performance baseline, an initial model was developed using pre-processed 2D pose estimation keypoints. This approach tests the efficacy of using abstract geometric data.
* **Architecture**: A stacked LSTM network consisting of `LSTM(64) -> Dropout(0.5) -> LSTM(64) -> Dropout(0.5) -> Dense(32)`.

### Phase 2: Advanced Model (CNN-LSTM)
To leverage richer visual information, an advanced model was built to process raw video frames directly. This model combines a Convolutional Neural Network (CNN) for spatial feature extraction and a Long Short-Term Memory (LSTM) network for temporal modeling.
* **CNN Base**: A pre-trained **MobileNetV2** is used as the feature extractor.
* **Two-Stage Training Strategy**:
    1.  **Feature Extraction Training**: Initially, the loaded MobileNetV2 base model has all of its layers **frozen** (`trainable = False`). In this stage, only the weights of the newly added LSTM and Dense classification layers are trained. This allows the temporal part of the model to learn how to interpret the powerful, general-purpose features from the CNN without destabilizing the pre-trained weights.
    2.  **Fine-Tuning**: After the model achieves stability (e.g., after 20-30 epochs), the top layers of the MobileNetV2 base are **unfrozen** (made trainable). The entire model is then trained for a few more epochs with a very low learning rate. This allows the pre-trained CNN to slightly adapt its feature extraction to the specific nuances of sign language video data.
* **Manually-Tuned Hyperparameters**: For this phase, the manually-set hyperparameters were `LSTM(128)` and `Dropout(0.5)`.

### Phase 3: Hyperparameter Optimization (Bayesian Optimization)
To systematically find a more optimal configuration for the CNN-LSTM architecture's initial training stage, Bayesian Optimization was performed using the **Optuna** framework. The search space was defined as follows:
* **`learning_rate`**: A log-uniform distribution between `1e-5` and `1e-3`.
* **`lstm_units`**: An integer value between `64` and `256`.
* **`dropout_rate`**: A uniform distribution between `0.2` and `0.5`.

## 3. Key Data Processing and Feature Engineering Decisions

Several critical decisions were made during data preprocessing to ensure model compatibility and computational efficiency.

### 3.1. Sequence Standardization (Fixed Frame Count)
* **Selection**: All video clips were standardized to a fixed length of **`MAX_FRAMES = 30`**.
* **Rationale**: Recurrent Neural Networks like LSTMs require fixed-length input sequences for efficient batch processing. A length of 30 was chosen as a balance to capture the motion of most signs without excessive computational overhead.
* **Calculation**: Videos with **more than 30 frames** were **truncated**. Videos with **fewer than 30 frames** were **padded** with zero-value vectors (black frames).

### 3.2. Frame Resolution (Image Downsampling)
* **Selection**: Each video frame was resized to a resolution of **`IMG_SIZE = 64x64`** pixels.
* **Rationale**: Downsampling from the original high resolution significantly reduces the computational load on the CNN, making training feasible while retaining sufficient visual detail.
* **Calculation**: This was achieved using the `cv2.resize()` function from the OpenCV library.

### 3.3. Feature Vector for LSTM Model (274 Dimensions)
* **Selection**: For the baseline model, a 274-dimensional feature vector was engineered from the 2D keypoint data for each frame.
* **Rationale**: This process converts the structured JSON output from pose estimation into a flat numerical vector that can be directly fed into an LSTM, focusing solely on geometric positions.
* **Calculation**: The vector was created by concatenating the flattened (X, Y) coordinates from four keypoint sources: Pose (50 features), Face (140), Left Hand (42), and Right Hand (42).

## 4. Dataset and Environment

* **Dataset**: How2Sign. The models were trained on the official training split and evaluated on the validation and test splits.
* **Hardware**: All models were trained on an **NVIDIA A100** GPU.
* **Software**: The project was developed in a Google Colab environment using `TensorFlow`, `Keras`, `Optuna`, `Pandas`, and `Scikit-learn`.

## 5. Project File Structure

The project is organized into multiple Jupyter Notebooks, each corresponding to a phase of the research. The `0_` prefixed notebooks are primarily for pipeline verification and "smoke tests".

* `1_1_Train_LSTM.ipynb`: Contains the implementation and training for the Baseline LSTM model.
* `2_2_Train_CNN+LSTM.ipynb`: Contains the implementation and training for the manually-tuned CNN-LSTM model.
* `3_1.ipynb`: Contains the setup and execution of the Optuna hyperparameter search.
* `3_2_Train_final_optimized_model.ipynb`: Contains the final training and evaluation run using the best hyperparameters discovered by Optuna.

***