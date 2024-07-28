# Audio & Speech Emotion Recognition

## Overview
Identifying emotion from speech is a non-trivial task pertaining to the ambiguous definition of emotion itself. In this work, we build light-weight multimodal machine learning models and compare it against the heavier and less interpretable deep learning counterparts. For both types of models, we use hand-crafted features from a given audio signal. Our experiments show that the light-weight models are comparable to the deep learning baselines and even outperform them in some cases, achieving state-of-the-art performance on the IEMOCAP dataset.

The hand-crafted feature vectors obtained are used to train two types of models:

1. ML-based: Logistic Regression, SVMs, Random Forest, eXtreme Gradient Boosting and Multinomial Naive-Bayes.
2. DL-based: Multi-Layer Perceptron, LSTM Classifier

## Datasets
The [IEMOCAP](https://link.springer.com/content/pdf/10.1007%2Fs10579-008-9076-6.pdf) dataset was used for all the experiments in this work. 
## Requirements
All the experiments have been tested using the following libraries:
- xgboost==0.82
- torch==1.0.1.post2
- scikit-learn==0.20.3
- numpy==1.16.2
- jupyter==1.0.0
- pandas==0.24.1
- librosa==0.7.0

1. Start a jupyter notebook by running `jupyter notebook` from the root of this project.
2. Run `1_extract_emotion_labels.ipynb` to extract labels from transriptions and compile other required data into a csv.
3. Run `2_build_audio_vectors.ipynb` to build vectors from the original wav files and save into a pickle file
4. Run `3_extract_audio_features.ipynb` to extract 8-dimensional audio feature vectors for the audio vectors
5. Run `4_prepare_data.ipynb` to preprocess and prepare audio + video data for experiments
6. It is recommended to train `LSTMClassifier` before running any other experiments for easy comparsion with other models later on:
  - Change `config.py` for any of the experiment settings. For instance, if you want to train a speech2emotion classifier, make necessary changes to `lstm_classifier/s2e/config.py`. Similar procedure follows for training text2emotion (`t2e`) and text+speech2emotion (`combined`) classifiers.
  - Run `python lstm_classifier.py` from `lstm_classifier/{exp_mode}` to train an LSTM classifier for the respective experiment mode (possible values of `exp_mode: s2e/t2e/combined`)
7. Run `5_audio_classification.ipynb` to train ML classifiers for audio
8. Run `5.1_sentence_classification.ipynb` to train ML classifiers for text
9. Run `5.2_combined_classification.ipynb` to train ML classifiers for audio+text

## Results
Accuracy, F-score, Precision and Recall has been reported for the different experiments.
