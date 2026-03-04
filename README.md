# Spotify Song Energy Classifier 🎵⚡

This project builds a machine learning system that classifies songs into **High Energy** or **Low Energy** categories using Spotify audio features.

The goal of this project is to analyze musical attributes such as **loudness, tempo, speechiness, acousticness, and valence** and determine whether a song belongs to a high-energy category.

Multiple machine learning models were implemented using **Python and R** and compared to evaluate their performance in predicting song energy levels.

---

# Project Objective

Music streaming platforms organize songs into playlists based on mood and intensity (workout, party, study, relaxation, etc.).

One important attribute used in music recommendation systems is **song energy**, which measures the intensity and activity level of a track.

This project develops machine learning models that automatically classify songs based on their **audio characteristics**.

---

# Dataset

The dataset used is **spotify_songs.csv**, which contains thousands of songs with audio features extracted from Spotify.

These features describe the musical characteristics of each track and are used to train classification models.

---

# Features Used

The following audio features were selected for modeling:

| Feature | Description |
|------|------|
| Loudness | Overall volume level of the track |
| Speechiness | Presence of spoken words in a song |
| Tempo | Speed of the song (beats per minute) |
| Valence | Musical positivity or emotional tone |
| Acousticness | Likelihood that the track is acoustic |

Target Variable:

| Variable | Description |
|------|------|
| Energy | Intensity and activity level of a song |

The energy variable was converted into a **binary classification problem**:

```
Energy < 0.65 → Low Energy (0)
Energy ≥ 0.65 → High Energy (1)
```

---

# Data Preprocessing

The following preprocessing steps were applied before model training:

### Feature Selection
Only relevant audio attributes were selected.

### Label Conversion
Energy values were converted into binary classes.

### Min-Max Normalization

All features were normalized using Min-Max scaling:

```
(x - min(x)) / (max(x) - min(x))
```

This ensures that all features are scaled between **0 and 1**.

### Train-Test Split

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

---

# Machine Learning Models

The project compares several classification algorithms.

---

## Logistic Regression

Implemented in **Python using Scikit-Learn**.

Logistic Regression estimates the probability that a song belongs to a particular class.

Evaluation metrics include:

- Accuracy
- Confusion Matrix
- ROC Curve
- AUC Score

---

## Random Forest

Implemented in **Python**.

Random Forest is an ensemble algorithm that constructs multiple decision trees and aggregates their predictions.

Advantages:

- Handles nonlinear relationships
- Reduces overfitting
- Works well with structured data

---

## Support Vector Machine (SVM)

Implemented in **R using the e1071 library**.

SVM identifies the optimal hyperplane that separates two classes with the maximum margin.

---

## Naive Bayes

Implemented in **R using the naivebayes package**.

Naive Bayes is a probabilistic classifier based on Bayes’ theorem with the assumption of feature independence.

---

## K-Nearest Neighbors (KNN)

Implemented in **R using the class package**.

KNN classifies songs based on the majority class among the nearest neighbors.

Steps:

1. Calculate distance between samples
2. Identify the K nearest neighbors
3. Assign the majority class

---

# Model Evaluation

The models were evaluated using:

- Accuracy
- Confusion Matrix
- Classification metrics
- ROC Curve (Logistic Regression)

These metrics help determine which algorithm performs best for song energy classification.

---

# Project Structure

```
Song-Energy-Classifier
│
├── LR_RF.ipynb
│   Logistic Regression and Random Forest implementation (Python)
│
├── Project SVM_NB.R
│   Support Vector Machine and Naive Bayes implementation (R)
│
├── Song_Energy_Classifier.R
│   Data preprocessing and K-Nearest Neighbors model (R)
│
├── Song_Energy_Classifer_Report.pdf
│   Project report explaining methodology, analysis, and results
│
├── spotify_songs.csv
│   Dataset used in the project
│
└── README.md
```

---

# Technologies Used

### Python

- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn

### R

- caret
- class
- e1071
- naivebayes
- ggplot2

---

# How to Run the Project

### Python Models

Run the Jupyter notebook:

```
LR_RF.ipynb
```

This notebook trains:

- Logistic Regression
- Random Forest

---

### R Models

Run the following scripts:

```
Song_Energy_Classifier.R
Project SVM_NB.R
```

These scripts train:

- KNN
- SVM
- Naive Bayes

---

# Applications

This project can be used in:

- Music recommendation systems
- Automated playlist generation
- Mood-based music classification
- Audio feature analysis
- Music streaming platforms

---

# Authors

Rohan Swain  
University of Delaware

Divyam Patel  
University of Delaware

Vineet Kotian  
University of Delaware

---

# Future Improvements

Potential improvements include:

- Testing advanced models such as XGBoost or Neural Networks
- Hyperparameter tuning for improved accuracy
- Feature importance analysis
- Building a web application for energy prediction
- Integrating with the Spotify API for real-time predictions
