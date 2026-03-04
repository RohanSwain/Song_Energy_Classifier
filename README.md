# Song Energy Classifier 🎵⚡

A machine learning project that classifies songs into **High Energy** or **Low Energy** categories using audio features derived from Spotify-style datasets.

This project explores how musical characteristics such as **loudness, acousticness, tempo, speechiness, and valence** influence the perceived energy of a song. Multiple machine learning models were trained and compared to determine the most effective approach for energy classification.

---

# Project Overview

Music streaming services often categorize songs based on mood, activity, or listening context. One of the most important attributes used in playlist generation is **song energy**.

Energy levels can influence:

- Workout playlists
- Party or dance playlists
- Study or relaxation playlists
- Music recommendation systems

However, determining energy levels manually is inefficient for large music libraries. This project builds a **machine learning pipeline that automatically predicts whether a song is high-energy or low-energy based on its audio features.**

---

# Dataset

The dataset contains **30,000+ songs** with multiple audio features commonly used in music analytics.

### Source
Spotify-style dataset from Kaggle.

### Data Split

- Training Data: **80%**
- Testing Data: **20%**

---

# Features Used

The model focuses on five primary audio attributes:

| Feature | Description |
|------|------|
| Loudness | Overall volume level of the track |
| Acousticness | Probability that the track is acoustic |
| Tempo | Beats per minute of the song |
| Speechiness | Amount of spoken words present |
| Valence | Musical positivity or emotional tone |

---

# Data Preprocessing

Several preprocessing steps were applied before training the models:

1. Data cleaning and validation
2. Handling missing values
3. Feature normalization using **Min-Max Scaling**
4. Feature selection to retain the most predictive attributes

Normalization ensures that all features are scaled between **0 and 1**, preventing any single feature from dominating the model.

---

# Target Variable

Songs were categorized using an **energy threshold of 0.7**.

Energy > 0.7 → High Energy  
Energy ≤ 0.7 → Low Energy

This converts the problem into a **binary classification task**.

---

# Machine Learning Models

The following classification algorithms were implemented and evaluated:

### Logistic Regression
A baseline linear model used for binary classification.

### Random Forest
An ensemble learning method that builds multiple decision trees and aggregates their predictions.

### Support Vector Machine (SVM)
A powerful classifier that finds the optimal boundary between classes.

### Gaussian Naive Bayes
A probabilistic classifier based on Bayes’ theorem assuming feature independence.

### K-Nearest Neighbors (KNN)
A distance-based algorithm that classifies songs based on the labels of their nearest neighbors.

---

# Hyperparameter Tuning

For the **KNN model**, hyperparameter tuning was performed using:

- Grid Search
- 5-fold Cross Validation
- K values tested from **1 to 10**

This helped identify the optimal number of neighbors for classification.

---

# Model Performance

| Model | Accuracy |
|------|------|
| Logistic Regression | 75.05% |
| Random Forest | 77.37% |
| Support Vector Machine | 79.07% |
| Gaussian Naive Bayes | 74.36% |
| **K-Nearest Neighbors (Best Model)** | **81.10%** |

---

# Key Insights

Several interesting observations emerged from the analysis:

- **Loudness** was the most influential feature affecting energy classification.
- Songs with **lower acousticness** often corresponded to higher energy levels.
- **KNN performed best**, likely because songs with similar audio characteristics naturally cluster together.
- Ensemble models like Random Forest also performed well but slightly underperformed compared to KNN.

---

# Applications

This model can be useful in multiple music technology applications:

- Automated playlist generation
- Music recommendation systems
- Audio content tagging
- Music discovery platforms
- Mood-based music classification

---

# Project Structure

song-energy-classifier

data  
└── dataset.csv  

notebooks  
└── exploratory_analysis.ipynb  

src  
├── train_model.py  
├── evaluate_model.py  
└── preprocessing.py  

requirements.txt  
README.md  

---

# Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Future Improvements

Several improvements could enhance the system:

- Incorporating **deep learning models**
- Expanding feature engineering
- Using **larger and more diverse datasets**
- Building a **real-time song energy prediction API**
- Developing a **web interface for song classification**

---

# Authors

**Rohan Swain**  
University of Delaware  

**Divyam Patel**  
University of Delaware  

**Vineet Kotian**  
University of Delaware  

---

# References

- Spotify audio feature dataset (Kaggle)
- Research on music energy prediction
- Machine learning classification literature
