# Song Energy Classifier 🎵⚡  
Classify songs as **High Energy** vs **Low Energy** using Spotify-style audio features and classic machine learning models.

This project explores how features like **loudness**, **acousticness**, **tempo**, **speechiness**, and **valence** relate to perceived song energy, and benchmarks multiple ML classifiers to identify the most effective approach.

---

## 📌 Problem Statement
Music streaming platforms and playlist curators often need to quickly identify songs by their “energy” to match moods, workouts, parties, study sessions, and more. The challenge is that “energy” is subjective, and audio features interact in non-trivial ways.

This project builds and compares multiple ML models to classify songs into:
- **High Energy**
- **Low Energy**

---

## 📂 Dataset
- **Size:** 30,000+ tracks  
- **Source:** Kaggle (Spotify-style track features)  
- **Split:** 80/20 train-test split  

---

## 🎛️ Features Used
We focus on a compact set of impactful audio attributes:

- **Loudness**
- **Acousticness**
- **Tempo**
- **Speechiness**
- **Valence**

### Preprocessing
- Cleaned missing/corrupt entries
- **Min–Max Scaling** applied to normalize features to `[0, 1]`

---

## 🧠 Models Compared
We trained and evaluated the following classifiers:

- Logistic Regression
- Random Forest
- Support Vector Machine (Linear Kernel)
- Gaussian Naive Bayes
- K-Nearest Neighbors (KNN)  
  - Tuned with Grid Search (`k` from 1 to 10)
  - 5-fold cross-validation

> **Energy thresholding:** energy values were binarized using a **0.7 threshold**  
> (`energy > 0.7 → High Energy`, else Low Energy)

---

## ✅ Results Summary
| Model | Accuracy |
|------|----------|
| Logistic Regression | **75.05%** |
| Random Forest | **77.37%** |
| SVM (Linear) | **79.07%** |
| Gaussian Naive Bayes | **74.36%** |
| **KNN (Best)** | **81.10%** |

### Key Takeaways
- **KNN performed best (81.10%)**, benefiting from proximity-based voting where similar songs cluster well.
- **Loudness** consistently emerged as the most influential feature.
- **Acousticness** was also a major signal: lower acousticness often correlated with higher energy.

---

## 📊 Visuals (from report)

### Related-work visuals
**Figure 1 — Genre composition (Top 100 popular songs)**  
![Figure 1](assets/figure_01_genre_composition.png)

**Figure 2 — Danceability by genre**  
![Figure 2](assets/figure_02_danceability_by_genre.png)

**Figure 3 — Energy vs Loudness (Linear fit)**  
![Figure 3](assets/figure_03_energy_vs_loudness.png)

---

### Confusion Matrices + Metrics

#### Logistic Regression
![Figure 4](assets/figure_04_logreg_confusion_matrix.jpg)  
![Figure 5](assets/figure_05_logreg_metrics.jpg)

#### Random Forest
![Figure 6](assets/figure_06_rf_confusion_matrix.jpg)  
![Figure 7](assets/figure_07_rf_metrics.jpg)

#### Support Vector Machine (SVM)
![Figure 8](assets/figure_08_svm_confusion_matrix.jpg)  
![Figure 9](assets/figure_09_svm_metrics.png)

#### Gaussian Naive Bayes
![Figure 10](assets/figure_10_nb_confusion_matrix.jpg)  
![Figure 11](assets/figure_11_nb_metrics.png)

#### K-Nearest Neighbors (Best Model)
![Figure 12](assets/figure_12_knn_confusion_matrix.png)  
![Figure 13](assets/figure_13_knn_metrics.png)

---

## 🚀 How To Run (Typical Setup)

> Update filenames/paths as needed to match your repo.

### 1) Create environment & install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # (Mac/Linux)
# .venv\Scripts\activate    # (Windows)

pip install -r requirements.txt
