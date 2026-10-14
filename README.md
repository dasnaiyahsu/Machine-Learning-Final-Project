📊 Machine Learning Final Project — Classification, Clustering & Recommendation System

Python · scikit-learn · XGBoost · Pandas

End-of-semester Machine Learning project covering three independent tasks: supervised classification, unsupervised clustering, and a content-based recommendation system.

Overview

This project was completed as a final exam (UAS) assignment, applying three different machine learning paradigms to three real-world datasets. Each task follows the same workflow: exploratory data analysis and anomaly handling, model building and tuning, and evaluation with a written conclusion.

#	Task	Dataset	Technique
1	Credit score classification	50,000 customer records	Random Forest & XGBoost
2	Regional crime clustering	6,574 incident records (Bangladesh)	K-Means
3	Song recommendation	5,000 tracks (Spotify audio features)	Hybrid content-based filtering
1️⃣ Credit Score Classification

Goal: Predict a customer's credit score category (Poor / Standard / Good) from their financial profile.

Data Cleaning & EDA
Removed stray underscore characters from Age and Annual_Income and converted them to numeric types.
Parsed the text-based Credit_History_Age field (e.g. "11 Years and 3 Months") into a numeric value.
Addressed additional inconsistencies found during EDA before modeling.
Modeling
Trained Random Forest and XGBoost classifiers.
Tuned 3 hyperparameters per model with GridSearchCV (3-fold cross-validation, scored on accuracy):
Random Forest: n_estimators, max_depth, min_samples_split
XGBoost: n_estimators, max_depth, learning_rate
Evaluated both models on an independent test set using accuracy, precision, recall, and F1-score.
Results
Model	Accuracy	Precision	Recall	F1-Score
Random Forest	0.7592	0.7591	0.7592	—
XGBoost ✓	0.7637	0.7633	0.7637	0.7632

✅ Best model: XGBoost, with more consistent performance across all three credit score categories.

Feature importance: Payment_of_Min_Amount (whether a customer pays at least the minimum amount due) was the strongest predictor of credit score.

2️⃣ Bangladesh Regional Crime Clustering

Goal: Segment regions into distinct profiles based on demographic and crime-related characteristics.

Data Cleaning & EDA
Standardized inconsistent text formatting in incident_division (e.g. "Dhaka" / "dhaka" / "dHaka").
Addressed additional data quality issues identified during EDA.
Modeling
Trained K-Means across a range of cluster counts (K = 2–6).
Selected the optimal K using two metrics: the Elbow Method (inertia) and Silhouette Score.
Visualized the resulting clusters.
Results

Two distinct regional profiles emerged:

Cluster	Profile	Characteristics
0	Urban Metropolitan	Dominated by Dhaka; high population density (~1.45M avg.); higher literacy rate (70.1%)
1	Rural Developing	Lower density; lower access to social infrastructure
3️⃣ Hybrid Song Recommendation System

Goal: Build a function that takes free-text input and returns 5 relevant song recommendations.

Data Cleaning & EDA
Removed 9 duplicate rows and 1 row with a missing track name.
Handled missing values across Artist, Genres, Tempo, and Acousticness.
Identified and corrected a physical anomaly in the audio feature data.
How it works

The system automatically switches between two modes depending on the input:

Rekomendasi (Recommend) mode — triggered when the input exactly matches a track already in the dataset. Returns similar songs using a hybrid similarity score (audio features + text metadata).
Pencarian (Search) mode — triggered for any other text input (an artist name, a genre, or a free-text description). Uses TF-IDF + cosine similarity over track/artist/genre text to find the closest matches.
Validation

Tested against 5 different input types to confirm the system generalizes beyond exact song titles:

Input	Mode triggered	Result
"If I Ain't Got You"	Rekomendasi	Returned Alicia Keys tracks with similar audio characteristics
"Taylor Swift"	Pencarian	Returned the artist's discography
"Disco"	Pencarian	Returned classic disco tracks and related artists
Instrument description	Pencarian	Matched songs by instrumentation
Tech Stack
Layer	Technology
Data processing	Pandas, NumPy
Classification	scikit-learn (Random Forest), XGBoost
Clustering	scikit-learn (K-Means, PCA, silhouette/Davies–Bouldin metrics)
Recommendation	scikit-learn (TF-IDF, cosine similarity)
Visualization	Matplotlib, Seaborn
Project Structure
UAS_PROJECT_ML/
├── Nomor 1/
│   ├── nomor 1.ipynb                    # Credit score classification
│   ├── Credis_Score_Dataset_A.csv
│   └── video presentation number 1.mp4
├── Nomor 2/
│   ├── Nomor 2.ipynb                    # Crime clustering
│   ├── Bangladesh_Crime_Dataset_A.csv
│   └── video presentation number 2.mp4
├── Nomor 3/
│   ├── nomor 3.ipynb                    # Song recommendation
│   ├── song_recomendation_A.csv
│   └── video presentation number 3.mp4
└── README.md
How to Run
bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

Each task is self-contained in its own folder — open the notebook in Jupyter or Google Colab and run the cells in order, with the corresponding CSV file in the same directory.

Author

Dasnaiya Hsu · Binus University 📧 dasnaiyaiirene@gmail.com

Made for the Machine Learning Final Project (UAS) — Binus University 2026
