# Music Recommendation System

Recommends songs similar to an input track using Spotify audio features and K-Nearest Neighbors.

## How it works
1. Load the Spotify tracks dataset (`data/dataset.csv`) and normalize audio features such as tempo (BPM), valence, energy, and danceability.
2. Train KNN classifiers to predict a song's genre and mood.
3. Classify the input song, filter the catalog to matching genre and mood, and rank candidates by Euclidean or cosine similarity.

## Run
```
pip install pandas numpy scikit-learn matplotlib seaborn scipy
python genre_mood_classifier.py
```

Built with Python, pandas, scikit-learn, and Matplotlib.
