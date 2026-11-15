import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class MusicRecommendationSystem:
    def __init__(self):
        self.df = None
        self.genre_classifier = None
        self.mood_classifier = None
        self.scaler = StandardScaler()
        self.audio_features = ['tempo', 'valence', 'energy', 'danceability', 
                              'acousticness', 'instrumentalness', 'liveness', 'speechiness']
        
    def load_data(self, file_path=None):
        """Step 0: Load and prepare the dataset"""
        if file_path:
            try:
                # Load Spotify dataset
                print("Loading dataset...")
                self.df = pd.read_csv(file_path, na_values=['NA', 'NaN', 'null', ''])
                print(f"Initial dataset shape: {self.df.shape}")
                
                # Check for missing values
                print("\nChecking for missing values in audio features:")
                missing_values = self.df[self.audio_features].isnull().sum()
                print(missing_values)
                
                # Rename columns to match expected format
                self.df = self.df.rename(columns={
                    'track_name': 'track_name',
                    'artists': 'artist_name',
                    'track_genre': 'genre'
                })
                
                # Clean the data
                print("\nCleaning data...")
                
                # 1. Handle missing values in text columns
                text_columns = ['track_name', 'artist_name', 'genre']
                for col in text_columns:
                    if col in self.df.columns:
                        # Replace NaN with empty string for text columns
                        self.df[col] = self.df[col].fillna('')
                        # Remove any remaining whitespace
                        self.df[col] = self.df[col].str.strip()
                
                # 2. Handle missing values in audio features
                for feature in self.audio_features:
                    if feature in self.df.columns:
                        # Calculate mean for each genre
                        genre_means = self.df.groupby('genre')[feature].transform('mean')
                        # Fill NaN with genre-specific mean, fallback to overall mean
                        self.df[feature] = self.df[feature].fillna(genre_means)
                        self.df[feature] = self.df[feature].fillna(self.df[feature].mean())
                
                # 3. Ensure all audio features are within valid ranges (0-1)
                for feature in ['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']:
                    if feature in self.df.columns:
                        self.df[feature] = self.df[feature].clip(0, 1)
                
                # Create mood labels based on audio features
                print("\nCreating mood labels...")
                self.df['mood'] = self.df.apply(self._assign_mood, axis=1)
                
                # Print genre distribution
                print("\nGenre distribution in dataset:")
                print(self.df['genre'].value_counts().head(10))
                print(f"\nTotal number of unique genres: {self.df['genre'].nunique()}")
                
                # Final check for any remaining NaN values
                remaining_nulls = self.df[self.audio_features].isnull().sum()
                if remaining_nulls.any():
                    print("\nWARNING: There are still NaN values in the dataset!")
                    print(remaining_nulls)
                    # Drop any remaining rows with NaN values
                    initial_rows = len(self.df)
                    self.df = self.df.dropna(subset=self.audio_features)
                    print(f"Dropped {initial_rows - len(self.df)} rows with remaining NaN values")
                else:
                    print("\nNo NaN values remaining in audio features")
                
            except Exception as e:
                print(f"\nError during data loading: {str(e)}")
                raise
        
        else:
            # Create sample dataset for demonstration
            print("Creating sample dataset for demonstration...")
            self.df = self._create_sample_dataset()
        
        print(f"\nFinal dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        return self.df
    
    def _create_sample_dataset(self):
        """Create a sample dataset with realistic audio features"""
        np.random.seed(42)
        n_songs = 1000
        
        # Generate realistic audio features
        data = {
            'track_name': [f'Song_{i}' for i in range(n_songs)],
            'artist_name': [f'Artist_{i//10}' for i in range(n_songs)],
            'tempo': np.random.normal(120, 30, n_songs),
            'valence': np.random.beta(2, 2, n_songs),
            'energy': np.random.beta(2, 2, n_songs),
            'danceability': np.random.beta(2, 2, n_songs),
            'acousticness': np.random.beta(1, 3, n_songs),
            'instrumentalness': np.random.beta(1, 5, n_songs),
            'liveness': np.random.beta(1, 4, n_songs),
            'speechiness': np.random.beta(1, 4, n_songs),
            'popularity': np.random.randint(0, 100, n_songs)
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic genre and mood labels based on audio features
        df['genre'] = df.apply(self._assign_genre, axis=1)
        df['mood'] = df.apply(self._assign_mood, axis=1)
        
        return df
    
    def _assign_genre(self, row):
        """Assign genre based on audio features"""
        if row['acousticness'] > 0.7:
            return 'acoustic'
        elif row['energy'] > 0.8 and row['danceability'] > 0.7:
            return 'electronic'
        elif row['energy'] > 0.7 and row['acousticness'] < 0.3:
            return 'rock'
        elif row['valence'] > 0.7 and row['danceability'] > 0.6:
            return 'pop'
        elif row['instrumentalness'] > 0.5:
            return 'instrumental'
        else:
            return 'alternative'
    
    def _assign_mood(self, row):
        """Assign mood based on audio features"""
        if row['valence'] > 0.7 and row['energy'] > 0.6:
            return 'happy'
        elif row['valence'] < 0.3 and row['energy'] < 0.4:
            return 'sad'
        elif row['energy'] > 0.8:
            return 'energetic'
        elif row['valence'] > 0.5 and row['energy'] < 0.5:
            return 'peaceful'
        else:
            return 'neutral'
    
    def explore_data(self):
        """Exploratory Data Analysis"""
        print("=== EXPLORATORY DATA ANALYSIS ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nGenre distribution:")
        print(self.df['genre'].value_counts())
        print(f"\nMood distribution:")
        print(self.df['mood'].value_counts())
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Genre distribution
        self.df['genre'].value_counts().plot(kind='bar', ax=axes[0,0], title='Genre Distribution')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Mood distribution
        self.df['mood'].value_counts().plot(kind='bar', ax=axes[0,1], title='Mood Distribution')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Audio features correlation
        corr_matrix = self.df[self.audio_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1,0])
        axes[1,0].set_title('Audio Features Correlation')
        
        # Valence vs Energy scatter plot
        scatter = axes[1,1].scatter(self.df['valence'], self.df['energy'], 
                                   c=pd.Categorical(self.df['mood']).codes, alpha=0.6)
        axes[1,1].set_xlabel('Valence')
        axes[1,1].set_ylabel('Energy')
        axes[1,1].set_title('Valence vs Energy by Mood')
        
        plt.tight_layout()
        plt.show()
        
        return self.df.describe()
    
    def train_classifiers(self):
        """Step 2: Train KNN classifiers for genre and mood prediction"""
        print("=== TRAINING KNN CLASSIFIERS ===")
        
        # Prepare features
        X = self.df[self.audio_features].copy()
        y_genre = self.df['genre']
        y_mood = self.df['mood']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_genre_train, y_genre_test = train_test_split(
            X_scaled, y_genre, test_size=0.2, random_state=42, stratify=y_genre)
        
        _, _, y_mood_train, y_mood_test = train_test_split(
            X_scaled, y_mood, test_size=0.2, random_state=42, stratify=y_mood)
        
        # Train genre classifier
        self.genre_classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        self.genre_classifier.fit(X_train, y_genre_train)
        
        # Train mood classifier
        self.mood_classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        self.mood_classifier.fit(X_train, y_mood_train)
        
        # Evaluate classifiers
        genre_pred = self.genre_classifier.predict(X_test)
        mood_pred = self.mood_classifier.predict(X_test)
        
        print("Genre Classifier Performance:")
        print(f"Accuracy: {accuracy_score(y_genre_test, genre_pred):.3f}")
        print("\nGenre Classification Report:")
        print(classification_report(y_genre_test, genre_pred))
        
        print("\nMood Classifier Performance:")
        print(f"Accuracy: {accuracy_score(y_mood_test, mood_pred):.3f}")
        print("\nMood Classification Report:")
        print(classification_report(y_mood_test, mood_pred))
        
        return self.genre_classifier, self.mood_classifier
    
    def extract_audio_features(self, song_input):
        """Step 1: Extract audio features from input song"""
        if isinstance(song_input, dict):
            # Direct feature input
            features = {feature: song_input.get(feature, 0) for feature in self.audio_features}
        elif isinstance(song_input, str):
            # Song name lookup
            song_row = self.df[self.df['track_name'].str.contains(song_input, case=False)]
            if song_row.empty:
                raise ValueError(f"Song '{song_input}' not found in dataset")
            features = song_row[self.audio_features].iloc[0].to_dict()
        elif isinstance(song_input, int):
            # Index lookup
            if song_input >= len(self.df):
                raise ValueError(f"Index {song_input} out of range")
            features = self.df.iloc[song_input][self.audio_features].to_dict()
        else:
            raise ValueError("Invalid input type. Use dict, string, or int.")
        
        return features
    
    def classify_song(self, audio_features):
        """Step 2: Classify genre and mood using pre-trained KNN classifiers"""
        if self.genre_classifier is None or self.mood_classifier is None:
            raise ValueError("Classifiers not trained. Call train_classifiers() first.")
        
        # Convert to array and scale
        feature_array = np.array([list(audio_features.values())]).reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_array)
        
        # Predict genre and mood
        predicted_genre = self.genre_classifier.predict(feature_scaled)[0]
        predicted_mood = self.mood_classifier.predict(feature_scaled)[0]
        
        # Get prediction probabilities
        genre_proba = self.genre_classifier.predict_proba(feature_scaled)[0]
        mood_proba = self.mood_classifier.predict_proba(feature_scaled)[0]
        
        return {
            'predicted_genre': predicted_genre,
            'predicted_mood': predicted_mood,
            'genre_confidence': max(genre_proba),
            'mood_confidence': max(mood_proba)
        }
    
    def filter_dataset(self, genre, mood, strict_genre=True, strict_mood=True):
        # Drop rows with missing genre or mood labels
        df = self.df.dropna(subset=['genre', 'mood'])

        genre_mask = df['genre'] == genre if strict_genre else df['genre'].notna()
        mood_mask = df['mood'] == mood if strict_mood else df['mood'].notna()

        filtered_df = df[genre_mask & mood_mask].copy()
        return filtered_df
    
    def compute_similarity(self, input_features, filtered_df, distance_metric='euclidean'):
        """Step 4: Compute similarity between input song and filtered candidates"""
        if filtered_df.empty:
            return pd.DataFrame()
        
        try:
            # Get audio features for filtered songs
            candidate_features = filtered_df[self.audio_features].values
            input_array = np.array([list(input_features.values())])
            
            # Calculate distances
            if distance_metric == 'euclidean':
                distances = cdist(input_array, candidate_features, metric='euclidean')[0]
            elif distance_metric == 'cosine':
                # Convert to cosine similarity (1 - cosine distance)
                similarities = cosine_similarity(input_array, candidate_features)[0]
                distances = 1 - similarities
            else:
                raise ValueError("distance_metric must be 'euclidean' or 'cosine'")
            
            # Add distances to dataframe
            result_df = filtered_df.copy()
            result_df['similarity_score'] = distances
            result_df = result_df.sort_values('similarity_score')
            
            return result_df
            
        except Exception as e:
            print(f"\nError during similarity computation: {str(e)}")
            return pd.DataFrame()
    
    def get_recommendations(self, song_input, n_recommendations=5, distance_metric='euclidean',
                          strict_genre=True, strict_mood=True):
        """Step 5: Complete recommendation pipeline"""
        print("=== MUSIC RECOMMENDATION SYSTEM ===")

        try:
            # Step 1: Extract audio features
            print("\nStep 1: Extracting audio features...")
            audio_features = self.extract_audio_features(song_input)
            print(f"Audio features extracted: {audio_features}")

            # Step 2: Classify genre and mood
            print("\nStep 2: Classifying genre and mood...")
            classification = self.classify_song(audio_features)
            predicted_genre = classification['predicted_genre']
            predicted_mood = classification['predicted_mood']
            print(f"Predicted genre: {predicted_genre} (confidence: {classification['genre_confidence']:.3f})")
            print(f"Predicted mood: {predicted_mood} (confidence: {classification['mood_confidence']:.3f})")

            # Step 3: Filter dataset with relaxed matching
            print("\nStep 3: Finding similar songs...")
            
            # Start with the entire dataset for maximum flexibility
            filtered_df = self.df.copy()
            
            # Try to find songs in the same genre first
            genre_matches = filtered_df[filtered_df['genre'].str.contains(predicted_genre, case=False, na=False)]
            if not genre_matches.empty:
                print(f"Found {len(genre_matches)} songs in the same genre")
                filtered_df = genre_matches
            else:
                print("No exact genre matches, using all songs for recommendations")

            # Step 4: Compute similarity with audio features
            print("\nStep 4: Computing similarity based on audio features...")
            results_df = self.compute_similarity(audio_features, filtered_df, distance_metric=distance_metric)

            # Step 5: Return Top Matches
            print("\nStep 5: Top Recommendations:")
            # Get top matches but exclude the input song if it exists
            if isinstance(song_input, str):
                results_df = results_df[~results_df['track_name'].str.contains(song_input, case=False, na=False)]
            
            top_matches = results_df.head(n_recommendations)
            
            # Print recommendations in a more readable format
            print("\nRecommended Songs:")
            for idx, row in top_matches.iterrows():
                print(f"\n{row['track_name']} by {row['artist_name']}")
                print(f"Genre: {row['genre']}")
                print(f"Mood: {row['mood']}")
                print(f"Similarity score: {row['similarity_score']:.3f}")
                # Print key audio features for comparison
                print("Audio Features:")
                for feature in self.audio_features:
                    if feature in row:
                        print(f"  {feature}: {row[feature]:.3f}")
            
            return top_matches

        except Exception as e:
            print(f"Error during recommendation pipeline: {str(e)}")
            # If there's an error, try one more time with the entire dataset
            print("Retrying with all songs...")
            try:
                results_df = self.compute_similarity(audio_features, self.df, distance_metric=distance_metric)
                return results_df.head(n_recommendations)
            except:
                print("Failed to generate recommendations. Please try a different song.")
                return pd.DataFrame()


# Example usage and testing
def main():
    # Initialize the system
    recommender = MusicRecommendationSystem()
    
    # Load data (use actual Spotify dataset)
    print("Loading Spotify dataset...")
    df = recommender.load_data(file_path='data/dataset.csv')
    
    # Train classifiers
    print("\nTraining genre and mood classifiers...")
    recommender.train_classifiers()
    
    while True:
        print("\n" + "="*50)
        print("SPOTIFY SONG RECOMMENDER")
        print("="*50)
        print("\nChoose an option:")
        print("1. Search by song name")
        print("2. Search by song index")
        print("3. Search by custom audio features")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            song_name = input("\nEnter song name: ")
            try:
                recommendations = recommender.get_recommendations(
                    song_input=song_name,
                    n_recommendations=5,
                    distance_metric='euclidean',
                    strict_genre=True,
                    strict_mood=True
                )
                
                print("\nRECOMMENDATIONS:")
                if not recommendations.empty:
                    for idx, row in recommendations.iterrows():
                        print(f"\n{row['track_name']} by {row['artist_name']}")
                        print(f"Genre: {row['genre']}")
                        print(f"Mood: {row['mood']}")
                        print(f"Similarity score: {row['similarity_score']:.3f}")
                else:
                    print("No recommendations found. Try a different song or relax the genre/mood filters.")
            
            except ValueError as e:
                print(f"Error: {str(e)}")
        
        elif choice == '2':
            try:
                song_idx = int(input("\nEnter song index (0-113999): "))
                if song_idx < 0 or song_idx >= len(df):
                    print("Invalid index. Please enter a number between 0 and 113999.")
                    continue
                
                print(f"\nSelected song: {df.iloc[song_idx]['track_name']} by {df.iloc[song_idx]['artist_name']}")
                print(f"Actual genre: {df.iloc[song_idx]['genre']}")
                
                recommendations = recommender.get_recommendations(
                    song_input=song_idx,
                    n_recommendations=5,
                    distance_metric='euclidean',
                    strict_genre=True,
                    strict_mood=True
                )
                
                print("\nRECOMMENDATIONS:")
                if not recommendations.empty:
                    for idx, row in recommendations.iterrows():
                        print(f"\n{row['track_name']} by {row['artist_name']}")
                        print(f"Genre: {row['genre']}")
                        print(f"Mood: {row['mood']}")
                        print(f"Similarity score: {row['similarity_score']:.3f}")
                else:
                    print("No recommendations found. Try a different song or relax the genre/mood filters.")
            
            except ValueError:
                print("Please enter a valid number.")
        
        elif choice == '3':
            print("\nEnter custom audio features (values between 0 and 1):")
            try:
                custom_features = {
                    'tempo': float(input("Tempo (BPM): ")),
                    'valence': float(input("Valence (0-1): ")),
                    'energy': float(input("Energy (0-1): ")),
                    'danceability': float(input("Danceability (0-1): ")),
                    'acousticness': float(input("Acousticness (0-1): ")),
                    'instrumentalness': float(input("Instrumentalness (0-1): ")),
                    'liveness': float(input("Liveness (0-1): ")),
                    'speechiness': float(input("Speechiness (0-1): "))
                }
                
                recommendations = recommender.get_recommendations(
                    song_input=custom_features,
                    n_recommendations=5,
                    distance_metric='cosine'
                )
                
                print("\nRECOMMENDATIONS:")
                if not recommendations.empty:
                    for idx, row in recommendations.iterrows():
                        print(f"\n{row['track_name']} by {row['artist_name']}")
                        print(f"Genre: {row['genre']}")
                        print(f"Mood: {row['mood']}")
                        print(f"Similarity score: {row['similarity_score']:.3f}")
                else:
                    print("No recommendations found. Try different audio features.")
            
            except ValueError:
                print("Please enter valid numbers between 0 and 1.")
        
        elif choice == '4':
            print("\nThank you for using the Spotify Song Recommender!")
            break
        
        else:
            print("\nInvalid choice. Please enter a number between 1 and 4.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()