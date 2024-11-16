VibeCraft: A Personalized Music Recommendation System
VibeCraft is a focused music recommendation tool that leverages machine learning to suggest five songs based on a single input track. By analyzing the audio characteristics and patterns of the input song, VibeCraft identifies tracks that match the vibe, genre, and mood, offering a quick and personalized way to explore music.

Features
Personalized Recommendations: Get five songs tailored to match the input track's audio features.
Audio Feature Analysis: Uses features like energy, danceability, tempo, and valence to ensure accurate recommendations.
Interactive Interface: User-friendly Streamlit app for inputting song details and viewing recommendations.
Spotify Integration: Fetches song details and audio features using the Spotify API.
Clustering and Visualization: Uses K-Means clustering, PCA, and t-SNE for grouping and visualizing similar songs.

Technologies Used
Programming Languages: Python
Frameworks and Libraries:
Streamlit: For the web app interface.
Spotipy: For Spotify API integration.
Scikit-learn: For clustering and preprocessing.
Pandas & NumPy: For data manipulation.
Plotly: For visualizations.
Spotify API: To fetch audio features and song details.

Project Workflow
Data Collection: Fetch song data and audio features using the Spotify API.
Clustering: Apply K-Means to group similar songs based on their features.
Dimensionality Reduction: Use PCA and t-SNE for visualization and efficient processing.
Recommendation Algorithm: Find songs similar to the input track using cosine similarity.
Output: Display recommended songs with details like name, artist, and release year.

Future Enhancements
Incorporate user preferences for improved recommendations.
Expand the dataset to cover more genres and regional music.
Use advanced deep learning models for feature extraction.
Enable real-time playlist generation on Spotify.
