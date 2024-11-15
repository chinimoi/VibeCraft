import streamlit as st
import pandas as pd
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy

# Spotify API Setup
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id='1a7c870a1c314d41adf9244e1b563783',
    client_secret='400d2a0dfc014bc3a1ab53e3c1c04b01'))

# Data Loading
data = pd.read_csv("data.csv")

# Define necessary functions
def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q=f'track: {name} year: {year}', limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name'])
                                 & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    except IndexError:
        return find_song(song['name'], song['year'])

def get_mean_vector(song_list, spotify_data):
    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            st.warning(f"Warning: {song['name']} does not exist in Spotify or database")
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []

    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    return flattened_dict

def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

# Pipeline Setup
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms',
               'energy', 'explicit', 'instrumentalness', 'key', 'liveness',
               'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('kmeans', KMeans(n_clusters=20, verbose=False))])
X = data[number_cols]
song_cluster_pipeline.fit(X)

# Frontend with Streamlit
st.title("Music Recommender System 🎵")
st.write("Enter a song and year to get recommendations!")

song_name = st.text_input("Enter Song Name", "Bloody Sweet")
song_year = st.number_input("Enter Song Year", min_value=1900, max_value=2024, value=2023)

if st.button("Get Recommendations"):
    song_list = [{'name': song_name, 'year': song_year}]
    recommendations = recommend_songs(song_list, data)
    if recommendations:
        st.write("**Recommended Songs**")
        for rec in recommendations:
            st.write(f"**{rec['name']}** by {rec['artists']} ({rec['year']})")
    else:
        st.write("No recommendations found.")
