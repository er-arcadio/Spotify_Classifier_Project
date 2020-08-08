import math
import time
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from WebScrape._SpotifyCredentials import credentials


def get_features_df(tracks_artists):
    final_df = pd.DataFrame()
    length = len(tracks_artists)
    if not length:
        final_df = None
    elif length > 100:
        partitions = math.ceil(length / 100)
        for partition in range(partitions):
            start = partition*100
            end = start + 100 if partition != partitions-1 else length
            under100_tracks_artists = tracks_artists[start:end]
            final_df = pd.concat([final_df,
                                  get_under100_features(under100_tracks_artists)],
                                 ignore_index=True)
            print(f"{start+1} to {end} complete!")
            time.sleep(1)
    else:
        final_df = get_under100_features(tracks_artists)

    # check if there are any null values in df
    null_count = final_df.isnull().sum().sum() // 13
    if null_count:
        print(f'{null_count} null values found. Now adjusting...')
        mask = np.isnan(final_df['key'])
        add_these = final_df[mask].iloc[:, :2].to_numpy()
        nan_tracks = get_features_df(add_these)
        nan_indexes = final_df.index[mask].tolist()
        for good_row, bad_row in enumerate(nan_indexes):
            final_df.iloc[bad_row] = nan_tracks.iloc[good_row]
        print("Null values recovered")

    return final_df


def get_under100_features(under100_tracks_artists):
    track_ids = []

    # fetch the ids for each song
    for pair in under100_tracks_artists:
        query = f"track:{pair[0]} artist:{pair[1]}"
        response = credentials.search(q=query, type='track')
        try:
            track_ids.append(response['tracks']['items'][0]['id'])
        except IndexError:
            print(f"Having trouble finding '{pair[0]}' by '{pair[1]}'")

    # 100 at a time! careful
    audio_features_dicts = credentials.audio_features(tracks=track_ids)

    df_dict = {
        'track': [],
        'artist': [],
        'danceability': [],
        'energy': [],
        'key': [],
        'loudness': [],
        'mode': [],
        'speechiness': [],
        'acousticness': [],
        'instrumentalness': [],
        'liveness': [],
        'valence': [],
        'tempo': [],
        'duration_ms': [],
        'time_signature': [],
        'top100': [],
        'top10': []
    }

    for idx, pair in enumerate(under100_tracks_artists):
        df_dict['track'].append(pair[0])
        df_dict['artist'].append(pair[1])
        for i, key in enumerate(df_dict):
            if 1 < i < 15:
                try:
                    df_dict[key].append(audio_features_dicts[idx][key])
                except IndexError:
                    df_dict[key].append(np.nan)
        df_dict['top100'].append(0)
        df_dict['top10'].append(0)

    return pd.DataFrame(df_dict)


def get_top200(date='latest'):
    # Date should be 'yyyy-mm-dd--yyyy-mm-dd' From Friday to Friday (1 week span)
    # For example '2020-07-03--2020-07-10' where July 3rd and July 10th is a Friday
    response = requests.get(f'https://spotifycharts.com/regional/global/weekly/{date}')
    soup = BeautifulSoup(response.text, 'html5lib')

    top200_html = soup.find_all('td', attrs={'class': 'chart-table-track'})

    top200_songs = []
    for song_html in top200_html:
        track = song_html.find('strong').text.rsplit('(')[0]
        artist = song_html.find('span').text[3:]

        top200_songs.append((track, artist))

    return top200_songs


def get_songs(link='https://open.spotify.com/playlist/5tIkO3qnEYSRYnEs1jgP8x', song_count=1000):
    songs = []
    for offset in range(0, song_count, 100):
        playlist = credentials.playlist_tracks(
            link, limit=100, offset=offset)
        for idx in range(100):
            try:
                track = playlist['items'][idx]['track']['name'].rsplit('(')[0].rsplit('-')[0]
                artist = playlist['items'][idx]['track']['artists'][0]['name']
                songs.append((track, artist))
            except IndexError:
                pass
    print(f'{len(songs)} track and artist names uploaded')
    return songs
