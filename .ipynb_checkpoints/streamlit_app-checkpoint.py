
import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans

st.set_page_config(page_title="Netflix & Spotify Explorer", layout="wide")

def load_csv(uploaded_file, fallback_path=None):
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.warning(f"Could not read uploaded CSV: {e}")
            return None
    if fallback_path:
        try:
            return pd.read_csv(fallback_path)
        except Exception as e:
            st.info(f"Tried to load fallback '{fallback_path}' but failed: {e}")
    return None

def ensure_list_col_exploded(df, col):
    if col not in df.columns:
        return df.assign(**{col: np.nan})
    d = df.copy()
    d[col] = d[col].fillna('').astype(str).str.split(',')
    d = d.explode(col)
    d[col] = d[col].str.strip()
    return d

def plot_and_show(fig):
    st.pyplot(fig, clear_figure=True, use_container_width=True)

st.sidebar.title("Data")
st.sidebar.write("Upload your CSVs (or keep blank to try fallback paths).")

netflix_file = st.sidebar.file_uploader("Netflix CSV (netflix_titles.csv)", type=["csv"], key="netflix")
spotify_file = st.sidebar.file_uploader("Spotify CSV (tracks.csv / Spotify_Youtube.csv)", type=["csv"], key="spotify")

fallback_netflix = st.sidebar.text_input("Fallback Netflix path", value="data/netflix_titles.csv")
fallback_spotify = st.sidebar.text_input("Fallback Spotify path", value="data/tracks.csv")

netflix = load_csv(netflix_file, fallback_netflix)
spotify = load_csv(spotify_file, fallback_spotify)

tab_netflix, tab_spotify, tab_about = st.tabs(["🎬 Netflix", "🎧 Spotify", "ℹ️ About"])

with tab_netflix:
    st.header("Netflix EDA")
    if netflix is None or netflix.empty:
        st.info("Upload or provide a valid Netflix CSV to begin.")
    else:
        netflix = netflix.copy()
        if 'release_year' in netflix.columns:
            netflix['release_year'] = pd.to_numeric(netflix['release_year'], errors='coerce')
        if 'date_added' in netflix.columns:
            netflix['date_added'] = pd.to_datetime(netflix['date_added'], errors='coerce')

        st.subheader("Filters")
        min_year = int(np.nanmin(netflix.get('release_year', pd.Series([1900])))) if 'release_year' in netflix.columns else 1900
        max_year = int(np.nanmax(netflix.get('release_year', pd.Series([2025])))) if 'release_year' in netflix.columns else 2025
        y1, y2 = st.slider("Release Year Range", min_value=min_year, max_value=max_year, value=(max(min_year, max_year-20), max_year), step=1)
        filtered = netflix.copy()
        if 'release_year' in filtered.columns:
            filtered = filtered[(filtered['release_year'] >= y1) & (filtered['release_year'] <= y2)]

        st.subheader("1) Movies vs TV Shows")
        if 'type' in filtered.columns:
            counts = filtered['type'].value_counts().sort_values(ascending=False)
            fig = plt.figure()
            counts.plot(kind='bar')
            plt.title('Netflix Library: Movies vs TV Shows')
            plt.xlabel('Type')
            plt.ylabel('Count')
            plt.tight_layout()
            plot_and_show(fig)
        else:
            st.warning("Column 'type' not found.")

        st.subheader("2) Top Genres")
        topN = st.number_input("Show Top N Genres", min_value=5, max_value=30, value=10, step=1)
        genres_long = ensure_list_col_exploded(filtered, 'listed_in')
        if 'listed_in' in genres_long.columns:
            top_genres = genres_long['listed_in'].value_counts().head(int(topN)).sort_values(ascending=True)
            fig = plt.figure()
            top_genres.plot(kind='barh')
            plt.title(f'Top {int(topN)} Genres on Netflix')
            plt.xlabel('Count')
            plt.ylabel('Genre')
            plt.tight_layout()
            plot_and_show(fig)
        else:
            st.warning("Column 'listed_in' not found.")

        st.subheader("3) Releases Over Time")
        if 'release_year' in filtered.columns:
            year_counts = filtered['release_year'].dropna().astype(int).value_counts().sort_index()
            fig = plt.figure()
            year_counts.plot(kind='line')
            plt.title('Netflix Releases Over Time')
            plt.xlabel('Release Year')
            plt.ylabel('Count')
            plt.tight_layout()
            plot_and_show(fig)
        else:
            st.warning("Column 'release_year' not found.")

        st.subheader("4) Country-wise Production")
        countries_long = ensure_list_col_exploded(filtered, 'country')
        topC = st.number_input("Show Top N Countries", min_value=5, max_value=30, value=10, step=1, key="ncountries")
        if 'country' in countries_long.columns:
            top_countries = (countries_long['country']
                             .replace('', np.nan)
                             .dropna()
                             .value_counts()
                             .head(int(topC))
                             .sort_values(ascending=True))
            fig = plt.figure()
            top_countries.plot(kind='barh')
            plt.title(f'Top {int(topC)} Countries by Netflix Titles Produced')
            plt.xlabel('Count')
            plt.ylabel('Country')
            plt.tight_layout()
            plot_and_show(fig)
        else:
            st.warning("Column 'country' not found.")

        st.subheader("5) Heatmap: Genres × Release Year")
        if 'listed_in' in genres_long.columns and 'release_year' in filtered.columns:
            tmp = genres_long[['listed_in','release_year']].dropna()
            tmp['release_year'] = tmp['release_year'].astype(int)
            pivot = tmp.groupby(['listed_in','release_year']).size().unstack(fill_value=0)

            topN_heat = st.number_input("Top N Genres for Heatmap", min_value=5, max_value=30, value=15, step=1)
            top_rows = pivot.sum(axis=1).sort_values(ascending=False).head(int(topN_heat)).index
            heat_small = pivot.loc[top_rows]

            fig = plt.figure()
            plt.imshow(heat_small.values, aspect='auto')
            plt.title('Genres vs Release Year (Top Genres)')
            plt.xlabel('Release Year')
            plt.ylabel('Genre')
            plt.xticks(ticks=np.arange(heat_small.shape[1]), labels=heat_small.columns, rotation=90)
            plt.yticks(ticks=np.arange(heat_small.shape[0]), labels=heat_small.index)
            plt.colorbar()
            plt.tight_layout()
            plot_and_show(fig)
        else:
            st.warning("Required columns for heatmap not found.")

with tab_spotify:
    st.header("Spotify EDA")
    if spotify is None or spotify.empty:
        st.info("Upload or provide a valid Spotify CSV to begin.")
    else:
        spotify = spotify.copy()

        genre_col = None
        for cand in ['track_genre', 'genre', 'playlist_genre', 'Artist_genre', 'artist_genre']:
            if cand in spotify.columns:
                genre_col = cand
                break

        pop_col = None
        for cand in ['popularity', 'Popularity']:
            if cand in spotify.columns:
                pop_col = cand
                break

        feature_candidates = [
            'danceability','energy','loudness','speechiness','acousticness','instrumentalness',
            'liveness','valence','tempo'
        ]
        audio_features = [c for c in feature_candidates if c in spotify.columns]

        st.write(f"Detected genre column: **{genre_col}**" if genre_col else "No genre column detected.")
        st.write(f"Detected popularity column: **{pop_col}**" if pop_col else "No popularity column detected.")
        st.write(f"Audio features: {audio_features if audio_features else 'None found'}")

        st.subheader("Filters")
        sample_frac = st.slider("Sample fraction (for faster plotting)", min_value=0.05, max_value=1.0, value=0.5, step=0.05)
        sp = spotify.sample(frac=sample_frac, random_state=42) if 0 < sample_frac < 1.0 else spotify

        st.subheader("6) Distributions of Audio Features")
        if audio_features:
            for feat in audio_features:
                fig = plt.figure()
                sp[feat].dropna().plot(kind='hist', bins=40)
                plt.title(f'Distribution of {feat}')
                plt.xlabel(feat)
                plt.ylabel('Frequency')
                plt.tight_layout()
                plot_and_show(fig)
        else:
            st.warning("No audio feature columns found.")

        st.subheader("7) Danceability vs Energy")
        if 'danceability' in sp.columns and 'energy' in sp.columns:
            alpha = st.slider("Point alpha", 0.1, 1.0, 0.3, 0.1)
            size = st.slider("Point size", 5, 100, 12, 1)
            fig = plt.figure()
            plt.scatter(sp['danceability'], sp['energy'], alpha=alpha, s=size)
            plt.title('Danceability vs Energy')
            plt.xlabel('danceability')
            plt.ylabel('energy')
            plt.tight_layout()
            plot_and_show(fig)
        else:
            st.warning("danceability/energy not found.")

        st.subheader("8) Average Popularity by Genre (Top 15)")
        if genre_col and pop_col:
            pop_by_genre = (sp[[genre_col, pop_col]]
                            .dropna(subset=[genre_col, pop_col])
                            .groupby(genre_col)[pop_col].mean()
                            .sort_values(ascending=False)
                            .head(15)
                            .sort_values(ascending=True))
            fig = plt.figure()
            pop_by_genre.plot(kind='barh')
            plt.title('Average Popularity by Genre (Top 15)')
            plt.xlabel('Average Popularity')
            plt.ylabel('Genre')
            plt.tight_layout()
            plot_and_show(fig)
        else:
            st.warning("Genre and/or popularity columns not found.")

        st.subheader("9) Correlation Heatmap of Audio Features")
        if len(audio_features) >= 2:
            corr = sp[audio_features].corr(numeric_only=True)
            fig = plt.figure()
            plt.imshow(corr.values, aspect='auto', vmin=-1, vmax=1)
            plt.title('Correlation Heatmap (Audio Features)')
            plt.xticks(ticks=np.arange(corr.shape[1]), labels=corr.columns, rotation=90)
            plt.yticks(ticks=np.arange(corr.shape[0]), labels=corr.index)
            plt.colorbar()
            plt.tight_layout()
            plot_and_show(fig)
        else:
            st.warning("Not enough audio features to compute correlation.")

        st.subheader("10) K-Means Clustering: Mood Groups")
        if {'danceability','energy'}.issubset(set(sp.columns)):
            k = st.slider("Number of clusters (k)", min_value=2, max_value=8, value=4, step=1)
            X = sp[['danceability','energy']].dropna().clip(0, 1)
            km = KMeans(n_clusters=int(k), n_init=10, random_state=42)
            labels = km.fit_predict(X)

            fig = plt.figure()
            plt.scatter(X['danceability'], X['energy'], c=labels, s=12, alpha=0.5)
            plt.title('K-Means Clusters: Mood Groups (danceability vs energy)')
            plt.xlabel('danceability')
            plt.ylabel('energy')
            plt.tight_layout()
            plot_and_show(fig)

            st.caption("Tip: Try k=3..6 and see how clusters separate 'chill', 'party', 'workout', etc.")
        else:
            st.warning("danceability and/or energy not found; cannot cluster.")


#streamlit run streamlit_app.py