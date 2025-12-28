
import io
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans

st.set_page_config(page_title="Netflix & Spotify Explorer", layout="wide")

def load_csv(uploaded_file, fallback_path):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Save uploaded file to fallback_path for persistence
            df.to_csv(fallback_path, index=False)
            return df
        except Exception as e:
            st.warning(f"Could not read uploaded CSV: {e}")
            return None
    if fallback_path and os.path.exists(fallback_path):
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
    st.pyplot(fig, clear_figure=True, width='stretch')
    plt.close(fig)

st.sidebar.title("Data")
st.sidebar.write("Upload your CSVs (or keep blank to try fallback paths).")

netflix_file = st.sidebar.file_uploader("Netflix CSV (netflix_titles.csv)", type=["csv"], key="netflix")
spotify_file = st.sidebar.file_uploader("Spotify CSV (tracks.csv / Spotify_Youtube.csv)", type=["csv"], key="spotify")

fallback_netflix = "netflix_titles.csv"
fallback_spotify = "Spotify_Youtube.csv"

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

        # Additional Netflix time-series & seasonality plots
        # Use filtered (year range) for time series where possible
        ds = filtered.copy()
        if 'date_added' in ds.columns:
            ds['date_added'] = pd.to_datetime(ds['date_added'], errors='coerce')
            ds_dates = ds.dropna(subset=['date_added']).copy()

            if not ds_dates.empty:
                ts_weekly = ds_dates.set_index('date_added').resample('W').size()
                ts_monthly = ds_dates.set_index('date_added').resample('ME').size()

                fig = plt.figure(figsize=(8, 2.5))
                plt.plot(ts_weekly.index, ts_weekly.values, color='tab:blue')
                plt.title('Weekly Netflix Additions')
                plt.xlabel('Week')
                plt.ylabel('Count')
                plt.tight_layout()
                plot_and_show(fig)

                fig = plt.figure(figsize=(8, 2.5))
                plt.plot(ts_monthly.index, ts_monthly.values, color='tab:orange')
                plt.title('Monthly Netflix Additions')
                plt.xlabel('Month')
                plt.ylabel('Count')
                plt.tight_layout()
                plot_and_show(fig)

                fig = plt.figure(figsize=(8, 2.5))
                plt.plot(ts_monthly.index, ts_monthly.cumsum().values, color='tab:green')
                plt.title('Cumulative Netflix Additions (by month)')
                plt.xlabel('Month')
                plt.ylabel('Cumulative Count')
                plt.tight_layout()
                plot_and_show(fig)
            else:
                st.info('No valid `date_added` values to plot time series in selected range.')
        elif 'release_year' in ds.columns:
            year_counts = ds['release_year'].dropna().astype(int).value_counts().sort_index()
            if not year_counts.empty:
                fig = plt.figure()
                plt.plot(year_counts.index, year_counts.values)
                plt.title('Yearly Netflix Releases')
                plt.xlabel('Year')
                plt.ylabel('Count')
                plt.tight_layout()
                plot_and_show(fig)
            else:
                st.info('No release_year data available for the selected range.')
        else:
            st.info('No date columns (`date_added` or `release_year`) available for time series.')

        # Seasonality: month-by-month distribution and monthly average
        if 'date_added' in ds.columns:
            ds['date_added'] = pd.to_datetime(ds['date_added'], errors='coerce')
            df_dates = ds.dropna(subset=['date_added']).copy()
            if not df_dates.empty:
                df_dates['year'] = df_dates['date_added'].dt.year
                df_dates['month'] = df_dates['date_added'].dt.month
                monthly_by_year = (df_dates.groupby(['year','month']).size().reset_index(name='count'))

                if not monthly_by_year.empty:
                    # boxplot of counts per month across years
                    month_groups = [monthly_by_year.loc[monthly_by_year['month'] == m, 'count'].values for m in range(1,13)]
                    fig = plt.figure(figsize=(10,3))
                    plt.boxplot(month_groups, tick_labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
                    plt.title('Distribution of Monthly Additions by Month (across years)')
                    plt.xlabel('Month')
                    plt.ylabel('Count')
                    plt.tight_layout()
                    plot_and_show(fig)

                    monthly_avg = monthly_by_year.groupby('month')['count'].mean()
                    fig = plt.figure(figsize=(8,2.5))
                    plt.plot(monthly_avg.index, monthly_avg.values, marker='o')
                    plt.xticks(range(1,13), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
                    plt.title('Average Titles Added per Month (across years)')
                    plt.xlabel('Month')
                    plt.ylabel('Average Count')
                    plt.tight_layout()
                    plot_and_show(fig)
                else:
                    st.info('Not enough date data to compute month-by-month distributions.')
            else:
                st.info('No valid `date_added` values to compute seasonality.')
        else:
            st.info('`date_added` column not present; seasonality plots skipped.')

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

        # Additional Spotify visualizations
        st.subheader("Density (KDE) of Audio Features")
        if audio_features:
            cols = audio_features[:6]
            fig = plt.figure(figsize=(8,4))
            for f in cols:
                sp[f].dropna().plot(kind='kde', label=f)
            plt.legend()
            plt.title('Density (KDE) of Audio Features')
            plt.xlabel('Value')
            plt.tight_layout()
            plot_and_show(fig)
        else:
            st.info('No audio features found for density plots.')

        st.subheader('Pairwise Scatter Matrix')
        from pandas.plotting import scatter_matrix
        pair_cols = [c for c in ['danceability','energy','valence','tempo'] if c in sp.columns]
        if len(pair_cols) >= 2:
            sample = sp[pair_cols].dropna().sample(n=min(1000, len(sp)), random_state=42)
            fig = plt.figure(figsize=(10,10))
            _ = scatter_matrix(sample, figsize=(10,10), diagonal='kde')
            plt.suptitle('Pairwise Scatter Matrix (sampled)')
            plt.tight_layout()
            # scatter_matrix draws directly to plt; capture current figure
            plot_and_show(plt.gcf())
        else:
            st.info('Not enough audio feature columns for pair plots.')

        st.subheader('Violin & Box: Danceability by Genre (Top)')
        if genre_col and 'danceability' in sp.columns:
            topg = sp[genre_col].dropna().value_counts().head(6).index
            data = [sp.loc[sp[genre_col] == g, 'danceability'].dropna().values for g in topg]
            if any(len(d) > 0 for d in data):
                fig = plt.figure(figsize=(10,4))
                plt.violinplot(data, showmedians=True)
                plt.xticks(range(1, len(topg) + 1), topg, rotation=45)
                plt.title('Violin plot: Danceability by Genre (Top 6)')
                plt.tight_layout()
                plot_and_show(fig)

                fig = plt.figure(figsize=(10,4))
                plt.boxplot(data)
                plt.xticks(range(1, len(topg) + 1), topg, rotation=45)
                plt.title('Box plot: Danceability by Genre (Top 6)')
                plt.tight_layout()
                plot_and_show(fig)
            else:
                st.info('Not enough danceability data per genre to draw violin/box plots.')
        else:
            st.info('Genre column or danceability column not available for violin/box plots.')

        st.subheader('KMeans Model Visualization (centroids)')
        if {'danceability','energy'}.issubset(set(sp.columns)):
            Xv = sp[['danceability','energy']].dropna()
            if len(Xv) > 10:
                from sklearn.cluster import KMeans as _KMeans
                k = st.slider('K for model viz', 2, 8, 4)
                km2 = _KMeans(n_clusters=int(k), n_init=10, random_state=42)
                labels2 = km2.fit_predict(Xv)
                centers2 = km2.cluster_centers_
                fig = plt.figure(figsize=(6,5))
                plt.scatter(Xv['danceability'], Xv['energy'], c=labels2, alpha=0.4, s=20)
                plt.scatter(centers2[:,0], centers2[:,1], c='red', s=120, marker='X', label='centroids')
                plt.title(f'KMeans (k={k}) — danceability vs energy with centroids')
                plt.xlabel('danceability')
                plt.ylabel('energy')
                plt.legend()
                plt.tight_layout()
                plot_and_show(fig)
            else:
                st.info('Not enough rows with danceability & energy to build model viz.')
        else:
            st.info('danceability and/or energy not present — cannot visualize model.')



with tab_about:
    st.header("About Us")

    st.markdown("""
**App:** Netflix & Spotify Explorer  
**Purpose:** Turn EDA into an interactive, portfolio-ready tool.

**Built with:** Streamlit, pandas, numpy, matplotlib, scikit-learn

**Datasets:**
- Netflix Movies & TV Shows (Kaggle)
- Spotify Tracks (Kaggle)

**Team:**
- Owais Ali Khadim Batti
- Navneet Pitani
- Neha Susan
- Parth Patil

""")

#streamlit run streamlit_app.py