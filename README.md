# Netflix & Spotify Data Explorer

An interactive web application for exploratory data analysis (EDA) of Netflix and Spotify datasets. Built with Streamlit, this app allows users to visualize trends, distributions, and insights from movie/TV show data and music track features.

## 🚀 Live Demo

Access the deployed app here: [https://ns-data-viz.streamlit.app/](https://ns-data-viz.streamlit.app/)

## 📋 Features

### Netflix Tab
- **Movies vs TV Shows**: Bar chart of content types.
- **Top Genres**: Horizontal bar chart of most popular genres.
- **Releases Over Time**: Line plot of content releases by year.
- **Country-wise Production**: Top countries producing Netflix content.
- **Genres × Release Year Heatmap**: Visualizing genre popularity over time.
- **Time Series Analysis**: Weekly, monthly, and cumulative additions (if date data available).
- **Seasonality Plots**: Monthly distributions and averages for content additions.

### Spotify Tab
- **Audio Feature Distributions**: Histograms for danceability, energy, etc.
- **Danceability vs Energy Scatter Plot**: Interactive scatter with adjustable alpha and size.
- **Popularity by Genre**: Average popularity scores for top genres.
- **Correlation Heatmap**: Relationships between audio features.
- **K-Means Clustering**: Mood-based clustering with adjustable k.
- **Density Plots (KDE)**: Kernel density estimates for audio features.
- **Pairwise Scatter Matrix**: Correlations between selected features.
- **Violin & Box Plots**: Danceability distributions by genre.
- **Model Visualization**: K-Means clusters with centroids.

### General
- **Data Upload**: Upload custom CSV files or use defaults.
- **Filters**: Year range sliders, sample fractions for performance.
- **Interactive Controls**: Sliders for cluster count, point styles, etc.

## 🛠️ Technologies Used

- **Streamlit**: For building the interactive web app.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Matplotlib**: Plotting and visualizations.
- **Scikit-learn**: K-Means clustering.
- **Python**: Core programming language.

## 📊 Datasets

- **Netflix Movies and TV Shows**: Sourced from Kaggle, containing details like type, genre, release year, country, etc.
- **Spotify Tracks**: Audio features and metadata for music tracks, including danceability, energy, popularity, etc.

## 🚀 How to Run Locally

1. **Clone or Download** the project repository.
2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```
3. **Run the App**:
   ```
   streamlit run streamlit_app.py
   ```
4. Open your browser to `http://localhost:8501`.

## 👥 Team

- **Owais Ali Khadim Batti**
- **Navneet Pitani**
- **Neha Susan**
- **Parth Patil**

## 📝 License

This project is for educational and portfolio purposes. Datasets are publicly available from Kaggle.

---

*Built as part of a college project to demonstrate data visualization and interactive app development.*