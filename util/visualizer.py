import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Union

from util.corpus_dataloader import CorpusDataLoader


class EmotionVisualizer:
    """A class for visualizing emotion analysis of Vietnamese song corpus."""
    
    EMOTION_MAPPING = {
        '1-Vui': 'joy', 
        '2-Buồn': 'sadness', 
        '3-Giận dữ': 'anger', 
        '4-Sợ hãi': 'fear', 
        '5-Ghê tởm': 'disgust', 
        '6-Ngạc nhiên': 'surprise'
    }
    
    def __init__(self, labeled_path: str, corpus_path: str):
        """Initialize the visualizer with dataset paths."""
        self.dataset = self._load_and_prepare_data(labeled_path, corpus_path)
        
    def _load_and_prepare_data(self, labeled_path: str, corpus_path: str) -> pd.DataFrame:
        """Load and merge datasets, convert emotion labels."""
        dataset = pd.read_csv(labeled_path)
        old_dataset = CorpusDataLoader(corpus_path)
        old_df = old_dataset.data
        
        # Convert list columns to string format
        old_df['composers'] = old_df['composers'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x
        )
        old_df['lyricists'] = old_df['lyricists'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x
        )
        
        # Merge datasets
        dataset = dataset.merge(
            old_df[['title', 'composers', 'lyricists', 'year', 'note']], 
            on=['title', 'composers'], 
            how='left'
        )
        
        # Remove unnecessary columns
        cols_to_drop = ['id', 'lyrics', 'text_segment', 'avg_confidence', 'confidence_score']
        cols_to_drop = [c for c in cols_to_drop if c in dataset.columns]
        dataset = dataset.drop(columns=cols_to_drop)
        
        # Convert emotion labels to English
        dataset['label_emotion'] = dataset['label_emotion'].replace(self.EMOTION_MAPPING)
        
        return dataset
    
    # ==================== EXPLODE UTILITIES ====================
    
    def _explode_column(self, df: pd.DataFrame, column: str, 
                        exclude_unknown: bool = True) -> pd.DataFrame:
        """Explode a comma-separated column into individual rows."""
        result = df.dropna(subset=[column]).copy()
        result[column] = result[column].str.split(', ')
        result = result.explode(column)
        result[column] = result[column].str.strip()
        
        if exclude_unknown:
            result = result[result[column] != 'Unknown']
            
        return result
    
    # ==================== YEAR ANALYSIS ====================
    
    def prepare_year_data(self, min_year: int = 1975, max_year: int = 2026,
                          exclude_wikipedia: bool = True) -> pd.DataFrame:
        """Prepare filtered year data with decade column."""
        yeardf = self.dataset.dropna(subset=['year']).copy()
        yeardf = yeardf[(yeardf['year'] >= min_year) & (yeardf['year'] <= max_year)]
        
        if exclude_wikipedia:
            yeardf = yeardf[yeardf['note'] != "đã fill 'year' sử dụng Wikipedia"]
            
        yeardf['decade'] = (yeardf['year'] // 10 * 10).astype(int)
        return yeardf
    
    def plot_emotion_by_year(self, yeardf: Optional[pd.DataFrame] = None, 
                             figsize: tuple = (14, 6)):
        """Plot emotion trends by year."""
        if yeardf is None:
            yeardf = self.prepare_year_data()
            
        emotion_by_year = yeardf.groupby(['year', 'label_emotion']).size().reset_index(name='count')
        
        plt.figure(figsize=figsize)
        sns.lineplot(data=emotion_by_year, x='year', y='count', 
                     hue='label_emotion', marker='o')
        plt.title('Emotion Trends by Year')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
    def plot_emotion_by_decade(self, yeardf: Optional[pd.DataFrame] = None,
                               figsize: tuple = (12, 6)):
        """Plot emotion trends by decade."""
        if yeardf is None:
            yeardf = self.prepare_year_data()
            
        emotion_by_decade = yeardf.groupby(['decade', 'label_emotion']).size().reset_index(name='count')
        
        plt.figure(figsize=figsize)
        sns.barplot(data=emotion_by_decade, x='decade', y='count', hue='label_emotion')
        plt.title('Emotion Trends by Decade')
        plt.xlabel('Decade')
        plt.ylabel('Count')
        plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
    def plot_sadness_ratio_by_decade(self, yeardf: Optional[pd.DataFrame] = None,
                                     figsize: tuple = (14, 6)):
        """Plot sadness ratio by decade."""
        if yeardf is None:
            yeardf = self.prepare_year_data()
            
        emotion_counts = yeardf.groupby(['decade', 'label_emotion']).size().unstack(fill_value=0)
        emotion_counts['other'] = emotion_counts.drop(columns='sadness', errors='ignore').sum(axis=1)
        emotion_counts['sad_ratio'] = emotion_counts['sadness'] / (
            emotion_counts['sadness'] + emotion_counts['other']
        )
        
        plt.figure(figsize=figsize)
        plt.plot(emotion_counts.index, emotion_counts['sad_ratio'], 
                 marker='o', color='steelblue', linewidth=2)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
        plt.title('Ratio of Sadness to All Emotions by Decade')
        plt.xlabel('Decade')
        plt.ylabel('Sadness Ratio')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def summary_by_decade(self, yeardf: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get summary table by decade."""
        if yeardf is None:
            yeardf = self.prepare_year_data()
            
        summary = yeardf.groupby(['decade', 'label_emotion']).size().unstack(fill_value=0)
        summary['total'] = summary.sum(axis=1)
        summary['sadness_ratio'] = (summary['sadness'] / summary['total'] * 100).round(2)
        summary['other_ratio'] = (100 - summary['sadness_ratio']).round(2)
        
        cols = ['sadness', 'joy', 'anger', 'fear', 'total', 'sadness_ratio', 'other_ratio']
        cols = [c for c in cols if c in summary.columns]
        return summary[cols]
    
    # ==================== GENERIC ANALYSIS BY COLUMN ====================
    
    def _prepare_exploded_data(self, column: str, normalize_map: dict = None) -> pd.DataFrame:
        """Prepare exploded data for a column."""
        df = self._explode_column(self.dataset, column)
        
        if normalize_map:
            df[column] = df[column].replace(normalize_map)
            
        return df
    
    def _get_top_values(self, df: pd.DataFrame, column: str, n: int = 20) -> List[str]:
        """Get top N values by count."""
        return df[column].value_counts().head(n).index.tolist()
    
    def plot_emotion_distribution(self, df: pd.DataFrame, column: str, 
                                  top_n: int = 20, figsize: tuple = (14, 8)):
        """Plot emotion distribution for a column."""
        top_values = self._get_top_values(df, column, top_n)
        emotion_data = df.groupby([column, 'label_emotion']).size().reset_index(name='count')
        emotion_data_top = emotion_data[emotion_data[column].isin(top_values)]
        
        plt.figure(figsize=figsize)
        sns.barplot(data=emotion_data_top, x=column, y='count', hue='label_emotion')
        plt.title(f'Emotion Distribution by {column.title()} (Top {top_n})')
        plt.xlabel(column.title())
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
    def plot_emotion_percentage(self, df: pd.DataFrame, column: str, 
                                top_n: int = 20, figsize: tuple = (12, 10)):
        """Plot emotion percentage for a column."""
        top_values = self._get_top_values(df, column, top_n)
        emotion_pct = df.groupby([column, 'label_emotion']).size().unstack(fill_value=0)
        emotion_pct = emotion_pct.loc[top_values]
        emotion_pct_norm = emotion_pct.div(emotion_pct.sum(axis=1), axis=0) * 100
        
        emotion_pct_norm.plot(kind='barh', stacked=True, figsize=figsize, colormap='Set2')
        plt.title(f'Emotion Percentage by {column.title()} (Top {top_n})')
        plt.xlabel('Percentage (%)')
        plt.ylabel(column.title())
        plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
    def summary_by_column(self, df: pd.DataFrame, column: str, 
                          top_n: int = 30) -> pd.DataFrame:
        """Get summary table for a column."""
        summary = df.groupby([column, 'label_emotion']).size().unstack(fill_value=0)
        summary['total'] = summary.sum(axis=1)
        summary['sadness_ratio'] = (summary['sadness'] / summary['total'] * 100).round(2)
        summary['other_ratio'] = (100 - summary['sadness_ratio']).round(2)
        summary = summary.sort_values('total', ascending=False).head(top_n)
        
        cols = ['sadness', 'joy', 'anger', 'fear', 'total', 'sadness_ratio', 'other_ratio']
        cols = [c for c in cols if c in summary.columns]
        return summary[cols]
    
    # ==================== GENRE ANALYSIS ====================
    
    def prepare_genre_data(self) -> pd.DataFrame:
        """Prepare exploded genre data."""
        genredf = self._prepare_exploded_data('genres')
        genredf['genres'] = genredf['genres'].replace({'Quê hương': 'Quê Hương'})
        return genredf
    
    def analyze_genres(self, top_n: int = 15, show_plots: bool = True) -> pd.DataFrame:
        """Complete genre analysis with plots and summary."""
        genredf = self.prepare_genre_data()
        
        print(f"Top {top_n} genres:")
        print(genredf['genres'].value_counts().head(top_n))
        print(f"\nTotal records: {len(genredf)}")
        
        if show_plots:
            self.plot_emotion_distribution(genredf, 'genres', top_n)
            self.plot_emotion_percentage(genredf, 'genres', top_n)
            
        return self.summary_by_column(genredf, 'genres', top_n * 2)
    
    # ==================== COMPOSER ANALYSIS ====================
    
    def prepare_composer_data(self) -> pd.DataFrame:
        """Prepare exploded composer data."""
        return self._prepare_exploded_data('composers')
    
    def analyze_composers(self, top_n: int = 20, show_plots: bool = True) -> pd.DataFrame:
        """Complete composer analysis with plots and summary."""
        composerdf = self.prepare_composer_data()
        
        print(f"Top {top_n} composers:")
        print(composerdf['composers'].value_counts().head(top_n))
        print(f"\nTotal records: {len(composerdf)}")
        print(f"Unique composers: {composerdf['composers'].nunique()}")
        
        if show_plots:
            self.plot_emotion_distribution(composerdf, 'composers', top_n)
            self.plot_emotion_percentage(composerdf, 'composers', top_n)
            
        return self.summary_by_column(composerdf, 'composers', top_n + 10)
    
    # ==================== LYRICIST ANALYSIS ====================
    
    def prepare_lyricist_data(self) -> pd.DataFrame:
        """Prepare exploded lyricist data."""
        return self._prepare_exploded_data('lyricists')
    
    def analyze_lyricists(self, top_n: int = 20, show_plots: bool = True) -> pd.DataFrame:
        """Complete lyricist analysis with plots and summary."""
        lyricistdf = self.prepare_lyricist_data()
        
        print(f"Top {top_n} lyricists:")
        print(lyricistdf['lyricists'].value_counts().head(top_n))
        print(f"\nTotal records: {len(lyricistdf)}")
        print(f"Unique lyricists: {lyricistdf['lyricists'].nunique()}")
        
        if show_plots:
            self.plot_emotion_distribution(lyricistdf, 'lyricists', top_n)
            self.plot_emotion_percentage(lyricistdf, 'lyricists', top_n)
            
        return self.summary_by_column(lyricistdf, 'lyricists', top_n + 10)
    
    # ==================== FIND SONGS BY PERSON ====================
    
    def find_songs_by_person(self, name: str, 
                             show_all: bool = False) -> pd.DataFrame:
        """Find all songs associated with a person (as composer or lyricist)."""
        # Find as composer
        composerdf = self._explode_column(self.dataset, 'composers', exclude_unknown=False)
        as_composer = composerdf[composerdf['composers'] == name]
        
        # Find as lyricist
        lyricistdf = self._explode_column(self.dataset, 'lyricists', exclude_unknown=False)
        as_lyricist = lyricistdf[lyricistdf['lyricists'] == name]
        
        # Combine and remove duplicates
        cols = ['title', 'composers', 'lyricists', 'label_emotion', 'genres', 'year']
        songs = pd.concat([
            as_composer[cols],
            as_lyricist[cols]
        ]).drop_duplicates(subset=['title'])
        
        print(f"Total songs associated with '{name}': {len(songs)}")
        print(f"  - As composer: {len(as_composer)}")
        print(f"  - As lyricist: {len(as_lyricist)}")
        print(f"\nEmotion distribution:")
        print(songs['label_emotion'].value_counts())
        
        if show_all:
            return songs
        return songs.head(20)
    
    def get_songs_by_composer(self, name: str) -> pd.DataFrame:
        """Get all songs by a specific composer (returns DataFrame)."""
        composerdf = self._explode_column(self.dataset, 'composers', exclude_unknown=False)
        songs = composerdf[composerdf['composers'] == name].copy()
        return songs[['title', 'composers', 'lyricists', 'label_emotion', 'genres', 'year']]
    
    def get_songs_by_lyricist(self, name: str) -> pd.DataFrame:
        """Get all songs by a specific lyricist (returns DataFrame)."""
        lyricistdf = self._explode_column(self.dataset, 'lyricists', exclude_unknown=False)
        songs = lyricistdf[lyricistdf['lyricists'] == name].copy()
        return songs[['title', 'composers', 'lyricists', 'label_emotion', 'genres', 'year']]
    
    def plot_person_timeline(self, name: str, figsize: tuple = (14, 6)):
        """Plot emotion timeline by year for a specific person (composer or lyricist)."""
        # Get songs by person
        composerdf = self._explode_column(self.dataset, 'composers', exclude_unknown=False)
        lyricistdf = self._explode_column(self.dataset, 'lyricists', exclude_unknown=False)
        
        as_composer = composerdf[composerdf['composers'] == name]
        as_lyricist = lyricistdf[lyricistdf['lyricists'] == name]
        
        cols = ['title', 'composers', 'lyricists', 'label_emotion', 'genres', 'year']
        songs = pd.concat([as_composer[cols], as_lyricist[cols]]).drop_duplicates(subset=['title'])
        
        # Remove NA years and filter to valid range (1900-2026)
        songs = songs.dropna(subset=['year'])
        songs['year'] = songs['year'].astype(int)
        songs = songs[(songs['year'] >= 1900) & (songs['year'] <= 2026)]
        
        if len(songs) == 0:
            print(f"No songs with year data found for '{name}'")
            return
        
        print(f"Songs with year data for '{name}': {len(songs)}")
        print(f"Year range: {songs['year'].min()} - {songs['year'].max()}")
        
        # Group by year and emotion
        timeline = songs.groupby(['year', 'label_emotion']).size().reset_index(name='count')
        
        plt.figure(figsize=figsize)
        sns.lineplot(data=timeline, x='year', y='count', hue='label_emotion', marker='o')
        plt.title(f'Emotion Timeline - {name}')
        plt.xlabel('Year')
        plt.ylabel('Number of Songs')
        plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Also show stacked bar by year
        year_emotion = songs.groupby(['year', 'label_emotion']).size().unstack(fill_value=0)
        
        year_emotion.plot(kind='bar', stacked=True, figsize=(14, 6), colormap='Set2')
        plt.title(f'Songs by Year - {name}')
        plt.xlabel('Year')
        plt.ylabel('Number of Songs')
        plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        return songs
    
    def analyze_famous_people(self, people: List[str], 
                              show_plots: bool = True) -> pd.DataFrame:
        """Analyze emotion distribution for a list of famous people."""
        composerdf = self._explode_column(self.dataset, 'composers', exclude_unknown=False)
        lyricistdf = self._explode_column(self.dataset, 'lyricists', exclude_unknown=False)
        
        # Filter for famous people
        famous_composers = composerdf[composerdf['composers'].isin(people)]
        famous_lyricists = lyricistdf[lyricistdf['lyricists'].isin(people)]
        
        # Combine - keep title to properly deduplicate songs, not emotion combinations
        famous_all = pd.concat([
            famous_composers.rename(columns={'composers': 'artist'})[['title', 'artist', 'label_emotion']],
            famous_lyricists.rename(columns={'lyricists': 'artist'})[['title', 'artist', 'label_emotion']]
        ]).drop_duplicates(subset=['title', 'artist'])
        
        print("Famous people found in dataset:")
        print(famous_all['artist'].value_counts())
        
        if show_plots and len(famous_all) > 0:
            top_famous = famous_all['artist'].value_counts().head(10).index.tolist()
            
            # Distribution plot
            emotion_data = famous_all.groupby(['artist', 'label_emotion']).size().reset_index(name='count')
            emotion_data_top = emotion_data[emotion_data['artist'].isin(top_famous)]
            
            plt.figure(figsize=(14, 8))
            sns.barplot(data=emotion_data_top, x='artist', y='count', hue='label_emotion')
            plt.title('Emotion Distribution - Famous Vietnamese Musicians')
            plt.xlabel('Musician')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            
            # Percentage plot
            emotion_pct = famous_all.groupby(['artist', 'label_emotion']).size().unstack(fill_value=0)
            emotion_pct = emotion_pct.loc[top_famous]
            emotion_pct_norm = emotion_pct.div(emotion_pct.sum(axis=1), axis=0) * 100
            
            emotion_pct_norm.plot(kind='barh', stacked=True, figsize=(12, 8), colormap='Set2')
            plt.title('Emotion Percentage - Famous Vietnamese Musicians')
            plt.xlabel('Percentage (%)')
            plt.ylabel('Musician')
            plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
        
        # Summary table
        summary = famous_all.groupby(['artist', 'label_emotion']).size().unstack(fill_value=0)
        summary['total'] = summary.sum(axis=1)
        summary['sadness_ratio'] = (summary['sadness'] / summary['total'] * 100).round(2)
        summary['other_ratio'] = (100 - summary['sadness_ratio']).round(2)
        summary = summary.sort_values('total', ascending=False)
        
        cols = ['sadness', 'joy', 'anger', 'fear', 'total', 'sadness_ratio', 'other_ratio']
        cols = [c for c in cols if c in summary.columns]
        return summary[cols]
    
    # ==================== YEAR ANALYSIS (FULL) ====================
    
    def analyze_years(self, show_plots: bool = True) -> pd.DataFrame:
        """Complete year/decade analysis with all plots."""
        yeardf = self.prepare_year_data()
        
        print(f"Year data info:")
        print(f"Total records: {len(yeardf)}")
        print(f"Year range: {yeardf['year'].min()} - {yeardf['year'].max()}")
        
        if show_plots:
            self.plot_emotion_by_year(yeardf)
            self.plot_emotion_by_decade(yeardf)
            self.plot_sadness_ratio_by_decade(yeardf)
            
        return self.summary_by_decade(yeardf)
