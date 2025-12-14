from .dataloader import DataLoader
import pandas as pd
from typing import Dict
from underthesea import word_tokenize

CORPUS_COLUMNS = [
    'title', 'composers', 
    'lyricists', 'year', 
    'genres', 'lyrics', 
    'urls', 'source', 'note'
]

class CorpusDataLoader(DataLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.load()

    def load(self) -> pd.DataFrame:
        df = self._load_dataframe()
        self._data = df
        print(f"Corpus loaded: {len(self._data)} records.")
        return self._data
    
    def _load_dataframe(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.file_path)
            
            if df.empty:
                raise ValueError(f"File {self.file_path} contains no data")
            
            if len(df.columns) == len(CORPUS_COLUMNS):
                df.columns = CORPUS_COLUMNS
            else:
                print(f"Warning: Column count mismatch. Expected {len(CORPUS_COLUMNS)}, got {len(df.columns)}")
            
            # Cast string columns
            str_cols = ['title', 'composers', 'lyricists', 'genres', 'lyrics', 'urls', 'source', 'note']
            for col in str_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            
            # Cast year to integer, filling NaN with -1
            if 'year' in df.columns:
                df['year'] = df['year'].fillna(-1).astype(int)
            
            # Strip and lowercase title for consistency
            if 'title' in df.columns:
                df['title'] = df['title'].str.strip()
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.file_path}: {str(e)}")
    
    def tokenize_lyrics(self, column: str = 'lyrics') -> pd.Series:
        if column not in self._data.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        
        print(f"Tokenizing {column} column...")
        tokenized = self._data[column].apply(
            lambda text: word_tokenize(str(text)) if pd.notna(text) else ""
        )
        print(f"Tokenization complete.")
        return tokenized
    
    def add_tokenized_column(self, source_column: str = 'lyrics', 
                            target_column: str = 'lyrics_tokenized') -> None:
        self._data[target_column] = self.tokenize_lyrics(source_column)
        print(f"Added '{target_column}' column to dataframe.")