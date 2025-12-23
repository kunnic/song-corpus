from .dataloader import DataLoader
import pandas as pd
from typing import List, Optional
from underthesea import word_tokenize
import ast

class CorpusDataLoader(DataLoader):
    def __init__(self, file_path: str, limit: Optional[int] = None, columns: Optional[List[str]] = None):
        super().__init__(file_path)
        self.limit = limit
        self.columns = columns if columns is not None else [
            'title', 'composers', 
            'lyricists', 'year', 
            'genres', 'lyrics', 
            'urls', 'source', 'note'
        ]
        self.str_cols = ['title', 'lyrics', 'source', 'note']
        self.list_cols = ['composers', 'lyricists', 'urls', 'genres']
        self.load()

    def load(self) -> pd.DataFrame:
        df = self._load_dataframe()
        if self.limit is not None:
            df = df.head(self.limit)
        
        list_cols_to_parse = [c for c in self.list_cols if c in df.columns]
        for col in list_cols_to_parse:
            df[col] = df[col].apply(self._parse_list)
        
        if 'lyrics_tokenized' in df.columns:
            df['lyrics_tokenized'] = df['lyrics_tokenized'].apply(self._parse_list)
        
        self._data = df
        print(f"Corpus loaded: {len(self._data)} records.")
        return self._data
    
    def _parse_list(self, value):
        if pd.isna(value):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except:
                return []
        return []
    
    def _load_dataframe(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.file_path)
            
            if df.empty:
                raise ValueError(f"File {self.file_path} contains no data")
            
            if len(df.columns) == len(self.columns):
                df.columns = self.columns
            else:
                print(f"Warning: Column count mismatch. Expected {len(self.columns)}, got {len(df.columns)}")
            
            existing_str_cols = [c for c in self.str_cols if c in df.columns]
            df[existing_str_cols] = df[existing_str_cols].astype(str)
            
            if 'year' in df.columns:
                df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(-1).astype(int)
            
            if 'title' in df.columns:
                df['title'] = df['title'].str.strip()
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.file_path}: {str(e)}")

class LyricsTokenizer:
    def __init__(self, source_col: str = 'lyrics', target_col: str = 'lyrics_tokenized'):
        self.source_col = source_col
        self.target_col = target_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.source_col not in df.columns:
            raise ValueError(f"Column '{self.source_col}' not found in dataframe.")
        
        print(f"Tokenizing '{self.source_col}' column...")
        
        df = df.copy()
        df[self.target_col] = df[self.source_col].apply(
            lambda text: word_tokenize(str(text).lower())
        )
        
        print("Tokenization complete.")
        return df