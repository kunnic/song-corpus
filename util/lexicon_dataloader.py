from .dataloader import DataLoader
import pandas as pd
from typing import Dict, List, Optional

class LexiconDataLoader(DataLoader):
    def __init__(self, file_path: str, columns: Optional[List[str]] = None):
        super().__init__(file_path)
        
        self.columns = columns if columns is not None else [
            'english', 'vietnamese',
            'positive', 'negative',
            'anger', 'anticipation', 'disgust', 'fear',
            'joy', 'sadness', 'surprise', 'trust',
            'sum'
        ]
        self.text_cols = ['english', 'vietnamese']
        self.score_cols = self.columns[2:]
        
        self.load()

    def load(self) -> Dict[str, Dict[str, int]]:
        df = self._load_dataframe()
        df['id'] = range(len(df))
        self._data = df.set_index('vietnamese').to_dict(orient='index')
        print(f"Lexicon loaded: {len(self._data)} words.")
        return self._data
    
    def _load_dataframe(self) -> pd.DataFrame:
        try:
            df = pd.read_excel(self.file_path)
            
            if df.empty:
                raise ValueError(f"File {self.file_path} contains no data")
            
            if len(df.columns) == len(self.columns):
                df.columns = self.columns
            else:
                print(f"Warning: Column count mismatch. Expected {len(self.columns)}, got {len(df.columns)}")
            
            existing_text_cols = [c for c in self.text_cols if c in df.columns]
            df[existing_text_cols] = df[existing_text_cols].astype(str)
            
            if 'vietnamese' in df.columns:
                df['vietnamese'] = df['vietnamese'].str.lower().str.strip()
            
            existing_score_cols = [c for c in self.score_cols if c in df.columns]
            df[existing_score_cols] = df[existing_score_cols].fillna(0).astype(int)
            
            df = df.drop_duplicates(subset=['vietnamese'], keep='first')
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.file_path}: {str(e)}")