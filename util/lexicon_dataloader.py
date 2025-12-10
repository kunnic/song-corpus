from .dataloader import DataLoader
import pandas as pd
from typing import Dict

LEXICON_COLUMNS = [
    'english', 'vietnamese',
    'positive', 'negative',
    'anger', 'anticipation', 'disgust', 'fear',
    'joy', 'sadness', 'surprise', 'trust',
    'sum'
]

class LexiconDataLoader(DataLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path)
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
            
            if len(df.columns) == len(LEXICON_COLUMNS):
                df.columns = LEXICON_COLUMNS
            else:
                print(f"Warning: Column count mismatch. Expected {len(LEXICON_COLUMNS)}, got {len(df.columns)}")
            
            cols_to_str = ['english', 'vietnamese']
            df[cols_to_str] = df[cols_to_str].astype(str)
            df['vietnamese'] = df['vietnamese'].str.lower().str.strip()
            
            score_cols = LEXICON_COLUMNS[2:]
            df[score_cols] = df[score_cols].fillna(0).astype(int)
            
            # keep only unique vietnamese words for indexing
            df = df.drop_duplicates(subset=['vietnamese'], keep=False)
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.file_path}: {str(e)}")
