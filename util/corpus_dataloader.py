from .dataloader import DataLoader
import pandas as pd
from typing import Dict

class LexiconDataLoader(DataLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.load()

    def load(self) -> Dict[str, Dict[str, int]]:
        return None
    
    def _load_dataframe(self) -> pd.DataFrame:
        return None