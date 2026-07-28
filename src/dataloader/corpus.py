# ───────────────────────────────────────────────────────────────────
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
import re

from underthesea import word_tokenize

from emolex import EmolexDictionary

VNESE_REGEX = re.compile(
    r'[^a-zA-Z0-9\s'
    r'àáảãạăắằẳẵặâấầẩẫậ'
    r'èéẻẽẹêếềểễệ'
    r'ìíỉĩị'
    r'òóỏõọôốồổỗộơớờởỡợ'
    r'ùúủũụưứừửữự'
    r'ỳýỷỹỵđ'
    r'ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ'
    r'ÈÉẺẼẸÊẾỀỂỄỆ'
    r'ÌÍỈĨỊ'
    r'ÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ'
    r'ÙÚỦŨỤƯỨỪỬỮỰ'
    r'ỲÝỶỸỴĐ]'
)
# ───────────────────────────────────────────────────────────────────
@dataclass(slots = True)
class CorpusInstance():
    title: str
    lyrics_in_string: str
    lyrics_in_token: np.ndarray

@dataclass(slots = True)
class CorpusData():
    instances: list[CorpusInstance]
    
    @property
    def total_songs(self) -> int:
        return len(self.instances)
# ───────────────────────────────────────────────────────────────────
class Dataloader(ABC):
    @abstractmethod
    def load(self) -> CorpusData:
        pass

    @staticmethod
    def load_emolex(
            words_list:     list[str], 
            scores_matrix:  list[list[int]], 
            feature_names:  list[str]
        ):
        
        word_to_index = {
            str(word): index 
            for index, word in enumerate(words_list)
        }
        numpy_matrix = np.array(scores_matrix, dtype = np.uint8)

        return EmolexDictionary(
            word_to_index, numpy_matrix, feature_names
        )

class CorpusDataloader(Dataloader):
    def __init__(
            self, 
            default_path: str = None, 
            min_word_count: int = 30
        ):
        self.default_path = default_path
        self.min_word_count = min_word_count

    def _clean_lyrics(self, text: str) -> str:
        if pd.isna(text) or not str(text).strip():
            return ""
        
        text = str(text).replace('\n', ' ')
        text = VNESE_REGEX.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    def load(self, custom_path: str = None) -> CorpusData:
        target_path = custom_path if custom_path else self.default_path

        if not target_path:
            raise ValueError("Path invalid.")


        file_path = Path(target_path)
        if file_path.suffix.lower() == '.csv':
            try:
                df = pd.read_csv(
                    file_path, 
                    encoding='utf-8'
                )
            except pd.errors.ParserError:
                df = pd.read_csv(
                    file_path, 
                    encoding='utf-8', 
                    sep='\t'
                )
        elif file_path.suffix.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Only .csv/.xlsx are supported")

        instances = []
        skipped_count = 0

        for index, row in df.iterrows():
            title = str(row.get('title', f'Unknown_{index}'))
            raw_lyrics = row['lyrics']

            cleaned_lyrics = self._clean_lyrics(raw_lyrics)

            tokens_list = word_tokenize(cleaned_lyrics)

            if len(tokens_list) < self.min_word_count:
                skipped_count += 1
                continue

            lyrics_in_token = np.array(tokens_list, dtype=str)

            instance = CorpusInstance(
                title = title,
                lyrics_in_string = cleaned_lyrics,
                lyrics_in_token  = lyrics_in_token
            )
            instances.append(instance)

        if skipped_count > 0:
            print(f"Skipped {skipped_count} songs.")

        return CorpusData(instances = instances)
# ───────────────────────────────────────────────────────────────────