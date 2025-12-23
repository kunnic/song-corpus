from dataclasses import dataclass, asdict
from typing import List, Optional
from tqdm import tqdm
from pandas import DataFrame
import pandas as pd
import json

from .model import BaseModel

class _LexiconParams:
    def __init__(
        self, 
        exclude_labels: Optional[List[str]] = None,
        emotions: Optional[List[str]] = None
    ):
        self.exclude_labels = exclude_labels if exclude_labels is not None else ['positive', 'negative']
        self.emotions = emotions if emotions is not None else [
            'positive', 'negative',
            'anger', 'anticipation', 'disgust', 'fear',
            'joy', 'sadness', 'surprise', 'trust'
        ]

@dataclass
class LexiconModel:
    scores: dict
    top_emotion: str
    word_count: int

    def to_flat_dict(self):
        return {
            'top_emotion': self.top_emotion,
            'word_count': self.word_count,
            **self.scores
        }
    
    def to_string(self):
        return json.dumps(self.to_flat_dict(), ensure_ascii=False)

class Lexicon(BaseModel, _LexiconParams):
    def __init__(
        self, 
        lexicon_dict: dict,
        exclude_labels: Optional[List[str]] = None,
        emotions: Optional[List[str]] = None
    ):
        super().__init__(exclude_labels, emotions)
        self.lexicon = lexicon_dict

    def predict(self, tokens: list) -> LexiconModel:
        scores = {emotion: 0 for emotion in self.emotions}
        count = 0

        for word in tokens:
            if word in self.lexicon:
                word_data = self.lexicon[word]
                count += 1
                for emotion in self.emotions:
                    score = word_data.get(emotion, 0)
                    if isinstance(score, (int, float)):
                        scores[emotion] += score

        valid_scores = {
            k: v for k, v in scores.items() 
            if k not in self.exclude_labels
        }

        if valid_scores and max(valid_scores.values()) > 0:
            winner = max(valid_scores, key=valid_scores.get)
        else:
            winner = 'neutral'
            
        return LexiconModel(
            scores=scores,
            top_emotion=winner,
            word_count=count
        )
    
    def predict_dataset(self, dataset: DataFrame, tokenized_col_name: str) -> DataFrame:
        tokens_series = dataset[tokenized_col_name]

        results_objects = [
            self.predict(tokens) 
            for tokens in tqdm(tokens_series, desc="Processing")
        ]
        
        results_data = [obj.to_flat_dict() for obj in results_objects]
        results_df = pd.DataFrame(results_data)
        
        final_df = pd.concat([dataset.reset_index(drop=True), results_df], axis=1)
        return final_df