from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm
from pandas import DataFrame

class LexiconParams:
    def __init__(self, exclude_labels: Optional[List[str]] = None):
        self.exclude_labels = exclude_labels
        if self.exclude_labels is None:
            self.exclude_labels = ['positive', 'negative']

@dataclass
class LexiconModel:
    scores: dict
    top_emotion: str
    word_count: int

class Lexicon:

    def __init__(self, lexicon_dict: dict, params: LexiconParams):
        self.lexicon = lexicon_dict
        self.params = params
        self.emotions = [
            'positive', 'negative',
            'anger', 'anticipation', 'disgust', 'fear',
            'joy', 'sadness', 'surprise', 'trust'
        ]

    def predict_single(self, tokens: list) -> LexiconModel:
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
            if k not in self.params.exclude_labels
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
    
    # def predict_dataset(self, dataset: List[List[str]]) -> List[LexiconModel]:
    #     results = []

    #     for tokens in tqdm(dataset, desc="Predict on dataset"):
    #         result = self.predict_single(tokens)
    #         results.append(result)
            
    #     return results

    def predict_dataset(self, dataset: DataFrame, tokenized_text_column: str) -> DataFrame:
        results = {
            'top_emotion': [],
            'word_count': []
        }
        for emotion in self.emotions:
            results[f'score_{emotion}'] = []

        for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Predict on dataset"):
            tokens = row[tokenized_text_column]
            result = self.predict_single(tokens)

            results['top_emotion'].append(result.top_emotion)
            results['word_count'].append(result.word_count)
            for emotion in self.emotions:
                results[f'score_{emotion}'].append(result.scores.get(emotion, 0))

        output = dataset.copy()
        for col, values in results.items():
            output[col] = values
        return output