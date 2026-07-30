# ───────────────────────────────────────────────────────────────────
import numpy as np

from model import BaseModel, ModelResult
from emolex import EmolexDictionary
# ───────────────────────────────────────────────────────────────────
class LexiconModel(BaseModel):
    def __init__(self, emolex: EmolexDictionary):
        self.emolex = emolex

    def predict_single(self, tokens: np.ndarray) -> ModelResult:

        indices = self.emolex.get_index(tokens.tolist())

        total_tokens   = len(tokens)
        matched_tokens = len(indices)

        if matched_tokens == 0:
            return ModelResult(
                emotions = 'neutral',
                scores   = {e: 0.0 for e in self.emolex.emotions},
                metadata = {
                    'total_tokens':   total_tokens,
                    'matched_tokens': 0,
                    'coverage':       0.0,
                },
            )

        token_matrix = self.emolex.matrix[indices]

        emotion_vector = token_matrix.sum(axis=0).astype(float)

        vector_sum = emotion_vector.sum()
        if vector_sum > 0:
            normalized = emotion_vector / vector_sum
        else:
            normalized = emotion_vector

        dominant_idx = np.argmax(normalized)
        dominant     = self.emolex.emotions[dominant_idx]

        scores = {
            name: round(float(val), 4)
            for name, val in zip(self.emolex.emotions, normalized)
        }

        return ModelResult(
            emotions = dominant,
            scores   = scores,
            metadata = {
                'total_tokens': total_tokens,
                'matched_tokens': matched_tokens,
                'coverage': round(matched_tokens / total_tokens, 4),
                'raw_counts': emotion_vector.astype(int).tolist(),
            },
        )

    def predict_batch(
            self,
            token_list: np.ndarray
        ) -> list[ModelResult]:
        return [
            self.predict_single(tokens)
            for tokens in token_list
        ]
# ───────────────────────────────────────────────────────────────────