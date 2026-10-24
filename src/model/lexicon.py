# ───────────────────────────────────────────────────────────────────
import numpy as np

from model.model import BaseModel, ModelResult
from dataloader.emolex import EmolexDictionary
# ───────────────────────────────────────────────────────────────────
class LexiconModel(BaseModel):
    def __init__(self, emolex: EmolexDictionary):
        self.emolex = emolex

    def predict_single(self, tokens: np.ndarray) -> ModelResult:

        indices = self.emolex.get_index(tokens.tolist())

        total_tokens   = len(tokens)
        matched_tokens = len(indices)

        sentiment_names = self.emolex.sentiments or ['positive', 'negative']

        if matched_tokens == 0:
            return ModelResult(
                emotions = 'neutral',
                scores   = {e: 0.0 for e in self.emolex.emotions},
                metadata = {
                    'total_tokens':      total_tokens,
                    'matched_tokens':    0,
                    'coverage':          0.0,
                    'raw_counts':        [0] * len(self.emolex.emotions),
                    'emotion_counts':    {e: 0 for e in self.emolex.emotions},
                    'sentiment_counts':  {s: 0 for s in sentiment_names},
                    'sentiment_scores':  {s: 0.0 for s in sentiment_names},
                    'dominant_sentiment': 'neutral',
                },
            )

        token_matrix = self.emolex.matrix[indices]
        emotion_vector = token_matrix.sum(axis=0).astype(float)

        vector_sum = emotion_vector.sum()
        if vector_sum > 0:
            normalized = emotion_vector / vector_sum
            dominant_idx = int(np.argmax(normalized))
            dominant     = self.emolex.emotions[dominant_idx]
        else:
            normalized = emotion_vector
            dominant   = 'neutral'

        scores = {
            name: round(float(val), 4)
            for name, val in zip(self.emolex.emotions, normalized)
        }
        raw_counts_list = emotion_vector.astype(int).tolist()
        emotion_counts = {
            name: int(cnt)
            for name, cnt in zip(self.emolex.emotions, raw_counts_list)
        }

        # Compute sentiment statistics if sentiment_matrix is available
        sentiment_counts: dict[str, int] = {s: 0 for s in sentiment_names}
        sentiment_scores: dict[str, float] = {s: 0.0 for s in sentiment_names}
        dominant_sentiment = 'neutral'

        if self.emolex.sentiment_matrix is not None:
            sent_vec = self.emolex.sentiment_matrix[indices].sum(axis=0).astype(float)
            sent_sum = sent_vec.sum()
            sent_norm = sent_vec / sent_sum if sent_sum > 0 else sent_vec

            for name, cnt, val in zip(sentiment_names, sent_vec.astype(int), sent_norm):
                sentiment_counts[name] = int(cnt)
                sentiment_scores[name] = round(float(val), 4)

            if sent_sum > 0:
                if sent_vec[0] > sent_vec[1]:
                    dominant_sentiment = sentiment_names[0]
                elif sent_vec[1] > sent_vec[0]:
                    dominant_sentiment = sentiment_names[1]
                else:
                    dominant_sentiment = 'mixed'

        return ModelResult(
            emotions = dominant,
            scores   = scores,
            metadata = {
                'total_tokens':       total_tokens,
                'matched_tokens':     matched_tokens,
                'coverage':           round(matched_tokens / total_tokens, 4) if total_tokens > 0 else 0.0,
                'raw_counts':         raw_counts_list,
                'emotion_counts':     emotion_counts,
                'sentiment_counts':   sentiment_counts,
                'sentiment_scores':   sentiment_scores,
                'dominant_sentiment': dominant_sentiment,
            },
        )

    def predict_batch(
            self,
            token_list: np.ndarray | list[np.ndarray]
        ) -> list[ModelResult]:
        return [
            self.predict_single(tokens)
            for tokens in token_list
        ]
# ───────────────────────────────────────────────────────────────────
