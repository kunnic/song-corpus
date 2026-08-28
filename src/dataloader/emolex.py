# ==============================================================================
from dataclasses import dataclass
import numpy as np
# ==============================================================================

ANGER         =  1 << 0
ANTICIPATION  =  1 << 1
DISGUST       =  1 << 2
FEAR          =  1 << 3
JOY           =  1 << 4
SADNESS       =  1 << 5
SURPRISE      =  1 << 6
TRUST         =  1 << 7

# ==============================================================================

@dataclass(slots  =  True, frozen  =  True)
class EmolexWord:
    word:        str
    sentiments:  np.uint8
    emotions:    np.uint8

    def is_positive(self) -> bool:
        return bool(self.sentiments & np.uint8(1))

    def is_negative(self) -> bool:
        return bool(self.sentiments & np.uint8(2))

    def has_emotion(
            self, 
            emotion: np.uint8
        ) -> bool:
        return bool(self.emotions & emotion)

# ==============================================================================

@dataclass(slots  =  True, frozen  =  True)
class EmolexDictionary:
    word_to_index:     dict[str, int]
    matrix:            np.ndarray
    emotions:          list[str]
    sentiment_matrix:  np.ndarray | None  =  None
    sentiments:        list[str] | None   =  None

    def get_index(
            self, 
            tokens: list[str]
        ) -> list[int]:
        return [
            self.word_to_index[token]
            for token in tokens if token in self.word_to_index
        ]

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_index)

# ==============================================================================
