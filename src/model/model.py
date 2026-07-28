# ───────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

# ───────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ModelResult:
    emotions:   str
    scores:     dict[str, float | int]
    metadata:   dict[str, object]

class BaseModel(ABC):
    @abstractmethod
    def predict_single(
            self, 
            tokens: np.ndarray
        ) -> ModelResult:
        pass

    @abstractmethod
    def predict_batch(
            self, 
            token_list: np.ndarray
        ) -> list[ModelResult]:
        pass

    def predict(
            self, 
            tokens: np.ndarray
        ) -> ModelResult | list[ModelResult]:
        
        if not isinstance(tokens, np.ndarray):
            raise TypeError("Args must be numpy.ndarray.")

        if tokens.ndim == 2:
            return self.predict_batch(tokens=tokens)
        elif tokens.ndim == 1:
            return self.predict_single(tokens=tokens)
        else:
            raise ValueError(
                "Only single token (1 dim) "
                "and an array of token (2 dim) is allowed."
            )
# ───────────────────────────────────────────────────────────────────