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
            token_list: np.ndarray | list[np.ndarray]
        ) -> list[ModelResult]:
        pass

    def predict(
            self, 
            tokens: np.ndarray | list[np.ndarray]
        ) -> ModelResult | list[ModelResult]:
        
        if isinstance(tokens, list):
            return self.predict_batch(token_list=tokens)

        if not isinstance(tokens, np.ndarray):
            raise TypeError("Args must be numpy.ndarray or list[numpy.ndarray].")

        if tokens.ndim == 2 or tokens.dtype == object:
            return self.predict_batch(token_list=tokens)
        elif tokens.ndim == 1:
            return self.predict_single(tokens=tokens)
        else:
            raise ValueError(
                "Only single token array (1 dim) "
                "and a batch of token arrays (2 dim or list) are allowed."
            )
# ───────────────────────────────────────────────────────────────────
