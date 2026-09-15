# ==============================================================================
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
# ==============================================================================

@dataclass(slots  =  True)
class ModelParams:
    threshold:      float              =  0.5
    normalize:      bool               =  True
    dominant_only:  bool               =  True
    extra_params:   dict[str, object]  =  field(default_factory  =  dict)

# ==============================================================================

@dataclass(slots  =  True)
class ModelResult:
    emotions:  str
    scores:    dict[str, float | int]
    metadata:  dict[str, object]

# ==============================================================================

class BaseModel(ABC):
    def __init__(
            self,
            params: ModelParams | None  =  None
        ):
        self.params  =  params or ModelParams()

    @abstractmethod
    def predict_single(
            self, 
            tokens: np.ndarray
        ) -> ModelResult | None:
        pass

    @abstractmethod
    def predict_batch(
            self, 
            token_list: np.ndarray | list[np.ndarray]
        ) -> list[ModelResult] | None:
        pass

    def predict(
            self, 
            tokens: np.ndarray | list[np.ndarray]
        ) -> ModelResult | list[ModelResult] | None:
        if isinstance(tokens, list):
            return self.predict_batch(token_list  =  tokens)

        if not isinstance(tokens, np.ndarray):
            raise TypeError("Args must be numpy.ndarray or list[numpy.ndarray].")

        if tokens.ndim == 2 or tokens.dtype == object:
            return self.predict_batch(token_list  =  tokens)
        elif tokens.ndim == 1:
            return self.predict_single(tokens  =  tokens)
        else:
            raise ValueError(
                "Only single token array (1 dim) "
                "and a batch of token arrays (2 dim or list) are allowed."
            )

# ==============================================================================

class ClassicalModel(BaseModel):
    def predict_single(
            self, 
            tokens: np.ndarray
        ) -> ModelResult | None:
        return None

    def predict_batch(
            self, 
            token_list: np.ndarray | list[np.ndarray]
        ) -> list[ModelResult] | None:
        return None

# ==============================================================================
