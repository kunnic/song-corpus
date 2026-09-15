# ==============================================================================
import numpy as np

from model.model import BaseModel, ModelParams, ModelResult
# ==============================================================================

class LLMModel(BaseModel):
    def __init__(
            self,
            model_name: str    =  "gemini-1.5-pro",
            temperature: float  =  0.0,
            params: ModelParams | None  =  None
        ):
        super().__init__(params  =  params)
        self.model_name    =  model_name
        self.temperature   =  temperature

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
