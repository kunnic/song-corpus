# ==============================================================================
import numpy as np

from model.model import BaseModel, ModelParams, ModelResult
# ==============================================================================

class PhoBERTModel(BaseModel):
    def __init__(
            self,
            checkpoint_name: str  =  "vinai/phobert-base-v2",
            params: ModelParams | None  =  None
        ):
        super().__init__(params  =  params)
        self.checkpoint_name  =  checkpoint_name

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
