# ==============================================================================
import numpy as np

from model.model import BaseModel, ModelParams, ModelResult
# ==============================================================================

class BiLSTMModel(BaseModel):
    def __init__(
            self,
            embedding_dim: int  =  300,
            hidden_dim: int     =  128,
            params: ModelParams | None  =  None
        ):
        super().__init__(params  =  params)
        self.embedding_dim  =  embedding_dim
        self.hidden_dim     =  hidden_dim

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
