from .model import BaseModel
from typing import List, Optional
from dataclasses import dataclass

class _PhoBertParams:
    def __init__(
        self, 
        model_name: str = "vinai/phobert-base"
    ):
        self.model_name = model_name

class PhoBertModel(_PhoBertParams):
    def __init__(
        self, 
        model_name: str = "vinai/phobert-base"
    ):
        super().__init__(model_name)

class PhoBert(BaseModel, _PhoBertParams):
    def __init__(
        self, 
        model_name: str = "vinai/phobert-base"
    ):
        super().__init__(model_name)

    def predict(self, tokens: list) -> PhoBertModel:
        pass

    def predict_batch(self, batch_tokens: list) -> List[PhoBertModel]:
        pass



