from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pandas import DataFrame

class BaseModel(ABC):
    @abstractmethod
    def predict(self, tokens: List[str]) -> Any:
        pass
    @abstractmethod
    def predict_dataset(self, dataframe: DataFrame, tokenized_text_column: str) -> DataFrame:
        pass