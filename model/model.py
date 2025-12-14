from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseModel(ABC):
    @abstractmethod
    def predict(self, tokens: List[str]) -> Dict[str, Any]:
        pass