# ───────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod
from dataclasses import dataclass
# ───────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ModelResult(ABC):
    emotions: str

class BaseModel(ABC):
    @abstractmethod
    def predict(self, 
            tokens: list[str] | list[list[str]]
        ) -> ModelResult | list[ModelResult]:
        raise NotImplementedError