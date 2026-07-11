from abc    import ABC, abstractmethod

class ModelResult(ABC):
    pass

class BaseModel(ABC):
    @abstractmethod
    def predict(self, tokens: list[str] | list[list[str]]) -> ModelResult:
        raise NotImplementedError