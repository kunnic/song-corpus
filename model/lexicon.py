# ───────────────────────────────────────────────────────────────────
from model import BaseModel, ModelResult
# ───────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────
class LexiconModel(BaseModel):
    def __init__(self):
        pass

    def predict_single(self, tokens: list[str]) -> ModelResult:
        pass

    # Implement later if optimization is possible,
    #   + multithreading or other way to make
    #       it run better than looping.
    def predict_batch(self, 
            token_list: list[list[str]]
        ) -> list[ModelResult]:
        pass

    def predict(self, 
            tokens: list[str] | list[list[str]]
        ) -> ModelResult | list[ModelResult]:
        if isinstance(tokens, list):
            return [
                self.predict_single(tokens = token) 
                for token in tokens
            ]
        return predict_single(tokens = tokens)

    # if isinstance(image, list):
    #         return [self._recognize_single(item) for item in image]
    #     return self._recognize_single(image)
    
# ───────────────────────────────────────────────────────────────────