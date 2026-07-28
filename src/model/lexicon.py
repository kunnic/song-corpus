# ───────────────────────────────────────────────────────────────────
from model import BaseModel, ModelResult
# ───────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────
class LexiconModel(BaseModel):
    def __init__(self):
        pass

    def predict_single(self, tokens: np.ndarray) -> ModelResult:
        '''
            using the emolex, predict the emotions of the line.
            - args:
            + input: tokens: the input line in 
        '''
        

    # Implement later if optimization is possible,
    #   + multithreading or other way to make
    #       it run better than looping.
    def predict_batch(self, 
            token_list: np.ndarray
        ) -> list[ModelResult]:
        pass

    def predict(self, 
            tokens: np.ndarray
        ) -> ModelResult | list[ModelResult]:
        if isinstance(tokens, np.ndarray):
            return [
                self.predict_single(tokens = token) 
                for token in tokens
            ]
        return predict_single(tokens = tokens)
# ───────────────────────────────────────────────────────────────────