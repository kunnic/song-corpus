# ───────────────────────────────────────────────────────────────────
from dataclasses import dataclass

import numpy as np
# ───────────────────────────────────────────────────────────────────
@dataclass(slots = True)
class CorpusInstance():
    title: str
    lyrics_in_string: str
    lyrics_in_token: np.ndarray

@dataclass(slots = True)
class CorpusData():
    instances: list[CorpusInstance]
    
    @property
    def total_songs(self) -> int:
        return len(self.instances)
# ───────────────────────────────────────────────────────────────────