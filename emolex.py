# ───────────────────────────────────────────────────────────────────
from dataclasses import dataclass
import numpy as np
# ───────────────────────────────────────────────────────────────────
ANGER        = 1 << 0  # 00000001 (1)
ANTICIPATION = 1 << 1  # 00000010 (2)
DISGUST      = 1 << 2  # 00000100 (4)
FEAR         = 1 << 3  # 00001000 (8)
JOY          = 1 << 4  # 00010000 (16)
SADNESS      = 1 << 5  # 00100000 (32)
SURPRISE     = 1 << 6  # 01000000 (64)
TRUST        = 1 << 7  # 10000000 (128)
# ───────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class EmolexWord:
    '''
        A word contains 3 attributes:
            + the word itself
            + emotions values
            + sentiments values

        Emotions list mapping:
            [angr, atcp, dsgt, fear, joy, sad, sprs, trst]

        Each of the 8 bits represents one emotion:
            00000001 -> anger is set to 1 (present).

        The same logic applies to the sentiment mapping.
    '''
    word: str
    sentiments: np.uint8
    emotions: np.uint8
        
    def is_positive(self) -> bool:
        '''
            - return True if the sentimental value is positive.
            - args:
                + output: bool value if it's positive or else.
        '''
        return bool(self.sentiments & np.uint8(1))

    def has_emotion(self, emotion: np.uint8) -> bool:
        '''
            - return True if the specified emotion bit is set.
            - args:
                + input: emotion e: a numpy number uint8.
                + output: bool value if e exist or not.
        '''
        return bool(self.emotions & emotion)