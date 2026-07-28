# ───────────────────────────────────────────────────────────────────
from dataclasses import dataclass
import numpy as np
# ───────────────────────────────────────────────────────────────────
ANGER        = 1 << 0 
ANTICIPATION = 1 << 1 
DISGUST      = 1 << 2 
FEAR         = 1 << 3 
JOY          = 1 << 4 
SADNESS      = 1 << 5 
SURPRISE     = 1 << 6 
TRUST        = 1 << 7
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