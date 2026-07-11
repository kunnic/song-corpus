import numpy as np

from song import Song
from emolex import (
    EmolexWord,
    ANGER,
    JOY,
    TRUST,
)

def main():

    song = Song(
        name="Xiaoxin",
        lyrics="I just wanna let you know, wo zhi shi ai de xiao xin."
    )

    print("Song")
    print(f"  Title : {song.name}")
    print(f"  Lyrics: {song.lyrics}\n")

    word = EmolexWord(
        word="happy",
        sentiments=np.uint8(0b00000001),
        emotions=np.uint8(JOY | TRUST)
    )

    print("EmoLex Entry")
    print(f"  Word      : {word.word}")
    print(f"  Positive? : {word.is_positive()}")
    print(f"  Joy?      : {word.has_emotion(JOY)}")
    print(f"  Trust?    : {word.has_emotion(TRUST)}")
    print(f"  Anger?    : {word.has_emotion(ANGER)}")


if __name__ == "__main__":
    main()