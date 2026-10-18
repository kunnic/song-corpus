# ==============================================================================
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataloader.corpus import CorpusData
from dataloader.emolex import EmolexDictionary
# ==============================================================================

@dataclass(slots  =  True)
class ZipfianBiasResult:
    total_lyric_tokens:     int
    total_emotion_tokens:   int
    unique_matched_words:   int
    pareto_80_word_count:   int
    pareto_80_token_share:  float
    joy_anticipation_share: float
    top_words_df:           pd.DataFrame
    lexicon_skew_df:        pd.DataFrame

# ==============================================================================

class ZipfianBiasAnalyzer:
    def __init__(
            self,
            emolex: EmolexDictionary
        ):
        self.emolex  =  emolex

    def analyze(
            self,
            corpus: CorpusData,
            pareto_threshold: float  =  0.80
        ) -> ZipfianBiasResult:
        word_to_idx   =  self.emolex.word_to_index
        emotions      =  self.emolex.emotions
        matrix        =  self.emolex.matrix

        token_counter: Counter  =  Counter()
        total_tokens            =  0

        for inst in corpus.instances:
            tokens        =  inst.lyrics_in_token
            total_tokens  +=  len(tokens)
            for tok in tokens:
                if tok in word_to_idx:
                    token_counter[tok]  +=  1

        total_emotion_hits    =  sum(token_counter.values())
        unique_matched_words  =  len(token_counter)

        sorted_words  =  token_counter.most_common()
        top_rows      =  []
        cum_hits      =  0
        p80_count     =  0

        for rank, (word, count) in enumerate(sorted_words, start  =  1):
            idx          =  word_to_idx[word]
            emo_bits     =  matrix[idx]
            active_emos  =  [emotions[i] for i, b in enumerate(emo_bits) if b > 0]
            cum_hits     +=  count
            share        =  count / total_emotion_hits if total_emotion_hits > 0 else 0.0
            cum_share    =  cum_hits / total_emotion_hits if total_emotion_hits > 0 else 0.0

            if cum_share <= pareto_threshold and p80_count == 0:
                pass
            elif cum_share > pareto_threshold and p80_count == 0:
                p80_count  =  rank

            top_rows.append({
                "rank":              rank,
                "word":              word,
                "count":             count,
                "share":             round(share, 5),
                "cum_share":         round(cum_share, 5),
                "active_emotions":   ", ".join(active_emos),
                "has_joy_or_antic":  any(e in ("joy", "anticipation") for e in active_emos),
            })

        top_df  =  pd.DataFrame(top_rows)

        if p80_count == 0:
            p80_count  =  len(sorted_words)

        p80_subset        =  top_df.iloc[:p80_count]
        joy_antic_subset  =  p80_subset[p80_subset["has_joy_or_antic"]]
        joy_antic_share   =  (
            joy_antic_subset["count"].sum() / p80_subset["count"].sum()
            if not p80_subset.empty else 0.0
        )

        lex_counts  =  matrix.sum(axis  =  0)
        total_lex   =  int(lex_counts.sum())
        lex_skew_rows  =  [
            {
                "emotion":     emo,
                "word_count":  int(cnt),
                "percentage":  round(float(cnt) / total_lex * 100.0, 2) if total_lex > 0 else 0.0,
            }
            for emo, cnt in zip(emotions, lex_counts)
        ]
        lex_skew_df  =  pd.DataFrame(lex_skew_rows)

        return ZipfianBiasResult(
            total_lyric_tokens      =  total_tokens,
            total_emotion_tokens    =  total_emotion_hits,
            unique_matched_words    =  unique_matched_words,
            pareto_80_word_count    =  p80_count,
            pareto_80_token_share   =  round(p80_subset["count"].sum() / total_emotion_hits, 4) if total_emotion_hits > 0 else 0.0,
            joy_anticipation_share  =  round(float(joy_antic_share), 4),
            top_words_df            =  top_df,
            lexicon_skew_df         =  lex_skew_df,
        )

# ==============================================================================
