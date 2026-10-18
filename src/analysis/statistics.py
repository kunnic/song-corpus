# ==============================================================================
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dataloader.corpus import CorpusData
from model.model import BaseModel, ModelResult
# ==============================================================================

@dataclass(slots  =  True)
class CorpusStatistics:
    overview:                   dict[str, float | int]
    songs_df:                   pd.DataFrame
    dominant_emotions_df:       pd.DataFrame
    dominant_sentiments_df:     pd.DataFrame
    emotion_distribution_df:    pd.DataFrame
    sentiment_distribution_df:  pd.DataFrame
    emotion_correlation_df:     pd.DataFrame

# ==============================================================================

class CorpusAnalyzer:
    def __init__(
            self, 
            model: BaseModel | None  =  None
        ):
        self.model  =  model

    def analyze(
            self,
            corpus: CorpusData,
            results: list[ModelResult] | None  =  None,
        ) -> CorpusStatistics:
        if results is None:
            if self.model is None:
                raise ValueError("Either pass results or initialize with a BaseModel.")
            token_list  =  [inst.lyrics_in_token for inst in corpus.instances]
            results     =  self.model.predict_batch(token_list  =  token_list)

        if len(results) != len(corpus.instances):
            raise ValueError(
                f"Length mismatch: {len(corpus.instances)} instances vs {len(results)} results."
            )

        if len(results) == 0:
            empty_df  =  pd.DataFrame()
            return CorpusStatistics(
                overview                   =  {"total_songs": 0},
                songs_df                   =  empty_df,
                dominant_emotions_df       =  empty_df,
                dominant_sentiments_df     =  empty_df,
                emotion_distribution_df    =  empty_df,
                sentiment_distribution_df  =  empty_df,
                emotion_correlation_df     =  empty_df,
            )

        emotion_names    =  list(results[0].scores.keys())
        first_sent       =  results[0].metadata.get("sentiment_scores", {})
        sentiment_names  =  list(first_sent.keys()) if isinstance(first_sent, dict) else []

        rows: list[dict[str, Any]]  =  []
        for inst, res in zip(corpus.instances, results):
            meta      =  res.metadata
            total_tok =  int(meta.get("total_tokens", len(inst.lyrics_in_token)))
            match_tok =  int(meta.get("matched_tokens", 0))
            cov       =  float(meta.get("coverage", 0.0))
            dom_sent  =  str(meta.get("dominant_sentiment", "neutral"))

            row: dict[str, Any]  =  {
                "title":               inst.title,
                "total_tokens":        total_tok,
                "matched_tokens":      match_tok,
                "coverage":            cov,
                "dominant_emotion":    res.emotions,
                "dominant_sentiment":  dom_sent,
            }
            for emo, score in res.scores.items():
                row[emo]  =  float(score)
            raw_c  =  meta.get("raw_counts", [])
            for emo, cnt in zip(emotion_names, raw_c):
                row[f"count_{emo}"]  =  int(cnt)

            sent_sc  =  meta.get("sentiment_scores", {})
            if isinstance(sent_sc, dict):
                for sent, s_val in sent_sc.items():
                    row[f"sentiment_{sent}"]  =  float(s_val)

            sent_cnt  =  meta.get("sentiment_counts", {})
            if isinstance(sent_cnt, dict):
                for sent, s_c in sent_cnt.items():
                    row[f"count_{sent}"]  =  int(s_c)

            rows.append(row)

        songs_df         =  pd.DataFrame(rows)
        total_songs      =  len(songs_df)
        total_tokens     =  int(songs_df["total_tokens"].sum())
        total_matched    =  int(songs_df["matched_tokens"].sum())
        neutral_songs    =  int((songs_df["dominant_emotion"] == "neutral").sum())
        emotional_songs  =  total_songs - neutral_songs

        overview  =  {
            "total_songs":            total_songs,
            "emotional_songs":        emotional_songs,
            "neutral_songs":          neutral_songs,
            "neutral_ratio":          round(neutral_songs / total_songs, 4),
            "total_tokens":           total_tokens,
            "total_matched_tokens":   total_matched,
            "corpus_coverage":        round(total_matched / total_tokens, 4) if total_tokens > 0 else 0.0,
            "coverage_mean":          round(float(songs_df["coverage"].mean()), 4),
            "coverage_std":           round(float(songs_df["coverage"].std(ddof  =  0)), 4),
            "coverage_median":        round(float(songs_df["coverage"].median()), 4),
            "coverage_min":           round(float(songs_df["coverage"].min()), 4),
            "coverage_max":           round(float(songs_df["coverage"].max()), 4),
        }

        dom_counts            =  songs_df["dominant_emotion"].value_counts()
        dominant_emotions_df  =  pd.DataFrame({
            "emotion":     dom_counts.index.tolist(),
            "song_count":  dom_counts.values.astype(int).tolist(),
            "percentage":  np.round(dom_counts.values / total_songs * 100.0, 2).tolist(),
        })

        sent_dom_counts         =  songs_df["dominant_sentiment"].value_counts()
        dominant_sentiments_df  =  pd.DataFrame({
            "sentiment":   sent_dom_counts.index.tolist(),
            "song_count":  sent_dom_counts.values.astype(int).tolist(),
            "percentage":  np.round(sent_dom_counts.values / total_songs * 100.0, 2).tolist(),
        })

        total_emotion_hits  =  sum(int(songs_df[f"count_{emo}"].sum()) for emo in emotion_names)
        emo_stats_rows      =  []
        for emo in emotion_names:
            scores_col      =  songs_df[emo]
            counts_col      =  songs_df[f"count_{emo}"]
            emo_total_hits  =  int(counts_col.sum())
            presence_count  =  int((scores_col > 0).sum())

            emo_stats_rows.append({
                "emotion":            emo,
                "mean_score":         round(float(scores_col.mean()), 4),
                "std_score":          round(float(scores_col.std(ddof  =  0)), 4),
                "median_score":       round(float(scores_col.median()), 4),
                "total_matches":      emo_total_hits,
                "match_share_pct":    round(emo_total_hits / total_emotion_hits * 100.0, 2) if total_emotion_hits > 0 else 0.0,
                "song_presence_pct":  round(presence_count / total_songs * 100.0, 2),
            })

        emotion_distribution_df  =  (
            pd.DataFrame(emo_stats_rows)
            .sort_values(by  =  "mean_score", ascending  =  False)
            .reset_index(drop  =  True)
        )

        sent_stats_rows  =  []
        if sentiment_names:
            total_sent_hits  =  sum(int(songs_df[f"count_{sent}"].sum()) for sent in sentiment_names)
            for sent in sentiment_names:
                s_scores_col    =  songs_df[f"sentiment_{sent}"]
                s_counts_col    =  songs_df[f"count_{sent}"]
                sent_hits       =  int(s_counts_col.sum())
                s_presence      =  int((s_scores_col > 0).sum())

                sent_stats_rows.append({
                    "sentiment":          sent,
                    "mean_score":         round(float(s_scores_col.mean()), 4),
                    "std_score":          round(float(s_scores_col.std(ddof  =  0)), 4),
                    "total_matches":      sent_hits,
                    "match_share_pct":    round(sent_hits / total_sent_hits * 100.0, 2) if total_sent_hits > 0 else 0.0,
                    "song_presence_pct":  round(s_presence / total_songs * 100.0, 2),
                })

        sentiment_distribution_df  =  pd.DataFrame(sent_stats_rows)
        emotion_correlation_df     =  songs_df[emotion_names].corr(method  =  "pearson").round(4)

        return CorpusStatistics(
            overview                   =  overview,
            songs_df                   =  songs_df,
            dominant_emotions_df       =  dominant_emotions_df,
            dominant_sentiments_df     =  dominant_sentiments_df,
            emotion_distribution_df    =  emotion_distribution_df,
            sentiment_distribution_df  =  sentiment_distribution_df,
            emotion_correlation_df     =  emotion_correlation_df,
        )

# ==============================================================================
