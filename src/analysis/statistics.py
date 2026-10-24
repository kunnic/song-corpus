# ───────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dataloader.corpus import CorpusData
from model.model import BaseModel, ModelResult
# ───────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class CorpusStatistics:
    overview:                  dict[str, float | int]
    songs_df:                  pd.DataFrame
    dominant_emotions_df:      pd.DataFrame
    dominant_sentiments_df:    pd.DataFrame
    emotion_distribution_df:   pd.DataFrame
    sentiment_distribution_df: pd.DataFrame
    emotion_correlation_df:    pd.DataFrame

    def summary_text(self) -> str:
        lines = [
            "=" * 65,
            "CORPUS EMOTION & SENTIMENT STATISTICAL SUMMARY",
            "=" * 65,
            f"  Total Songs           : {self.overview['total_songs']}",
            f"  Emotional Songs       : {self.overview['emotional_songs']} "
            f"({100 * (1 - self.overview['neutral_ratio']):.2f}%)",
            f"  Neutral Songs         : {self.overview['neutral_songs']} "
            f"({100 * self.overview['neutral_ratio']:.2f}%)",
            f"  Total Tokens          : {self.overview['total_tokens']}",
            f"  Matched EmoLex Tokens : {self.overview['total_matched_tokens']} "
            f"({100 * self.overview['corpus_coverage']:.2f}%)",
            f"  Song Coverage (Mean)  : {100 * self.overview['coverage_mean']:.2f}% "
            f"(std: {100 * self.overview['coverage_std']:.2f}%, "
            f"median: {100 * self.overview['coverage_median']:.2f}%)",
            "-" * 65,
            "DOMINANT EMOTION DISTRIBUTION:",
            self.dominant_emotions_df.to_string(index=False),
            "-" * 65,
            "EMOTION SCORE & TOKEN FREQUENCY STATISTICS:",
            self.emotion_distribution_df.to_string(index=False),
        ]
        if not self.sentiment_distribution_df.empty:
            lines.extend([
                "-" * 65,
                "SENTIMENT DISTRIBUTION:",
                self.sentiment_distribution_df.to_string(index=False),
            ])
        lines.append("=" * 65)
        return "\n".join(lines)
# ───────────────────────────────────────────────────────────────────
class CorpusAnalyzer:
    '''
        Statistical analyzer for song corpus emotion & sentiment results.
        Designed around BaseModel / ModelResult so it works seamlessly with
        LexiconModel (VnEmoLex) today and any future BaseModel subclasses.
    '''
    def __init__(self, model: BaseModel | None = None):
        self.model = model

    def analyze(
            self,
            corpus: CorpusData,
            results: list[ModelResult] | None = None,
        ) -> CorpusStatistics:

        if results is None:
            if self.model is None:
                raise ValueError(
                    "Either pass `results` or initialize CorpusAnalyzer with a `BaseModel`."
                )
            token_list = [inst.lyrics_in_token for inst in corpus.instances]
            results = self.model.predict_batch(token_list)

        if len(results) != len(corpus.instances):
            raise ValueError(
                f"Length mismatch: {len(corpus.instances)} corpus instances "
                f"vs {len(results)} model results."
            )

        if len(results) == 0:
            empty_df = pd.DataFrame()
            return CorpusStatistics(
                overview                  = {"total_songs": 0},
                songs_df                  = empty_df,
                dominant_emotions_df      = empty_df,
                dominant_sentiments_df    = empty_df,
                emotion_distribution_df   = empty_df,
                sentiment_distribution_df = empty_df,
                emotion_correlation_df    = empty_df,
            )

        emotion_names = list(results[0].scores.keys())
        first_sent = results[0].metadata.get("sentiment_scores", {})
        sentiment_names = list(first_sent.keys()) if isinstance(first_sent, dict) else []

        rows: list[dict[str, Any]] = []
        for inst, res in zip(corpus.instances, results):
            meta = res.metadata
            total_tok = int(meta.get("total_tokens", len(inst.lyrics_in_token)))
            matched_tok = int(meta.get("matched_tokens", 0))
            cov = float(meta.get("coverage", 0.0))
            dom_sent = str(meta.get("dominant_sentiment", "neutral"))

            row: dict[str, Any] = {
                "title":              inst.title,
                "dominant_emotion":   res.emotions,
                "dominant_sentiment": dom_sent,
                "total_tokens":       total_tok,
                "matched_tokens":     matched_tok,
                "coverage":           cov,
            }

            # Normalized emotion scores
            for emo in emotion_names:
                row[emo] = float(res.scores.get(emo, 0.0))

            # Raw emotion counts
            emo_counts = meta.get("emotion_counts")
            raw_counts = meta.get("raw_counts")
            if isinstance(emo_counts, dict):
                for emo in emotion_names:
                    row[f"count_{emo}"] = int(emo_counts.get(emo, 0))
            elif isinstance(raw_counts, list) and len(raw_counts) == len(emotion_names):
                for emo, cnt in zip(emotion_names, raw_counts):
                    row[f"count_{emo}"] = int(cnt)
            else:
                for emo in emotion_names:
                    row[f"count_{emo}"] = 0

            # Sentiment scores & raw counts
            sent_scores = meta.get("sentiment_scores", {})
            sent_counts = meta.get("sentiment_counts", {})
            for sent in sentiment_names:
                row[f"sentiment_{sent}"] = float(
                    sent_scores.get(sent, 0.0) if isinstance(sent_scores, dict) else 0.0
                )
                row[f"count_{sent}"] = int(
                    sent_counts.get(sent, 0) if isinstance(sent_counts, dict) else 0
                )

            rows.append(row)

        songs_df = pd.DataFrame(rows)

        # 1. Corpus Overview Statistics
        total_songs = len(songs_df)
        neutral_songs = int((songs_df["dominant_emotion"] == "neutral").sum())
        emotional_songs = total_songs - neutral_songs
        total_tokens = int(songs_df["total_tokens"].sum())
        total_matched = int(songs_df["matched_tokens"].sum())

        overview: dict[str, float | int] = {
            "total_songs":          total_songs,
            "emotional_songs":      emotional_songs,
            "neutral_songs":        neutral_songs,
            "neutral_ratio":        round(neutral_songs / total_songs, 4),
            "total_tokens":         total_tokens,
            "total_matched_tokens": total_matched,
            "corpus_coverage":      round(total_matched / total_tokens, 4) if total_tokens > 0 else 0.0,
            "coverage_mean":        round(float(songs_df["coverage"].mean()), 4),
            "coverage_std":         round(float(songs_df["coverage"].std(ddof=0)), 4),
            "coverage_median":      round(float(songs_df["coverage"].median()), 4),
            "coverage_min":         round(float(songs_df["coverage"].min()), 4),
            "coverage_max":         round(float(songs_df["coverage"].max()), 4),
        }

        # 2. Dominant Emotion Distribution
        dom_counts = songs_df["dominant_emotion"].value_counts()
        dominant_emotions_df = pd.DataFrame({
            "emotion":    dom_counts.index.tolist(),
            "song_count": dom_counts.values.astype(int).tolist(),
            "percentage": np.round(dom_counts.values / total_songs * 100.0, 2).tolist(),
        })

        # 3. Dominant Sentiment Distribution
        sent_dom_counts = songs_df["dominant_sentiment"].value_counts()
        dominant_sentiments_df = pd.DataFrame({
            "sentiment":  sent_dom_counts.index.tolist(),
            "song_count": sent_dom_counts.values.astype(int).tolist(),
            "percentage": np.round(sent_dom_counts.values / total_songs * 100.0, 2).tolist(),
        })

        # 4. Per-Emotion Score & Frequency Distribution
        total_emotion_hits = sum(int(songs_df[f"count_{emo}"].sum()) for emo in emotion_names)
        emo_stats_rows = []
        for emo in emotion_names:
            scores_col = songs_df[emo]
            counts_col = songs_df[f"count_{emo}"]
            emo_total_hits = int(counts_col.sum())
            presence_count = int((scores_col > 0).sum())

            emo_stats_rows.append({
                "emotion":           emo,
                "mean_score":        round(float(scores_col.mean()), 4),
                "std_score":         round(float(scores_col.std(ddof=0)), 4),
                "median_score":      round(float(scores_col.median()), 4),
                "q25_score":         round(float(scores_col.quantile(0.25)), 4),
                "q75_score":         round(float(scores_col.quantile(0.75)), 4),
                "max_score":         round(float(scores_col.max()), 4),
                "total_matches":     emo_total_hits,
                "match_share_pct":   round(emo_total_hits / total_emotion_hits * 100.0, 2)
                                     if total_emotion_hits > 0 else 0.0,
                "song_presence_pct": round(presence_count / total_songs * 100.0, 2),
            })

        emotion_distribution_df = pd.DataFrame(emo_stats_rows).sort_values(
            by="mean_score", ascending=False
        ).reset_index(drop=True)

        # 5. Per-Sentiment Score & Frequency Distribution
        sent_stats_rows = []
        if sentiment_names:
            total_sent_hits = sum(
                int(songs_df[f"count_{sent}"].sum()) for sent in sentiment_names
            )
            for sent in sentiment_names:
                s_scores_col = songs_df[f"sentiment_{sent}"]
                s_counts_col = songs_df[f"count_{sent}"]
                sent_hits = int(s_counts_col.sum())
                s_presence = int((s_scores_col > 0).sum())

                sent_stats_rows.append({
                    "sentiment":         sent,
                    "mean_score":        round(float(s_scores_col.mean()), 4),
                    "std_score":         round(float(s_scores_col.std(ddof=0)), 4),
                    "total_matches":     sent_hits,
                    "match_share_pct":   round(sent_hits / total_sent_hits * 100.0, 2)
                                         if total_sent_hits > 0 else 0.0,
                    "song_presence_pct": round(s_presence / total_songs * 100.0, 2),
                })

        sentiment_distribution_df = pd.DataFrame(sent_stats_rows)

        # 6. Emotion Correlation Matrix
        emotion_correlation_df = songs_df[emotion_names].corr(method="pearson").round(4)

        return CorpusStatistics(
            overview                  = overview,
            songs_df                  = songs_df,
            dominant_emotions_df      = dominant_emotions_df,
            dominant_sentiments_df    = dominant_sentiments_df,
            emotion_distribution_df   = emotion_distribution_df,
            sentiment_distribution_df = sentiment_distribution_df,
            emotion_correlation_df    = emotion_correlation_df,
        )
# ───────────────────────────────────────────────────────────────────
