# ==============================================================================
from dataclasses import dataclass
import re
from typing import Literal

import numpy as np
import pandas as pd

from dataloader.corpus import CorpusData, CorpusInstance
from model.model import BaseModel
# ==============================================================================

SPACE_REGEX            =  re.compile(r"\s+")
STRUCTURAL_LINE_REGEX  =  re.compile(
    r"[\r\n,;.!?]+|(?<=[a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ])\s+(?=[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ])"
)

# ==============================================================================

def compute_normalized_shannon_entropy(
        prob_matrix: np.ndarray
    ) -> np.ndarray:
    probs       =  np.asarray(prob_matrix, dtype  =  np.float64)
    row_sums    =  probs.sum(axis  =  -1)
    valid_mask  =  row_sums > 1e-9

    safe_sums  =  np.where(valid_mask, row_sums, 1.0)[..., np.newaxis]
    norm_p     =  probs / safe_sums

    with np.errstate(divide  =  "ignore", invalid  =  "ignore"):
        log_p        =  np.where(norm_p > 1e-12, np.log2(norm_p), 0.0)
        raw_entropy  =  -np.sum(norm_p * log_p, axis  =  -1)

    max_entropy   =  np.log2(probs.shape[-1])
    norm_entropy  =  raw_entropy / max_entropy
    norm_entropy  =  np.clip(norm_entropy, 0.0, 1.0)
    return np.where(valid_mask, norm_entropy, np.nan)

# ==============================================================================

@dataclass(slots  =  True)
class TrajectoryStatistics:
    strategy:                    str
    n_total_songs:               int
    n_valid_songs:               int
    min_word_count:              int
    emotion_names:               list[str]
    sentiment_names:             list[str]
    valid_song_indices:          np.ndarray
    has_stanza_breaks:           np.ndarray
    segment_token_counts:        np.ndarray
    segment_matched_counts:      np.ndarray
    segment_coverage:            np.ndarray
    segment_emotion_scores:      np.ndarray
    segment_emotion_raw_counts:  np.ndarray
    segment_sentiment_scores:    np.ndarray
    segment_entropy:             np.ndarray
    segment_dominant_emotion:    np.ndarray
    segment_summary_df:          pd.DataFrame
    transition_counts_8x8_df:    pd.DataFrame
    transition_probs_8x8_df:     pd.DataFrame
    s1_to_s3_probs_8x8_df:       pd.DataFrame
    transition_probs_9x9_df:     pd.DataFrame

# ==============================================================================

class TrajectoryAnalyzer:
    def __init__(
            self,
            model: BaseModel,
            min_word_count: int  =  16
        ):
        self.model           =  model
        self.min_word_count  =  max(4, min_word_count)

    @staticmethod
    def _split_equal_tokens(
            tokens: np.ndarray
        ) -> list[np.ndarray]:
        return [arr for arr in np.array_split(tokens, 4)]

    @staticmethod
    def _split_structural_stanza(
            inst: CorpusInstance
        ) -> tuple[list[np.ndarray], bool]:
        tokens    =  inst.lyrics_in_token
        n_tokens  =  len(tokens)
        raw_text  =  getattr(inst, "raw_lyrics", None) or ""

        if raw_text:
            lines  =  [ln.strip() for ln in STRUCTURAL_LINE_REGEX.split(raw_text) if ln.strip()]
        else:
            lines  =  []

        if len(lines) < 4:
            return [arr for arr in np.array_split(tokens, 4)], False

        line_quartiles  =  np.array_split(np.array(lines, dtype  =  object), 4)
        weights  =  np.array(
            [
                max(1, sum(len(SPACE_REGEX.split(str(ln).strip())) for ln in q))
                for q in line_quartiles
            ],
            dtype  =  np.float64,
        )
        proportions  =  weights / weights.sum()

        raw_sizes  =  np.round(proportions * n_tokens).astype(int)
        raw_sizes  =  np.maximum(1, raw_sizes)
        diff       =  n_tokens - int(raw_sizes.sum())
        if diff != 0:
            order  =  np.argsort(raw_sizes)[::-1]
            for idx in order:
                if raw_sizes[idx] + diff >= 1:
                    raw_sizes[idx]  +=  diff
                    break

        split_points  =  np.cumsum(raw_sizes)[:-1]
        segments      =  np.split(tokens, split_points)
        return segments, True

    def analyze(
            self,
            corpus: CorpusData,
            strategy: Literal["equal_tokens", "structural_stanza"]  =  "equal_tokens"
        ) -> TrajectoryStatistics:
        if strategy not in ("equal_tokens", "structural_stanza"):
            raise ValueError(f"Unsupported strategy: {strategy}")

        valid_indices      =  []
        has_stanza_list    =  []
        all_segments_flat  =  []

        for idx, inst in enumerate(corpus.instances):
            if len(inst.lyrics_in_token) < self.min_word_count:
                continue
            valid_indices.append(idx)
            if strategy == "equal_tokens":
                segs        =  self._split_equal_tokens(inst.lyrics_in_token)
                raw_text    =  getattr(inst, "raw_lyrics", None) or ""
                has_breaks  =  (
                    len([ln for ln in STRUCTURAL_LINE_REGEX.split(raw_text) if ln.strip()]) >= 4
                    if raw_text else False
                )
                has_stanza_list.append(has_breaks)
            else:
                segs, has_breaks  =  self._split_structural_stanza(inst)
                has_stanza_list.append(has_breaks)

            all_segments_flat.extend(segs)

        n_valid  =  len(valid_indices)
        if n_valid == 0:
            raise ValueError(
                f"No songs in corpus met min_word_count={self.min_word_count}."
            )

        emolex  =  getattr(self.model, "emolex", None)
        if emolex is not None:
            emotion_names    =  list(emolex.emotions)
            sentiment_names  =  list(emolex.sentiments or ["positive", "negative"])
            n_emo            =  len(emotion_names)
            n_sent           =  len(sentiment_names)

            seg_tok_counts    =  np.zeros((n_valid, 4), dtype  =  np.int32)
            seg_match_counts  =  np.zeros((n_valid, 4), dtype  =  np.int32)
            seg_cov           =  np.zeros((n_valid, 4), dtype  =  np.float64)
            seg_emo_scores    =  np.zeros((n_valid, 4, n_emo), dtype  =  np.float64)
            seg_emo_raw       =  np.zeros((n_valid, 4, n_emo), dtype  =  np.int32)
            seg_sent_scores   =  np.zeros((n_valid, 4, n_sent), dtype  =  np.float64)
            seg_dom_emo       =  np.empty((n_valid, 4), dtype  =  object)

            word_to_idx  =  emolex.word_to_index
            emo_mat      =  emolex.matrix
            sent_mat     =  emolex.sentiment_matrix

            for s_idx in range(n_valid):
                for m in range(4):
                    seg_tokens                =  all_segments_flat[s_idx * 4 + m]
                    t_len                     =  len(seg_tokens)
                    seg_tok_counts[s_idx, m]  =  t_len

                    indices  =  [
                        word_to_idx[t] for t in seg_tokens.tolist() if t in word_to_idx
                    ]
                    m_len                       =  len(indices)
                    seg_match_counts[s_idx, m]  =  m_len
                    seg_cov[s_idx, m]           =  m_len / t_len if t_len > 0 else 0.0

                    if m_len == 0:
                        seg_dom_emo[s_idx, m]  =  "neutral"
                        continue

                    emo_vec                   =  emo_mat[indices].sum(axis  =  0)
                    seg_emo_raw[s_idx, m]     =  emo_vec
                    emo_sum                   =  float(emo_vec.sum())
                    if emo_sum > 0:
                        norm_vec                  =  emo_vec.astype(np.float64) / emo_sum
                        seg_emo_scores[s_idx, m]  =  norm_vec
                        seg_dom_emo[s_idx, m]     =  emotion_names[int(np.argmax(norm_vec))]
                    else:
                        seg_dom_emo[s_idx, m]     =  "neutral"

                    if sent_mat is not None:
                        s_vec  =  sent_mat[indices].sum(axis  =  0).astype(np.float64)
                        s_sum  =  float(s_vec.sum())
                        if s_sum > 0:
                            seg_sent_scores[s_idx, m]  =  s_vec / s_sum
        else:
            flat_results     =  self.model.predict_batch(all_segments_flat)
            emotion_names    =  list(flat_results[0].scores.keys())
            first_sent       =  flat_results[0].metadata.get("sentiment_scores", {})
            sentiment_names  =  (
                list(first_sent.keys())
                if isinstance(first_sent, dict) and first_sent
                else ["positive", "negative"]
            )
            n_emo   =  len(emotion_names)
            n_sent  =  len(sentiment_names)

            seg_tok_counts    =  np.zeros((n_valid, 4), dtype  =  np.int32)
            seg_match_counts  =  np.zeros((n_valid, 4), dtype  =  np.int32)
            seg_cov           =  np.zeros((n_valid, 4), dtype  =  np.float64)
            seg_emo_scores    =  np.zeros((n_valid, 4, n_emo), dtype  =  np.float64)
            seg_emo_raw       =  np.zeros((n_valid, 4, n_emo), dtype  =  np.int32)
            seg_sent_scores   =  np.zeros((n_valid, 4, n_sent), dtype  =  np.float64)
            seg_dom_emo       =  np.empty((n_valid, 4), dtype  =  object)

            for s_idx in range(n_valid):
                for m in range(4):
                    res                       =  flat_results[s_idx * 4 + m]
                    meta                      =  res.metadata
                    seg_tok_counts[s_idx, m]  =  int(meta.get("total_tokens", 0))
                    seg_match_counts[s_idx, m] = int(meta.get("matched_tokens", 0))
                    seg_cov[s_idx, m]         =  float(meta.get("coverage", 0.0))
                    seg_dom_emo[s_idx, m]     =  res.emotions
                    for e_i, e_name in enumerate(emotion_names):
                        seg_emo_scores[s_idx, m, e_i]  =  float(res.scores.get(e_name, 0.0))
                    raw_c                     =  meta.get("raw_counts", [0] * n_emo)
                    seg_emo_raw[s_idx, m]     =  np.array(raw_c, dtype  =  np.int32)
                    s_sc                      =  meta.get("sentiment_scores", {})
                    for s_i, s_name in enumerate(sentiment_names):
                        seg_sent_scores[s_idx, m, s_i]  =  float(s_sc.get(s_name, 0.0))

        seg_entropy  =  compute_normalized_shannon_entropy(seg_emo_scores)

        summary_rows  =  []
        for m in range(4):
            seg_label  =  f"S{m + 1}"
            valid_ent  =  seg_entropy[:, m][~np.isnan(seg_entropy[:, m])]
            n_ent      =  len(valid_ent)
            row        =  {
                "segment":                 seg_label,
                "n_songs":                 n_valid,
                "n_active_emotion_songs":  n_ent,
                "zero_match_rate_pct":     round(100.0 * (1.0 - n_ent / n_valid), 2),
                "mean_tokens":             round(float(seg_tok_counts[:, m].mean()), 2),
                "mean_coverage":           round(float(seg_cov[:, m].mean()), 4),
                "std_coverage":            round(float(seg_cov[:, m].std(ddof  =  1)), 4),
                "mean_entropy":            round(float(np.nanmean(seg_entropy[:, m])), 4) if n_ent > 0 else np.nan,
                "std_entropy":             round(float(np.nanstd(seg_entropy[:, m], ddof  =  1)), 4) if n_ent > 1 else np.nan,
                "median_entropy":          round(float(np.nanmedian(seg_entropy[:, m])), 4) if n_ent > 0 else np.nan,
                "ci95_entropy":            round(1.96 * float(np.nanstd(seg_entropy[:, m], ddof  =  1)) / np.sqrt(n_ent), 4) if n_ent > 1 else np.nan,
            }
            for e_i, emo in enumerate(emotion_names):
                col              =  seg_emo_scores[:, m, e_i]
                row[f"mean_{emo}"]  =  round(float(col.mean()), 4)
                row[f"std_{emo}"]   =  round(float(col.std(ddof  =  1)), 4)
                row[f"ci95_{emo}"]  =  round(1.96 * float(col.std(ddof  =  1)) / np.sqrt(n_valid), 4)
            for s_i, sent in enumerate(sentiment_names):
                col               =  seg_sent_scores[:, m, s_i]
                row[f"mean_{sent}"]  =  round(float(col.mean()), 4)
                row[f"std_{sent}"]   =  round(float(col.std(ddof  =  1)), 4)
            summary_rows.append(row)

        segment_summary_df  =  pd.DataFrame(summary_rows)

        emo_to_idx    =  {e: i for i, e in enumerate(emotion_names)}
        all_labels_9  =  emotion_names + ["neutral"]
        emo9_to_idx   =  {e: i for i, e in enumerate(all_labels_9)}

        trans_counts_8x8  =  np.zeros((len(emotion_names), len(emotion_names)), dtype  =  np.int64)
        s1_s3_counts_8x8  =  np.zeros((len(emotion_names), len(emotion_names)), dtype  =  np.int64)
        trans_counts_9x9  =  np.zeros((len(all_labels_9), len(all_labels_9)), dtype  =  np.int64)

        for s_idx in range(n_valid):
            for m in range(3):
                e_curr  =  str(seg_dom_emo[s_idx, m])
                e_next  =  str(seg_dom_emo[s_idx, m + 1])
                if e_curr in emo9_to_idx and e_next in emo9_to_idx:
                    trans_counts_9x9[emo9_to_idx[e_curr], emo9_to_idx[e_next]]  +=  1
                if e_curr in emo_to_idx and e_next in emo_to_idx:
                    trans_counts_8x8[emo_to_idx[e_curr], emo_to_idx[e_next]]  +=  1

            e_s1  =  str(seg_dom_emo[s_idx, 0])
            e_s3  =  str(seg_dom_emo[s_idx, 2])
            if e_s1 in emo_to_idx and e_s3 in emo_to_idx:
                s1_s3_counts_8x8[emo_to_idx[e_s1], emo_to_idx[e_s3]]  +=  1

        def _normalize_rows(
                counts: np.ndarray, 
                labels: list[str]
            ) -> pd.DataFrame:
            row_sums  =  counts.sum(axis  =  1, keepdims  =  True).astype(np.float64)
            probs     =  np.where(row_sums > 0, counts / np.maximum(row_sums, 1.0), 0.0)
            return pd.DataFrame(np.round(probs, 4), index  =  labels, columns  =  labels)

        transition_counts_8x8_df  =  pd.DataFrame(
            trans_counts_8x8, index  =  emotion_names, columns  =  emotion_names
        )
        transition_probs_8x8_df   =  _normalize_rows(trans_counts_8x8, emotion_names)
        s1_to_s3_probs_8x8_df     =  _normalize_rows(s1_s3_counts_8x8, emotion_names)
        transition_probs_9x9_df   =  _normalize_rows(trans_counts_9x9, all_labels_9)

        return TrajectoryStatistics(
            strategy                    =  strategy,
            n_total_songs               =  len(corpus.instances),
            n_valid_songs               =  n_valid,
            min_word_count              =  self.min_word_count,
            emotion_names               =  emotion_names,
            sentiment_names             =  sentiment_names,
            valid_song_indices          =  np.array(valid_indices, dtype  =  np.int32),
            has_stanza_breaks           =  np.array(has_stanza_list, dtype  =  bool),
            segment_token_counts        =  seg_tok_counts,
            segment_matched_counts      =  seg_match_counts,
            segment_coverage            =  seg_cov,
            segment_emotion_scores      =  seg_emo_scores,
            segment_emotion_raw_counts  =  seg_emo_raw,
            segment_sentiment_scores    =  seg_sent_scores,
            segment_entropy             =  seg_entropy,
            segment_dominant_emotion    =  seg_dom_emo,
            segment_summary_df          =  segment_summary_df,
            transition_counts_8x8_df    =  transition_counts_8x8_df,
            transition_probs_8x8_df     =  transition_probs_8x8_df,
            s1_to_s3_probs_8x8_df       =  s1_to_s3_probs_8x8_df,
            transition_probs_9x9_df     =  transition_probs_9x9_df,
        )

# ==============================================================================
