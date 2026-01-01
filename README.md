# Sentiment Analysis on Vietnamese Song Lyrics

Sentiment and emotion analysis on a corpus of 62,275 Vietnamese song lyrics using NLP techniques.

## Dataset

| Field | Description | Data Type |
|-------|-------------|-----------|
| Title | The title of the song | String |
| Composers | The name(s) of the composer(s) | List[String] |
| Lyricists | The name(s) of the lyricist(s) | List[String] |
| Year | The release year of the song | Integer |
| Genres | The musical genres of the song | List[String] |
| Lyrics | The full text of the song lyrics | String |
| URLs | URLs related to the song | List[String] |
| Source | Origin (Translated or Vietnamese) | String |
| Note | Additional notes | String |

## Methodology

1. **Data Collection**: Gathered from karaoke websites, enriched with Spotify/MusicBrainz APIs
2. **Data Preprocessing**: Text cleaning, tokenization (Underthesea), normalization
3. **Analysis**: Lexicon-based (VnEmoLex) and Deep Learning-based (PhoBERT) approaches

## Results

| Approach | Accuracy | F1-score |
|----------|----------|----------|
| Lexicon-based (VnEmoLex) | 28% | 0.227 |
| Deep Learning (PhoBERT) | 89.59% | 0.896 |

## Key Findings

- Vietnamese song lyrics are predominantly associated with **Sadness** and **Joy**
- Emotions like Anger, Disgust, and Surprise are rarely expressed

## Requirements

See [requirements.txt](requirements.txt)

## Author

Huy (kunnic) Duc Nguyen