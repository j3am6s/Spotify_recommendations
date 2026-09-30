# Spotify ML Recommendation System

Python machine learning project exploring how songs can be recommended using their audio characteristics, from nearest-neighbour search to sequential playlist generation. 

## Overview

Rather than relying on listening history, we explore recommendations based on Spotify audio features. Using an existing tracks dataset, we compare similarity-based methods with exploratory supervised classification approaches, and connect the resulting track IDs to Spotify for optional playlist creation.

## How it works

1. **Data preparation:** load the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset), remove duplicates and incomplete entries, and min–max normalise eight audio features: danceability, energy, valence, tempo, acousticness, liveness, speechiness and instrumentalness.
2. **Comparative analysis:** use cosine similarity in feature space to inspect how recommended songs compare with a starting track. This is a *proxy* for musical resemblance, not a measure of listener preference.
3. **k-Nearest Neighbours:** recommend close tracks in normalised feature space, with optional genre, artist and popularity filters and feature weighting.
4. **Recommendation flow:** choose each next song relative to the previous one, avoiding repetitions. This explores gradually changing playlists rather than repeatedly choosing songs closest to the original seed.
5. **Other experiments:** investigate Logistic Regression and Random Forest using labels derived from a cosine-similarity threshold. These experiments are included in the notebook; because labels are derived from the same audio features, they should not be interpreted as independently validated listener-preference models.
6. **Spotify integration:** optionally create a private playlist from the recommended track IDs using Spotipy and your own Spotify account.

## Repository contents

```text
src/
  recommender.py          # Runnable kNN, flow and similarity methods
  playlist_creation.py    # Optional authenticated playlist export
notebooks/
  spotify_recommendation.ipynb  # Cleaned original project write-up and experiments
docs/
  presentation.pdf        # Original presentation
  recap.pdf               # Original recap
requirements.txt
.env.example
.gitignore
```

## Run locally

Install Python 3.10+ and, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Download the dataset linked above and put its `dataset.csv` at `data/dataset.csv` (data is intentionally not committed). Choose an existing `track_id` from that dataset:

```bash
python -m src.recommender --data data/dataset.csv --track-id YOUR_TRACK_ID --method knn --n 10 --save-ids tracks.txt
# Or build a non-repeating sequence:
python -m src.recommender --data data/dataset.csv --track-id YOUR_TRACK_ID --method flow --n 10 --save-ids tracks.txt
```

For filtered kNN suggestions you can add `--genre pop`, `--min-popularity 70`, or `--artist "ARTIST NAME"`. For the full exploratory comparisons, open `notebooks/spotify_recommendation.ipynb`. Notebook outputs and credential-dependent cells were cleared for public sharing; the notebook is an archived research narrative, not a guaranteed top-to-bottom standalone run.

### Optional Spotify playlist export

Create your own Spotify developer application and set its redirect URI to the value in `.env.example`. Copy `.env.example` to `.env` and fill in **your own newly generated credentials**. Then run:

```bash
python -m src.playlist_creation --tracks tracks.txt --name "My Recommendations"
```

This requires authorization to modify private playlists. It creates a playlist on your account; check its contents before running. **Never upload your `.env` or token cache.** The original repository contained exposed credentials: rotate/revoke those credentials and clean the old Git history before publishing this version.


## What I learnt

- Practical experience with data preparation, feature representation, unsupervised neighbour search and exploratory supervised learning
- How a change in recommendation strategy can change the resulting listening experience, even with the same dataset and features
- Why a similarity metric does not necessarily measure subjective recommendation quality, and how API integration brings a notebook experiment closer to a usable application

## Limitations

- This dataset is a historical snapshot and does not cover the entire current Spotify catalogue; some track IDs may no longer resolve.
- The system uses audio features rather than user feedback or listening history, and its numerical similarity scores do not establish listener satisfaction.
- The source notebook's classifier experiments generate labels from feature similarity itself, so their outputs are exploratory rather than independent predictive validation.
- The sequential method greedily selects the nearest next song; it does not optimise the quality of the whole playlist.

## References

- [Maharshi Pandya — Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- [Spotipy documentation](https://spotipy.readthedocs.io/)
