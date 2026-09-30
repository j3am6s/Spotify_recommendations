"""Audio-feature recommendation methods adapted from the project notebook.

These are feature-similarity experiments, not a listening-history model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

FEATURES = [
    "danceability", "energy", "valence", "tempo", "acousticness",
    "liveness", "speechiness", "instrumentalness",
]
DISPLAY = ["track_id", "track_name", "artists", "album_name", "track_genre"]


def load_tracks(csv_path: str | Path) -> pd.DataFrame:
    """Load the Maharshi Pandya Spotify tracks CSV and keep complete unique IDs."""
    df = pd.read_csv(csv_path)
    required = set(FEATURES + DISPLAY)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.dropna(subset=list(required)).drop_duplicates(subset="track_id").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid songs after cleaning")
    return df


class Recommender:
    def __init__(self, tracks: pd.DataFrame):
        self.tracks = tracks.reset_index(drop=True).copy()
        self.scaler = MinMaxScaler()
        self.vectors = self.scaler.fit_transform(self.tracks[FEATURES])
        self.positions = {tid: i for i, tid in enumerate(self.tracks["track_id"])}

    def _position(self, track_id: str) -> int:
        if track_id not in self.positions:
            raise ValueError(f"Unknown track_id: {track_id}. Check that it exists in the downloaded dataset.")
        return self.positions[track_id]

    def similar(self, track_id: str, n: int = 10, genre: str | None = None,
                min_popularity: int | None = None, artist: str | None = None,
                weights: dict[str, float] | None = None) -> pd.DataFrame:
        """Nearest-neighbour recommendations using the SAME feature scale for seed and candidates."""
        if n < 1: raise ValueError("n must be positive")
        ix = self._position(track_id)
        mask = self.tracks["track_id"].ne(track_id)
        if genre is not None: mask &= self.tracks["track_genre"].eq(genre)
        if min_popularity is not None:
            if "popularity" not in self.tracks: raise ValueError("Dataset has no popularity column")
            mask &= self.tracks["popularity"].ge(min_popularity)
        if artist is not None: mask &= self.tracks["artists"].str.contains(artist,case=False,regex=False,na=False)
        candidates = np.flatnonzero(mask.to_numpy())
        if len(candidates)==0: return self.tracks.iloc[[]][DISPLAY].copy()
        feature_weights = np.ones(len(FEATURES))
        for name, weight in (weights or {}).items():
            if name not in FEATURES: raise ValueError(f"Unknown feature: {name}")
            if weight < 0: raise ValueError("Weights must be non-negative")
            feature_weights[FEATURES.index(name)] = weight
        model = NearestNeighbors(n_neighbors=min(n,len(candidates)))
        model.fit(self.vectors[candidates] * feature_weights)
        distances, indices = model.kneighbors((self.vectors[ix]*feature_weights).reshape(1,-1))
        result=self.tracks.iloc[candidates[indices[0]]][DISPLAY].copy()
        result["distance"] = distances[0]
        return result.reset_index(drop=True)

    def flow(self, track_id: str, steps: int = 10) -> pd.DataFrame:
        """Build a non-repeating chain, each song nearest to the previous song."""
        if steps < 1: raise ValueError("steps must be positive")
        current=self._position(track_id); visited={current}; selected=[]; distances=[]
        for _ in range(min(steps,len(self.tracks)-1)):
            candidates=np.array([i for i in range(len(self.tracks)) if i not in visited])
            if not len(candidates): break
            knn=NearestNeighbors(n_neighbors=1).fit(self.vectors[candidates])
            distance, idx=knn.kneighbors(self.vectors[current].reshape(1,-1))
            current=int(candidates[idx[0,0]]);visited.add(current);selected.append(current);distances.append(float(distance[0,0]))
        result=self.tracks.iloc[selected][DISPLAY].copy();result["distance_from_previous"]=distances
        return result.reset_index(drop=True)

    def cosine_to_seed(self, track_id: str, recommended_ids: list[str]) -> np.ndarray:
        """Cosine similarity of the normalized audio features to the seed (not listener satisfaction)."""
        seed=self.vectors[self._position(track_id)].reshape(1,-1)
        if not recommended_ids: return np.array([])
        return cosine_similarity(self.vectors[[self._position(t) for t in recommended_ids]],seed).ravel()


def main() -> None:
    parser=argparse.ArgumentParser(description="Recommend Spotify dataset tracks using audio-feature nearest neighbours")
    parser.add_argument("--data",required=True,help="Path to dataset.csv (not included)")
    parser.add_argument("--track-id",required=True,help="Seed track ID present in the CSV")
    parser.add_argument("--method",choices=["knn","flow"],default="knn")
    parser.add_argument("--n",type=int,default=10)
    parser.add_argument("--genre")
    parser.add_argument("--min-popularity",type=int)
    parser.add_argument("--artist")
    parser.add_argument("--save-ids",type=Path,help="Optional output file containing one recommended Spotify track ID per line")
    args=parser.parse_args()
    recommender=Recommender(load_tracks(args.data))
    if args.method=="flow":
        if any(v is not None for v in (args.genre,args.min_popularity,args.artist)):
            parser.error("genre, popularity and artist filters are only supported for --method knn")
        result=recommender.flow(args.track_id,args.n)
    else: result=recommender.similar(args.track_id,args.n,args.genre,args.min_popularity,args.artist)
    print(result.to_string(index=False))
    if args.save_ids:
        args.save_ids.write_text("\n".join(result["track_id"].tolist())+"\n",encoding="utf-8")
        print(f"Saved track IDs to {args.save_ids}")

if __name__=="__main__": main()
