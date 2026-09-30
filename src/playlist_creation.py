"""Optional Spotify playlist export. Requires your own Spotify app and authorization."""
import argparse
import os
from pathlib import Path


def create_playlist(track_ids: list[str], name: str, description: str = "Created from audio-feature recommendations") -> str:
    from dotenv import load_dotenv
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    load_dotenv()
    missing=[x for x in ("SPOTIPY_CLIENT_ID","SPOTIPY_CLIENT_SECRET","SPOTIPY_REDIRECT_URI") if not os.getenv(x)]
    if missing: raise RuntimeError(f"Set these environment variables first: {', '.join(missing)}")
    if not track_ids: raise ValueError("No track IDs provided")
    sp=spotipy.Spotify(auth_manager=SpotifyOAuth(scope="playlist-modify-private",cache_path=".spotify_cache"))
    user_id=sp.current_user()["id"]
    playlist=sp.user_playlist_create(user=user_id,name=name,public=False,description=description)
    for start in range(0,len(track_ids),100):
        sp.playlist_add_items(playlist["id"],track_ids[start:start+100])
    return playlist["external_urls"]["spotify"]


def main():
    parser=argparse.ArgumentParser(description="Export recommended track IDs to a private Spotify playlist")
    parser.add_argument("--tracks",type=Path,required=True,help="Text file: one Spotify track ID per line")
    parser.add_argument("--name",required=True)
    parser.add_argument("--description",default="Created from audio-feature recommendations")
    args=parser.parse_args()
    track_ids=[line.strip() for line in args.tracks.read_text().splitlines() if line.strip()]
    print(create_playlist(track_ids,args.name,args.description))

if __name__=="__main__": main()
