"""Small synthetic tests; do not require external Spotify access or downloaded data."""
import unittest
import pandas as pd
from src.recommender import Recommender, FEATURES

class RecommenderTests(unittest.TestCase):
    def setUp(self):
        rows=[]
        for i in range(6):
            r={f:float(i) for f in FEATURES}
            r.update(track_id=str(i),track_name=f"Song {i}",artists="Artist",album_name="Album",track_genre="pop",popularity=50+i)
            rows.append(r)
        self.model=Recommender(pd.DataFrame(rows))
    def test_knn_excludes_seed(self):
        result=self.model.similar('0',3)
        self.assertEqual(len(result),3)
        self.assertNotIn('0',result.track_id.tolist())
    def test_flow_no_repetition(self):
        result=self.model.flow('0',10)
        self.assertEqual(len(result),5)
        self.assertEqual(len(result.track_id.unique()),5)
    def test_unknown_track(self):
        with self.assertRaises(ValueError): self.model.similar('missing')
    def test_filter(self):
        result=self.model.similar('0',10,min_popularity=54)
        self.assertEqual(set(result.track_id),{'4','5'})

if __name__=='__main__': unittest.main()
