import unittest

import networkx as nx
import pandas as pd

from lab6_ds.analysis import centrality_tables, sentiment_summary


class AnalysisTests(unittest.TestCase):
    def test_articulation_is_not_recurrence(self):
        graph = nx.Graph()
        graph.add_nodes_from([("a", {"tipo": "autor"}), ("b", {"tipo": "autor"}),
                              ("c", {"tipo": "autor"}), ("v", {"tipo": "video"}),
                              ("w", {"tipo": "video"})])
        graph.add_weighted_edges_from([("a", "v", 3), ("b", "v", 1), ("b", "w", 1), ("c", "w", 1)])
        comments = pd.DataFrame({"author_channel_id": ["a", "a", "a", "b", "b", "c"],
                                 "video_id": ["v", "v", "v", "v", "w", "w"],
                                 "channel_id": ["x", "x", "x", "x", "y", "y"],
                                 "comment_id": list("123456")})
        _, authors, videos = centrality_tables(graph, comments)
        authors = authors.set_index("node_id")
        self.assertTrue(authors.loc["a", "recurrent"])
        self.assertFalse(authors.loc["a", "articulation"])
        self.assertEqual(authors.loc["a", "strength"], 3)
        self.assertEqual(authors.loc["a", "degree"], 1)
        self.assertEqual(authors.loc["b", "component_increase"], 1)
        self.assertEqual(authors.loc["b", "distinct_channels"], 2)
        self.assertFalse(videos.projection_articulation.any())

    def test_small_samples_and_unclassified_denominators(self):
        comments = pd.DataFrame({"video_id": ["v", "v", "w"],
                                 "sentiment_status": ["classified", "empty_input", "classified"],
                                 "sentiment": ["NEG", "UNCLASSIFIED", "POS"],
                                 "sentiment_score": [-0.8, None, 0.9],
                                 "low_confidence": [False, False, False], "truncated": [False, False, True]})
        summary = sentiment_summary(comments, "video_id").set_index("video_id")
        self.assertEqual(summary.loc["v", "n_total"], 2)
        self.assertEqual(summary.loc["v", "n_classified"], 1)
        self.assertEqual(summary.loc["v", "pct_NEG"], 1)
        self.assertTrue(summary.small_sample.all())
        self.assertEqual(summary.loc["w", "truncated"], 1)


if __name__ == "__main__":
    unittest.main()
