
import unittest
import os
import memory_truth

class TestMemoryTruthHelpers(unittest.TestCase):

    def setUp(self):
        # Use a fresh test DB for isolation
        self.original_db_path = memory_truth.DB_PATH
        memory_truth.DB_PATH = 'data/test_memory_helper.db'
        memory_truth.init_db()

    def tearDown(self):
        if os.path.exists(memory_truth.DB_PATH):
            os.remove(memory_truth.DB_PATH)
        memory_truth.DB_PATH = self.original_db_path

    def test_get_episodes_by_ids(self):
        # Insert test episodes
        id1 = memory_truth.add_episode(topic="T1", notes="N1", outcome="O1")
        id2 = memory_truth.add_episode(topic="T2", notes="N2", outcome="O2")
        id3 = memory_truth.add_episode(topic="T3", notes="N3", outcome="O3")

        # Test valid retrieval
        episodes = memory_truth.get_episodes_by_ids([id1, id3])
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]['id'], id1)
        self.assertEqual(episodes[1]['id'], id3)
        self.assertEqual(episodes[0]['topic'], "T1")

        # Test with missing ID
        episodes = memory_truth.get_episodes_by_ids([id1, 999, id2])
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]['id'], id1)
        self.assertEqual(episodes[1]['id'], id2)

        # Test empty
        self.assertEqual(memory_truth.get_episodes_by_ids([]), [])

        # Test ordering
        episodes = memory_truth.get_episodes_by_ids([id3, id1])
        self.assertEqual(episodes[0]['id'], id3)
        self.assertEqual(episodes[1]['id'], id1)

    def test_get_facts_by_ids(self):
        # Insert test facts
        id1 = memory_truth.add_fact(topic="TF1", subject="S1", predicate="P1", object_="O1")
        id2 = memory_truth.add_fact(topic="TF2", subject="S2", predicate="P2", object_="O2")

        # Test valid retrieval
        facts = memory_truth.get_facts_by_ids([id1, id2])
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0]['subject'], "S1")

        # Test ordering and missing
        facts = memory_truth.get_facts_by_ids([id2, 888, id1])
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0]['id'], id2)
        self.assertEqual(facts[1]['id'], id1)

    print("ALL TESTS PASSED")

if __name__ == '__main__':
    unittest.main()
