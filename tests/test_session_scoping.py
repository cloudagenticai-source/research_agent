
import unittest
import uuid
import memory_truth
import report_writer
from unittest.mock import patch, MagicMock

class TestSessionScoping(unittest.TestCase):

    def setUp(self):
        # Use a fresh in-memory DB or temporary file if possible, 
        # but existing code uses 'data/memory.db'. 
        # We will use distinct topics to avoid collisions.
        memory_truth.init_db()
        self.test_topic = f"Test Topic {uuid.uuid4()}"
        self.session_1 = str(uuid.uuid4())
        self.session_2 = str(uuid.uuid4())

    def test_session_isolation(self):
        print(f"Testing session isolation for topic: {self.test_topic}")
        
        # 1. Create Data for Session 1
        memory_truth.add_episode(
            topic=self.test_topic, 
            notes="Session 1 Note", 
            url="http://s1.com",
            session_id=self.session_1
        )
        
        # 2. Create Data for Session 2
        memory_truth.add_episode(
            topic=self.test_topic, 
            notes="Session 2 Note", 
            url="http://s2.com",
            session_id=self.session_2
        )
        
        # 3. Verify get_latest_session_id
        latest = memory_truth.get_latest_session_id(self.test_topic)
        # Should be session 2 because it has the higher ID (auto-increment)
        self.assertEqual(latest, self.session_2)
        
        # 4. Verify get_episodes_by_topic_and_session
        # Session 1 Fetch
        s1_eps = memory_truth.get_episodes_by_topic_and_session(self.test_topic, self.session_1)
        self.assertEqual(len(s1_eps), 1)
        self.assertEqual(s1_eps[0]['url'], "http://s1.com")
        
        # Session 2 Fetch
        s2_eps = memory_truth.get_episodes_by_topic_and_session(self.test_topic, self.session_2)
        self.assertEqual(len(s2_eps), 1)
        self.assertEqual(s2_eps[0]['url'], "http://s2.com")

    @patch('report_writer.memory_vector.VectorMemory')
    @patch('report_writer.OpenAI')
    @patch('report_writer.router.retrieve_router')
    def test_report_generation_session_scoping(self, mock_router, mock_openai, mock_vm):
        # Mock retrieval to return NOTHING useful from Chroma to force SQL fallback
        mock_router.return_value = {'procedural': {}, 'episodic': {}, 'semantic': {}}
        
        mock_client = mock_openai.return_value
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Report content."
        mock_client.chat.completions.create.return_value = mock_resp

        # 1. Populate DB with mixed sessions
        memory_truth.add_episode(self.test_topic, "S1 Data", "http://s1.com", session_id=self.session_1)
        memory_truth.add_episode(self.test_topic, "S2 Data", "http://s2.com", session_id=self.session_2)

        # 2. Generate Report for Session 1
        print("Generating report for Session 1...")
        report_writer.generate_report(self.test_topic, session_id=self.session_1)
        
        # Verify Context
        # The prompt passed to LLM should ONLY contain S1 data
        call_args = mock_client.chat.completions.create.call_args
        _, kwargs = call_args
        user_content = kwargs['messages'][1]['content']
        
        self.assertIn("http://s1.com", user_content)
        self.assertNotIn("http://s2.com", user_content)
        
        # 3. Generate Report for Latest (should be S2)
        print("Generating report for Latest Session...")
        report_writer.generate_report(self.test_topic, session_id=None)
        
        call_args = mock_client.chat.completions.create.call_args
        _, kwargs = call_args
        user_content = kwargs['messages'][1]['content']
        
        self.assertIn("http://s2.com", user_content)
        self.assertNotIn("http://s1.com", user_content)

        print("ALL TESTS PASSED")

if __name__ == '__main__':
    unittest.main()
