
import unittest
from unittest.mock import patch, MagicMock
import json
import research_agent

class TestResearchAgent(unittest.TestCase):

    @patch('research_agent.memory_truth')
    @patch('research_agent.memory_vector.VectorMemory')
    @patch('research_agent.OpenAI')
    @patch('research_agent.router.retrieve_router')
    @patch('research_agent.web_search.search_web')
    @patch('research_agent.web_fetch.fetch_page')
    def test_run_research_flow(self, mock_fetch, mock_search, mock_router, mock_openai, mock_vm_cls, mock_truth):
        # 1. Setup Mock Router Retrieval
        mock_router.return_value = {
            'procedural': {'ids': ['skill.test']},
            'episodic': {},
            'semantic': {}
        }

        # 2. Setup Mock OpenAI (Subquestions -> Summary -> Facts)
        mock_client = mock_openai.return_value
        
        # We need to simulate different responses based on the call context (roughly)
        # Using a side_effect is robust.
        def openai_side_effect(**kwargs):
            mock_formatted_resp = MagicMock()
            messages = kwargs.get('messages', [])
            content = messages[0]['content']
            
            if "generate 3-6 specific sub-questions" in content:
                # Subquestions response
                mock_formatted_resp.choices[0].message.content = json.dumps({
                    "questions": ["q1", "q2"]
                })
            elif "Summarize the following text" in content:
                # Summary response
                mock_formatted_resp.choices[0].message.content = "This is a summary of the page."
            elif "Extract 5-12 key semantic facts" in content:
                # Facts response
                mock_formatted_resp.choices[0].message.content = json.dumps({
                    "facts": [
                        {"subject": "S", "predicate": "P", "object": "O", "confidence": 0.9}
                    ]
                })
            return mock_formatted_resp

        mock_client.chat.completions.create.side_effect = openai_side_effect

        # 3. Setup Mock Web Search
        mock_search.return_value = [
            {'link': 'http://example.com/1', 'title': 'Result 1'},
            {'link': 'http://example.com/2', 'title': 'Result 2'}
        ]

        # 4. Setup Mock Web Fetch
        mock_fetch.return_value = {
            'text': "Full page text content...",
            'title': "Page Title",
            'url': "http://example.com/1"
        }

        # 5. Setup Mock DB/Vector
        mock_truth.add_episode.return_value = 101
        mock_truth.add_fact.return_value = 202
        mock_truth.get_episode.return_value = {'topic': 'test', 'title': 'title', 'notes': 'notes'}
        mock_truth.get_fact.return_value = {'topic': 'test', 'subject': 's', 'predicate': 'p', 'object': 'o'}
        
        mock_vm = mock_vm_cls.return_value

        # --- Run ---
        print("Testing run_research execution...")
        trace = research_agent.run_research("Test Topic", max_sources=2)

        # --- Verify ---
        
        # Check trace structure
        self.assertEqual(trace['topic'], "Test Topic")
        self.assertEqual(trace['selected_skill'], "skill.test")
        self.assertEqual(trace['subquestions'], ["q1", "q2"])
        
        # Check source collection
        # We asked for max 2 sources, mock search returns 2 per query (2 queries), set logic handles uniqueness
        self.assertTrue(len(trace['sources_used']) > 0)
        
        # Check Episodes added (1 per source used)
        self.assertTrue(len(trace['episode_ids']) > 0)
        mock_truth.add_episode.assert_called()
        mock_vm.upsert_episode.assert_called()

        # Check Facts added
        self.assertTrue(len(trace['fact_ids']) > 0)
        mock_truth.add_fact.assert_called()
        mock_vm.upsert_fact.assert_called()

        print("ALL TESTS PASSED")

if __name__ == '__main__':
    unittest.main()
