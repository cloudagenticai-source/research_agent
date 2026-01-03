
import unittest
from unittest.mock import patch, MagicMock
import report_writer

class TestReportWriter(unittest.TestCase):

    @patch('report_writer.memory_truth')
    @patch('report_writer.memory_vector.VectorMemory')
    @patch('report_writer.OpenAI')
    @patch('report_writer.router.retrieve_router')
    def test_citation_formatting_validation(self, mock_router, mock_openai, mock_vm_cls, mock_truth):
        # 1. Setup Data
        mock_router.return_value = {
            'procedural': {},
            'episodic': {'ids': ['episode:1']}, # Topic Scoped ID
            'semantic': {}
        }
        
        mock_truth.get_latest_session_id.return_value = "session-123"
        mock_truth.get_episodes_by_topic_and_session.return_value = [
            {'id': 1, 'topic': 'Target', 'title': 'Valid', 'url': 'http://valid.com', 'notes': 'Valid'}
        ]
        mock_truth.get_facts_by_ids.return_value = []
        mock_truth.get_facts_by_topic_and_session.return_value = []

        mock_client = mock_openai.return_value
        mock_resp = MagicMock()
        
        # LLM Output with various citation styles
        llm_output = (
            "Report start.\n"
            "1. Valid Markdown: [Link](http://valid.com)\n"
            "2. Invalid Markdown: [Bad](http://invalid.com)\n"
            "3. Valid Paren: (http://valid.com)\n"
            "4. Invalid Paren: (http://invalid.com)\n"
            "5. Valid Bare: http://valid.com is good.\n"
            "6. Invalid Bare: http://invalid.com is bad.\n"
            "7. Bare Punctuation: http://valid.com."
        )
        mock_resp.choices[0].message.content = llm_output
        mock_client.chat.completions.create.return_value = mock_resp

        # 2. Execution
        print("Testing citation formatting validation...")
        report = report_writer.generate_report("Target")
        
        print("Generated Report Snippet:")
        print(report)

        # 3. Validations
        
        # 1. Markdown
        self.assertIn("[Link](http://valid.com)", report)
        self.assertIn("[Bad]([citation unavailable])", report)
        
        # 2. Paren
        self.assertIn("(http://valid.com)", report)
        self.assertIn("([citation unavailable])", report) # Replaced inside parens? 
        # Actually logic is: re.sub(r'\((http...)\)', replace_paren) -> replace_paren returns either "(url)" or "[citation unavailable]"
        # So it replaces the WHOLE "(http...)" group.
        # Let's check logic: r'\((http[^)]+)\)'
        # If match.group(1) valid -> return match.group(0) -> "(http://valid.com)"
        # If invalid -> return "[citation unavailable]"
        # So we expect "[citation unavailable]" NOT "([citation unavailable])"
        self.assertIn("[citation unavailable]", report)
        self.assertNotIn("(http://invalid.com)", report)

        # 3. Bare
        self.assertIn("http://valid.com is good", report)
        # Invalid bare should be redacted
        self.assertNotIn("http://invalid.com is bad", report)
        
        # 4. Punctuation
        self.assertIn("http://valid.com.", report)

        print("ALL TESTS PASSED")

if __name__ == '__main__':
    unittest.main()
