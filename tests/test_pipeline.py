"""
Local Test Suite: SEI CAD-to-LLM Telemetry Verification
Run this test to validate JSON B-Rep parsing and Gemini schema orchestration.
"""

import os
import json
import unittest
from src.orchestrator.sei_orchestrator import SEIOrchestrator


class TestSEIPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_payload_path = "sample_brep.json"
        self.orchestrator = SEIOrchestrator(
            gemini_model="gemini-2.5-pro",
            comfy_host="127.0.0.1:8188"
        )

    def test_sample_payload_exists(self):
        """Verify the test B-Rep payload is present and valid JSON."""
        self.assertTrue(os.path.exists(self.sample_payload_path))
        with open(self.sample_payload_path, "r") as f:
            data = json.load(f)
        self.assertIn("geometry", data)
        self.assertEqual(data["geometry"]["type"], "B-Rep")

    def test_prompt_formatting(self):
        """Verify prompt construction encapsulates drift threshold requirements."""
        with open(self.sample_payload_path, "r") as f:
            brep_data = json.load(f)
        prompt = self.orchestrator.tokenization_prompt(brep_data)
        self.assertIn("Visual drift tolerance is strictly < 0.5%", prompt)
        self.assertIn("B-Rep Metadata:", prompt)


if __name__ == "__main__":
    unittest.main()
