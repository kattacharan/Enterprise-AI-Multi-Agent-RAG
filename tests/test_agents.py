import unittest
from backend.app.agents.graph import execute_multi_agent_flow
from backend.app.database.session import init_db
from backend.app.rag.ingest import seed_sample_documents


class TestAgentWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        seed_sample_documents()

    def test_conversational_greeting_flow(self):
        """Verify Planner routes conversational greetings directly without FAISS retrieval."""
        result = execute_multi_agent_flow("Hello! How can you help me?")
        self.assertIn("response", result)
        self.assertFalse(result.get("requires_retrieval"))
        self.assertTrue(len(result.get("steps_executed", [])) >= 2)

    def test_knowledge_retrieval_rag_flow(self):
        """Verify full 4-agent execution: Planner -> Retriever -> Verifier -> Synthesizer."""
        query = "What is the equipment reimbursement allowance?"
        result = execute_multi_agent_flow(query)
        
        self.assertTrue(result.get("requires_retrieval"))
        self.assertIn("response", result)
        self.assertTrue(len(result["response"]) > 0)
        
        # Verify citations
        citations = result.get("citations", [])
        self.assertTrue(len(citations) > 0)
        self.assertTrue(any("policy" in c["filename"] or "company_policies.md" in c["filename"] for c in citations))

        # Verify agent execution steps trace
        steps = result.get("steps_executed", [])
        agent_names = [s["agent"] for s in steps]
        self.assertIn("Planner Agent", agent_names)
        self.assertIn("Retriever Agent", agent_names)
        self.assertIn("Verifier Agent", agent_names)
        self.assertIn("Synthesizer Agent", agent_names)


if __name__ == "__main__":
    unittest.main()
