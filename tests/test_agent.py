import os
import tempfile
import unittest

from src.agentic_ai import AgenticAI


class TestAgenticAI(unittest.TestCase):
    def test_plan_and_execute_returns_numbered_steps_without_api_key(self):
        agent = AgenticAI()
        plan = agent.plan_and_execute("Back up project folder", steps=3)
        self.assertIsInstance(plan, str)
        self.assertIn("1.", plan)
        self.assertIn("2.", plan)
        self.assertIn("3.", plan)

    def test_execute_tools_for_goal_runs_registered_tools(self):
        agent = AgenticAI()
        agent.register_tool("echo_tool", lambda goal: f"handled: {goal}")

        result = agent.execute_tools_for_goal("Backup the repo")

        self.assertIn("echo_tool", result)
        self.assertIn("handled: Backup the repo", result)

    def test_execute_workflow_builds_summary_state(self):
        agent = AgenticAI()
        agent.register_tool("echo_tool", lambda goal: f"handled: {goal}")

        state = agent.execute_workflow("Backup the repo")

        self.assertIn("final_summary", state)
        self.assertIn("Backup the repo", state["final_summary"])
        self.assertIn("echo_tool", state["final_summary"])

    def test_build_graph_invokes_real_state_graph(self):
        agent = AgenticAI()
        agent.register_tool("echo_tool", lambda goal: f"handled: {goal}")

        graph = agent.build_graph()
        result = graph.invoke({"goal": "Backup the repo"})

        self.assertIn("final_summary", result)
        self.assertIn("Backup the repo", result["final_summary"])
        self.assertIn("echo_tool", result["final_summary"])

    def test_memory_tracks_goal_history(self):
        agent = AgenticAI()

        agent.add_memory("Backup the repo")
        agent.add_memory("Validate the backup")

        self.assertEqual(len(agent.memory), 2)
        self.assertIn("Backup the repo", agent.memory)
        self.assertIn("Validate the backup", agent.memory)

    def test_memory_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = os.path.join(temp_dir, "agent_memory.json")
            agent = AgenticAI(memory_path=memory_path)
            agent.add_memory("Backup the repo")
            agent.persist_memory()

            reloaded = AgenticAI(memory_path=memory_path)
            self.assertIn("Backup the repo", reloaded.memory)

    def test_backup_directory_creates_backup_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = os.path.join(temp_dir, "source")
            backup_dir = os.path.join(temp_dir, "backup")
            os.makedirs(source_dir)
            with open(os.path.join(source_dir, "notes.txt"), "w", encoding="utf-8") as file_handle:
                file_handle.write("hello")

            agent = AgenticAI()
            result = agent.backup_directory(source_dir, backup_dir)

            self.assertIn("Backup complete", result)
            self.assertTrue(os.path.exists(backup_dir))
            self.assertTrue(os.path.exists(os.path.join(backup_dir, "notes.txt")))

    def test_execute_plan_runs_steps_in_sequence(self):
        agent = AgenticAI()
        agent.register_tool("prepare", lambda goal: f"prepared: {goal}")
        agent.register_tool("validate", lambda goal: f"validated: {goal}")

        result = agent.execute_plan("Backup the repo", steps=2)

        self.assertIn("Step 1", result)
        self.assertIn("prepare", result)
        self.assertIn("Step 2", result)
        self.assertIn("validate", result)

    def test_execute_plan_retries_failed_tools(self):
        agent = AgenticAI()
        attempts = {"fail_then_succeed": 0}

        def fail_then_succeed(goal):
            attempts["fail_then_succeed"] += 1
            if attempts["fail_then_succeed"] == 1:
                raise RuntimeError("transient failure")
            return f"succeeded: {goal}"

        agent.register_tool("fail_then_succeed", fail_then_succeed)

        result = agent.execute_plan("Backup the repo", steps=1, retries=1)

        self.assertIn("retry", result.lower())
        self.assertIn("succeeded: Backup the repo", result)
        self.assertEqual(attempts["fail_then_succeed"], 2)

    def test_execute_task_loop_reports_status(self):
        agent = AgenticAI()
        agent.register_tool("prepare", lambda goal: f"prepared: {goal}")

        result = agent.execute_task_loop("Backup the repo", steps=1, retries=1)

        self.assertIn("status", result)
        self.assertIn("execution_report", result)
        self.assertIn("prepared: Backup the repo", result["execution_report"])
        self.assertEqual(result["status"], "completed")


if __name__ == "__main__":
    unittest.main()
