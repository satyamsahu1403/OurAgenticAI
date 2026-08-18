import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from src.agentic_ai import AgenticAI
except ImportError:  # pragma: no cover - support running as a direct package
    from agentic_ai import AgenticAI


def list_project_files(goal: str) -> str:
    project_root = Path(__file__).resolve().parents[1]
    files = [str(path.relative_to(project_root)) for path in sorted(project_root.rglob('*')) if path.is_file()][:10]
    return f"Discovered files for '{goal}': {', '.join(files) if files else 'none'}"


def summarize_goal(goal: str) -> str:
    return f"Prepared a structured plan for: {goal}"


def backup_project_repo(goal: str) -> str:
    project_root = Path(__file__).resolve().parents[1]
    source_dir = str(project_root)
    destination_dir = str(project_root / "backup_snapshot")
    agent = AgenticAI()
    return agent.backup_directory(source_dir, destination_dir)


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set. Falling back to the built-in planner.")

    agent = AgenticAI(memory_path=str(Path(__file__).resolve().parents[1] / ".agent_memory.json"))
    agent.register_tool("list_project_files", list_project_files)
    agent.register_tool("summarize_goal", summarize_goal)
    agent.register_tool("backup_project_repo", backup_project_repo)

    print("AgenticAI CLI")
    print("Type 'exit' to quit.")

    while True:
        goal = input("What do you want the agent to do? \n> ").strip()
        if not goal:
            print("Please enter a goal.")
            continue
        if goal.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        print("Planning...")
        result = agent.execute_workflow(goal)
        print(result["final_summary"])
        print(f"Memory: {agent.memory}")


if __name__ == "__main__":
    main()
