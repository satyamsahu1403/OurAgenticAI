import json
import os
import shutil
from typing import Callable, Dict, List, Optional

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - depends on optional package
    ChatOpenAI = None

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - package optional for demo mode
    StateGraph = None
    END = "END"


class AgenticAI:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0, memory_path: Optional[str] = None):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = None
        self.tools: Dict[str, Callable[..., str]] = {}
        self.memory: List[str] = []
        self.memory_path = memory_path or os.path.join(os.getcwd(), ".agent_memory.json")
        self._load_memory()

        if ChatOpenAI is not None and self.api_key:
            self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature, api_key=self.api_key)

    def _load_memory(self) -> None:
        if not os.path.exists(self.memory_path):
            self.memory = []
            return
        try:
            with open(self.memory_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.memory = data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            self.memory = []

    def persist_memory(self) -> None:
        directory = os.path.dirname(self.memory_path) or "."
        os.makedirs(directory, exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as handle:
            json.dump(self.memory, handle, indent=2)

    def add_memory(self, goal: str) -> None:
        if goal and goal not in self.memory:
            self.memory.append(goal)
            self.persist_memory()

    def register_tool(self, name: str, func: Callable[..., str]) -> None:
        self.tools[name] = func

    def backup_directory(self, source_dir: str, destination_dir: str) -> str:
        if not os.path.isdir(source_dir):
            return f"Backup failed: source directory not found: {source_dir}"

        if os.path.exists(destination_dir):
            shutil.rmtree(destination_dir)

        shutil.copytree(source_dir, destination_dir)
        return f"Backup complete: {source_dir} -> {destination_dir}"

    def _fallback_plan(self, goal: str, steps: int) -> str:
        count = max(1, int(steps))
        bullets = [
            f"Define the objective and success criteria for '{goal}'.",
            "Gather the required inputs, tools, and constraints needed to complete the work.",
            "Execute the task in controlled steps and validate each milestone before moving on.",
            "Record the outcome, flag follow-up items, and plan the next action.",
        ]

        result = []
        for index in range(1, count + 1):
            detail = bullets[min(index - 1, len(bullets) - 1)]
            result.append(f"{index}. {detail}")
        return "\n".join(result)

    def run(self, prompt_text: str) -> str:
        if self.llm is None:
            return self._fallback_plan(prompt_text, 3)

        response = self.llm.invoke(prompt_text)
        return getattr(response, "content", str(response)).strip()

    def execute_tools_for_goal(self, goal: str) -> str:
        if not self.tools:
            return self._fallback_plan(goal, 3)

        tool_results = []
        for name, tool in self.tools.items():
            try:
                result = tool(goal)
                tool_results.append(f"- {name}: {result}")
            except Exception as exc:
                tool_results.append(f"- {name}: error: {exc}")

        summary = [
            f"Goal: {goal}",
            "Tool execution summary:",
            *tool_results,
        ]
        return "\n".join(summary)

    def _execute_tool_plan(self, goal: str, steps: int) -> str:
        available_tools = ", ".join(sorted(self.tools)) if self.tools else "none"
        if not self.tools:
            return self._fallback_plan(goal, steps)

        plan = [
            f"1. Identify the objective: {goal}",
            f"2. Select the best available tool(s): {available_tools}",
            f"3. Execute the task and validate results before finalizing.",
        ]
        plan.append("4. Tool result summary:")
        plan.append(self.execute_tools_for_goal(goal))
        return "\n".join(plan)

    def execute_workflow(self, goal: str) -> dict:
        self.add_memory(goal)
        state = {
            "goal": goal,
            "plan": self._fallback_plan(goal, 3),
            "tool_results": "",
            "final_summary": "",
        }

        state["plan"] = self.plan_and_execute(goal, steps=3)
        state["tool_results"] = self.execute_tools_for_goal(goal) if self.tools else "No tools registered."
        state["final_summary"] = (
            f"Workflow complete for '{goal}'. \n"
            f"Plan:\n{state['plan']}\n\nTool results:\n{state['tool_results']}"
        )
        return state

    def build_graph(self):
        if StateGraph is None:
            def fallback_graph(state):
                state = dict(state)
                goal = state.get("goal", "")
                self.add_memory(goal)
                state["plan"] = self.plan_and_execute(goal, steps=3)
                state["tool_results"] = self.execute_tools_for_goal(goal) if self.tools else "No tools registered."
                state["final_summary"] = (
                    f"Workflow complete for '{goal}'.\n"
                    f"Plan:\n{state['plan']}\n\nTool results:\n{state['tool_results']}"
                )
                return state

            return fallback_graph

        builder = StateGraph(dict)

        def planner_node(state):
            state = dict(state)
            goal = state.get("goal", "")
            self.add_memory(goal)
            state["plan"] = self.plan_and_execute(goal, steps=3)
            return state

        def tool_node(state):
            state = dict(state)
            goal = state.get("goal", "")
            state["tool_results"] = self.execute_tools_for_goal(goal) if self.tools else "No tools registered."
            return state

        def summarizer_node(state):
            state = dict(state)
            goal = state.get("goal", "")
            state["final_summary"] = (
                f"Workflow complete for '{goal}'.\n"
                f"Plan:\n{state['plan']}\n\nTool results:\n{state['tool_results']}"
            )
            return state

        builder.add_node("planner", planner_node)
        builder.add_node("tool_executor", tool_node)
        builder.add_node("summarizer", summarizer_node)
        builder.add_edge("planner", "tool_executor")
        builder.add_edge("tool_executor", "summarizer")
        builder.add_edge("summarizer", END)
        builder.set_entry_point("planner")
        return builder.compile()

    def execute_plan(self, goal: str, steps: Optional[int] = 3, retries: int = 0) -> str:
        step_count = max(1, int(steps or 3))
        tool_names = list(self.tools.keys())

        if not tool_names:
            return self._fallback_plan(goal, step_count)

        step_lines = []
        for index in range(1, step_count + 1):
            tool_name = tool_names[(index - 1) % len(tool_names)]
            tool = self.tools[tool_name]
            attempts = 0
            while True:
                try:
                    result = tool(goal)
                    break
                except Exception as exc:
                    attempts += 1
                    if attempts > retries:
                        result = f"error: {exc}"
                        break
                    step_lines.append(f"Step {index}: retrying {tool_name} after failure ({attempts}/{retries})")

            step_lines.append(f"Step {index}: {tool_name} -> {result}")

        return "\n".join(step_lines)

    def execute_task_loop(self, goal: str, steps: Optional[int] = 3, retries: int = 0) -> dict:
        self.add_memory(goal)
        execution_report = self.execute_plan(goal, steps=steps, retries=retries)
        status = "completed" if "error:" not in execution_report.lower() else "failed"
        return {
            "goal": goal,
            "status": status,
            "execution_report": execution_report,
            "memory": list(self.memory),
        }

    def plan_and_execute(self, goal: str, steps: Optional[int] = 3) -> str:
        step_count = max(1, int(steps or 3))
        planner_prompt = (
            f"You are an autonomous agent. Given the goal: {goal} produce {step_count} numbered actionable steps. "
            "Keep the output concise and directly actionable."
        )

        if self.llm is None:
            if self.tools:
                return self.execute_plan(goal, step_count)
            return self._fallback_plan(goal, step_count)

        try:
            plan = self.run(planner_prompt)
        except Exception:
            plan = self._fallback_plan(goal, step_count)

        if self.tools:
            return self.execute_plan(goal, step_count)

        if "1." not in plan and "2." not in plan and "3." not in plan:
            return self._fallback_plan(goal, step_count)

        return plan
