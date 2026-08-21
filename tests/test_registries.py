from guard_arch import PROJECT_ROOT
from guard_arch.core.agent import AgentRegistry
from guard_arch.core.context import ContextEngine
from guard_arch.core.memory import MemoryManager
from guard_arch.core.skill import SkillRegistry
from guard_arch.core.workspace import Workspace

AGENTS_DIR = PROJECT_ROOT / "agents"
SKILLS_DIR = PROJECT_ROOT / "skills"


def test_default_agent_loads():
    registry = AgentRegistry([AGENTS_DIR])
    agent = registry.get("assistant")
    assert agent.name == "通用助手"
    assert agent.model == "default"
    assert "coding" in agent.skills and "research" in agent.skills
    assert "read_file" in agent.tools and "run_command" in agent.tools
    assert "中文" in agent.instructions


def test_default_skills_load():
    registry = SkillRegistry([SKILLS_DIR])
    coding = registry.get("coding")
    research = registry.get("research")
    assert coding.description
    assert "read_file" in coding.tools
    assert "先读后写" in coding.instructions
    assert "引用来源" in research.instructions
    assert "run_command" not in research.tools


def test_skill_instructions_injected_into_prompt(tmp_path):
    memory = MemoryManager(tmp_path)
    agent = AgentRegistry([AGENTS_DIR]).get("assistant")
    skills_registry = SkillRegistry([SKILLS_DIR])
    skills = [skills_registry.get(name) for name in agent.skills]
    prompt = ContextEngine().build_system_prompt(agent, skills, memory, Workspace(tmp_path))
    assert "## Skill: coding" in prompt
    assert "先读后写" in prompt
    assert "## Skill: research" in prompt
    assert agent.instructions.strip().splitlines()[0] in prompt
    assert "## Environment" in prompt
    memory.close()


def test_memory_injected_into_prompt(tmp_path):
    memory = MemoryManager(tmp_path)
    memory.remember("project", "test_command", "pnpm test")
    agent = AgentRegistry([AGENTS_DIR]).get("assistant")
    prompt = ContextEngine().build_system_prompt(agent, [], memory, Workspace(tmp_path))
    assert "[project memory]" in prompt
    assert "test_command: pnpm test" in prompt
    memory.close()
