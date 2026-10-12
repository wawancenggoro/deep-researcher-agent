# tests/deep_researcher_agent/agents/nvidia_poem/test_agent.py

"""Tests for nvidia poem agent."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from deep_researcher_agent.agents.nvidia_poem.agent import NvidiaPoemAgent
from deep_researcher_agent.agents.nvidia_poem.models import NvidiaPoemAgentState
from deep_researcher_agent.agents.nvidia_poem.register import NvidiaPoemAgentConfig


def _mock_llm(reply: str) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=reply))
    return llm


def test_config_defaults():
    config = NvidiaPoemAgentConfig(llm="test_llm")
    assert config.max_history == 10


async def test_run_appends_llm_reply():
    agent = NvidiaPoemAgent(llm=_mock_llm("  the answer  "))

    result = await agent.run(NvidiaPoemAgentState(messages=[HumanMessage(content="question")]))

    assert result.messages[-1].content == "the answer"


async def test_run_sends_system_prompt_first():
    llm = _mock_llm("reply")
    agent = NvidiaPoemAgent(llm=llm)

    await agent.run(NvidiaPoemAgentState(messages=[HumanMessage(content="question")]))

    sent = llm.ainvoke.call_args.args[0]
    assert isinstance(sent[0], SystemMessage)
    assert sent[-1].content == "question"
