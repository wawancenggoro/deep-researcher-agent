# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the NVIDIA joke agent."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from deep_researcher_agent.agents.nvidia_joke.agent import NvidiaJokeAgent
from deep_researcher_agent.agents.nvidia_joke.models import NvidiaJokeAgentState
from deep_researcher_agent.agents.nvidia_joke.register import NvidiaJokeAgentConfig


def _mock_llm(reply: str) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=reply))
    return llm


class TestNvidiaJokeAgentConfig:
    def test_defaults(self):
        config = NvidiaJokeAgentConfig(llm="test_llm")

        assert config.llm == "test_llm"
        assert config.max_history == 10


class TestNvidiaJokeAgent:
    def test_prompt_loads(self):
        agent = NvidiaJokeAgent(llm=_mock_llm(""))

        assert "NVIDIA" in agent.system_prompt
        assert "joke" in agent.system_prompt

    async def test_run_appends_llm_reply(self):
        llm = _mock_llm("  Why did the GPU go to therapy? Too many unresolved threads.  ")
        agent = NvidiaJokeAgent(llm=llm)

        result = await agent.run(NvidiaJokeAgentState(messages=[HumanMessage(content="Tell me a joke")]))

        assert len(result.messages) == 2
        assert isinstance(result.messages[-1], AIMessage)
        assert result.messages[-1].content == "Why did the GPU go to therapy? Too many unresolved threads."

    async def test_run_sends_system_prompt_first(self):
        llm = _mock_llm("joke")
        agent = NvidiaJokeAgent(llm=llm)

        await agent.run(NvidiaJokeAgentState(messages=[HumanMessage(content="Tell me a joke")]))

        sent = llm.ainvoke.call_args.args[0]
        assert isinstance(sent[0], SystemMessage)
        assert sent[0].content == agent.system_prompt
        assert sent[-1].content == "Tell me a joke"

    async def test_run_truncates_history(self):
        llm = _mock_llm("joke")
        agent = NvidiaJokeAgent(llm=llm, max_history=2)
        messages = [HumanMessage(content=f"msg {i}") for i in range(5)]

        result = await agent.run(NvidiaJokeAgentState(messages=messages))

        sent = llm.ainvoke.call_args.args[0]
        assert [m.content for m in sent[1:]] == ["msg 3", "msg 4"]
        assert len(result.messages) == 6
