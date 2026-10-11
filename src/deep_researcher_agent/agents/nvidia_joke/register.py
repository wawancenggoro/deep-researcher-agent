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

"""NAT register function for the NVIDIA joke agent."""

import logging

from langchain_core.messages import HumanMessage
from pydantic import Field

from deep_researcher_agent.common import _create_chat_response
from deep_researcher_agent.relay.bootstrap import ensure_started as _ensure_relay_started
from deep_researcher_agent.relay.config import RelayConfig
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatResponse
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from .agent import NvidiaJokeAgent
from .models import NvidiaJokeAgentState

logger = logging.getLogger(__name__)


class NvidiaJokeAgentConfig(FunctionBaseConfig, name="nvidia_joke_agent"):
    """Configuration for the NVIDIA joke agent."""

    llm: LLMRef = Field(..., description="LLM to use")
    max_history: int = Field(default=10, description="Maximum number of recent messages sent to the LLM")


@register_function(config_type=NvidiaJokeAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def nvidia_joke_agent(config: NvidiaJokeAgentConfig, builder: Builder):
    """NVIDIA joke agent."""
    llm = await builder.get_llm(config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    agent = NvidiaJokeAgent(llm=llm, max_history=config.max_history)

    async def _run(state: NvidiaJokeAgentState) -> NvidiaJokeAgentState:
        return await agent.run(state)

    yield FunctionInfo.from_fn(_run, description="Answers joke requests with an NVIDIA-themed joke.")


########################################################
# NVIDIA Joke Workflow (Wrapper for CLI / Evaluation)
########################################################
class NvidiaJokeWorkflowConfig(FunctionBaseConfig, name="nvidia_joke_workflow"):
    """Configuration for the NVIDIA joke workflow wrapper.

    This wrapper accepts a string query and converts it to messages
    for the nvidia_joke_agent.
    """

    relay: RelayConfig = Field(default_factory=RelayConfig, description="NeMo Relay plugins and export destinations")


@register_function(config_type=NvidiaJokeWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def nvidia_joke_workflow(config: NvidiaJokeWorkflowConfig, builder: Builder):
    """Wrapper workflow that accepts string queries."""
    await _ensure_relay_started(config.relay)
    nvidia_joke_agent_fn = await builder.get_function("nvidia_joke_agent")
    workflow_id = config.name or config.type

    async def _run(query: str) -> ChatResponse:
        """Run the NVIDIA joke agent on a query string."""
        result = await nvidia_joke_agent_fn.ainvoke(NvidiaJokeAgentState(messages=[HumanMessage(content=query)]))
        return _create_chat_response(result.messages[-1].content, response_id="joke_response", model=workflow_id)

    yield FunctionInfo.from_fn(_run, description="NVIDIA joke workflow (accepts string query).")
