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

"""NAT register function for the NVIDIA poem agent."""

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

from .agent import NvidiaPoemAgent
from .models import NvidiaPoemAgentState

logger = logging.getLogger(__name__)


class NvidiaPoemAgentConfig(FunctionBaseConfig, name="nvidia_poem_agent"):
    """Configuration for the NVIDIA poem agent."""

    llm: LLMRef = Field(..., description="LLM to use")
    max_history: int = Field(default=10, description="Maximum number of recent messages sent to the LLM")


@register_function(config_type=NvidiaPoemAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def nvidia_poem_agent(config: NvidiaPoemAgentConfig, builder: Builder):
    """NVIDIA poem agent."""
    llm = await builder.get_llm(config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    agent = NvidiaPoemAgent(llm=llm, max_history=config.max_history)

    async def _run(state: NvidiaPoemAgentState) -> NvidiaPoemAgentState:
        return await agent.run(state)

    yield FunctionInfo.from_fn(_run, description="Answers poem requests with an NVIDIA-themed poem.")


########################################################
# NVIDIA Poem Workflow (Wrapper for CLI / Evaluation)
########################################################
class NvidiaPoemWorkflowConfig(FunctionBaseConfig, name="nvidia_poem_workflow"):
    """Configuration for the NVIDIA poem workflow wrapper.

    This wrapper accepts a string query and converts it to messages
    for the nvidia_poem_agent.
    """

    relay: RelayConfig = Field(default_factory=RelayConfig, description="NeMo Relay plugins and export destinations")


@register_function(config_type=NvidiaPoemWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def nvidia_poem_workflow(config: NvidiaPoemWorkflowConfig, builder: Builder):
    """Wrapper workflow that accepts string queries."""
    await _ensure_relay_started(config.relay)
    nvidia_poem_agent_fn = await builder.get_function("nvidia_poem_agent")
    workflow_id = config.name or config.type

    async def _run(query: str) -> ChatResponse:
        """Run the NVIDIA poem agent on a query string."""
        result = await nvidia_poem_agent_fn.ainvoke(NvidiaPoemAgentState(messages=[HumanMessage(content=query)]))
        return _create_chat_response(result.messages[-1].content, response_id="poem_response", model=workflow_id)

    yield FunctionInfo.from_fn(_run, description="NVIDIA poem workflow (accepts string query).")
