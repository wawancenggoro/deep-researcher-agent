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

"""NAT register function for the vision agent."""

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from .agent import VisionAgent
from .models import VisionAgentState

logger = logging.getLogger(__name__)


class VisionAgentConfig(FunctionBaseConfig, name="vision_agent"):
    """Configuration for the vision agent."""

    llm: LLMRef = Field(..., description="Vision-language model to use; must accept image inputs")
    max_history: int = Field(default=10, description="Maximum number of recent messages sent to the model")


@register_function(config_type=VisionAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def vision_agent(config: VisionAgentConfig, builder: Builder):
    """Vision agent."""
    llm = await builder.get_llm(config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    agent = VisionAgent(llm=llm, max_history=config.max_history)

    async def _run(state: VisionAgentState) -> VisionAgentState:
        return await agent.run(state)

    yield FunctionInfo.from_fn(_run, description="Answers questions about uploaded images with a vision model.")
