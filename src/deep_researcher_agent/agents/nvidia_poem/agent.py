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

"""NVIDIA poem agent: answers poem requests with an NVIDIA-themed poem."""

import logging
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage

from deep_researcher_agent.common import load_prompt

from .models import NvidiaPoemAgentState

logger = logging.getLogger(__name__)

AGENT_DIR = Path(__file__).parent


class NvidiaPoemAgent:
    """Single-call agent that tells NVIDIA poems and declines anything else."""

    def __init__(self, llm: BaseChatModel, max_history: int = 10):
        self.llm = llm
        self.max_history = max_history
        self.system_prompt = load_prompt(AGENT_DIR / "prompts", "poem")

    async def run(self, state: NvidiaPoemAgentState) -> NvidiaPoemAgentState:
        """Generate a reply to the latest user message."""
        history = state.messages[-self.max_history :] if self.max_history > 0 else state.messages
        response = await self.llm.ainvoke([SystemMessage(content=self.system_prompt), *history])
        content = response.content if isinstance(response.content, str) else str(response.content)
        return NvidiaPoemAgentState(messages=[*state.messages, AIMessage(content=content.strip())])
