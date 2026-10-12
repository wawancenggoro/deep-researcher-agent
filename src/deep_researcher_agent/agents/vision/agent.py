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

"""Vision agent: answers questions about uploaded images with a vision-language model."""

import base64
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from deep_researcher_agent.common import load_prompt

from .models import VisionAgentState
from .models import VisionImage

logger = logging.getLogger(__name__)

AGENT_DIR = Path(__file__).parent
DEFAULT_QUESTION = "Describe this image."
_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def encode_image_file(file_path: str, file_name: str, mime_type: str) -> VisionImage:
    """Read an image from disk and base64-encode it for the vision model."""
    data = Path(file_path).read_bytes()
    return VisionImage(file_name=file_name, mime_type=mime_type, data_base64=base64.b64encode(data).decode("ascii"))


def _message_text(content: Any) -> str:
    """Return the text of a message whose content is a string or a list of content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
            if not isinstance(part, dict) or part.get("type", "text") == "text"
        )
    return str(content or "")


class VisionAgent:
    """Single-call agent that sends the user's question and images to a vision-language model."""

    def __init__(self, llm: BaseChatModel, max_history: int = 10):
        self.llm = llm
        self.max_history = max_history
        self.system_prompt = load_prompt(AGENT_DIR / "prompts", "vision")

    def build_messages(self, state: VisionAgentState) -> list[Any]:
        """Build the model input: system prompt, prior text history, then the question with its images."""
        if not state.images:
            raise ValueError("Vision agent requires at least one image")
        history = state.messages[-self.max_history :] if self.max_history > 0 else state.messages
        context, latest = (history[:-1], history[-1]) if history else ([], None)
        question = _message_text(latest.content).strip() if isinstance(latest, HumanMessage) else ""
        if latest is not None and not isinstance(latest, HumanMessage):
            context = history

        content: list[dict[str, Any]] = [{"type": "text", "text": question or DEFAULT_QUESTION}]
        for image in state.images:
            content.append({"type": "text", "text": f"Image: {image.file_name}"})
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{image.mime_type};base64,{image.data_base64}"}}
            )
        return [SystemMessage(content=self.system_prompt), *context, HumanMessage(content=content)]

    async def run(self, state: VisionAgentState) -> VisionAgentState:
        """Answer the latest user message using the attached images."""
        response = await self.llm.ainvoke(self.build_messages(state))
        # Reasoning models normally return reasoning separately; strip inline blocks if one leaks.
        answer = _THINK_BLOCK.sub("", _message_text(response.content)).strip()
        # The images are not carried forward, so large base64 payloads never enter chat history.
        return VisionAgentState(messages=[*state.messages, AIMessage(content=answer)])
