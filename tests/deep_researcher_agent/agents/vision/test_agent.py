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

"""Tests for the vision agent."""

import base64
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from deep_researcher_agent.agents.vision.agent import DEFAULT_QUESTION
from deep_researcher_agent.agents.vision.agent import VisionAgent
from deep_researcher_agent.agents.vision.agent import encode_image_file
from deep_researcher_agent.agents.vision.models import VisionAgentState
from deep_researcher_agent.agents.vision.models import VisionImage
from deep_researcher_agent.agents.vision.register import VisionAgentConfig

PNG_IMAGE = VisionImage(file_name="cat.png", mime_type="image/png", data_base64="iVBORw0KGgo=")


def _mock_llm(reply: str) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=reply))
    return llm


def test_config_defaults():
    config = VisionAgentConfig(llm="vlm")
    assert config.max_history == 10


def test_encode_image_file(tmp_path):
    image = tmp_path / "cat.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nbytes")

    encoded = encode_image_file(str(image), "cat.png", "image/png")

    assert encoded.file_name == "cat.png"
    assert encoded.mime_type == "image/png"
    assert base64.b64decode(encoded.data_base64) == b"\x89PNG\r\n\x1a\nbytes"


def test_build_messages_sends_question_and_base64_image():
    agent = VisionAgent(llm=_mock_llm(""))
    state = VisionAgentState(messages=[HumanMessage(content="What breed is this?")], images=[PNG_IMAGE])

    messages = agent.build_messages(state)

    assert isinstance(messages[0], SystemMessage)
    content = messages[-1].content
    assert content[0] == {"type": "text", "text": "What breed is this?"}
    assert content[1] == {"type": "text", "text": "Image: cat.png"}
    assert content[2] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}


def test_build_messages_keeps_prior_text_history():
    agent = VisionAgent(llm=_mock_llm(""))
    state = VisionAgentState(
        messages=[HumanMessage(content="Hi"), AIMessage(content="Hello!"), HumanMessage(content="Describe it")],
        images=[PNG_IMAGE],
    )

    messages = agent.build_messages(state)

    assert [m.content for m in messages[1:3]] == ["Hi", "Hello!"]
    assert messages[-1].content[0]["text"] == "Describe it"


def test_build_messages_uses_default_question_for_empty_text():
    agent = VisionAgent(llm=_mock_llm(""))
    state = VisionAgentState(messages=[HumanMessage(content="  ")], images=[PNG_IMAGE])

    assert agent.build_messages(state)[-1].content[0]["text"] == DEFAULT_QUESTION


def test_build_messages_requires_an_image():
    agent = VisionAgent(llm=_mock_llm(""))

    with pytest.raises(ValueError):
        agent.build_messages(VisionAgentState(messages=[HumanMessage(content="Describe it")]))


async def test_run_returns_answer_without_images_or_reasoning():
    agent = VisionAgent(llm=_mock_llm("<think>looking closely</think>\nA tabby cat."))
    state = VisionAgentState(messages=[HumanMessage(content="What is this?")], images=[PNG_IMAGE])

    result = await agent.run(state)

    assert result.messages[-1].content == "A tabby cat."
    assert result.images == []


def test_image_data_is_excluded_from_repr():
    state = VisionAgentState(messages=[HumanMessage(content="What is this?")], images=[PNG_IMAGE])

    assert "iVBORw0KGgo=" not in repr(state)
    assert "cat.png" in repr(state)
