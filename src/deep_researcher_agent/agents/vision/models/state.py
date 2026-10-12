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

"""State models for the vision agent."""

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from pydantic import Field


class VisionImage(BaseModel):
    """
    An image passed to the vision model.

    Attributes:
        file_name: Original filename, used to label the image for the model.
        mime_type: Image MIME type, for example ``image/png``.
        data_base64: Base64-encoded image bytes. Excluded from ``repr`` so image
            content never reaches logs (NAT logs function inputs on error).
    """

    file_name: str
    mime_type: str
    data_base64: str = Field(repr=False)


class VisionAgentState(BaseModel):
    """
    State for the vision agent.

    Attributes:
        messages: Conversation history with LangGraph message reducer. The latest
            human message is the user's question about the images.
        images: Images attached to the latest message.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    images: list[VisionImage] = Field(default_factory=list)
