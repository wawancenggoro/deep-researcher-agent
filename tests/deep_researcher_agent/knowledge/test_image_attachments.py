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

"""Tests for the pending image attachment registry."""

import pytest

from deep_researcher_agent.knowledge import image_attachments
from deep_researcher_agent.knowledge.image_attachments import ImageAttachment
from deep_researcher_agent.knowledge.image_attachments import claim_image_attachments
from deep_researcher_agent.knowledge.image_attachments import clear_image_attachments
from deep_researcher_agent.knowledge.image_attachments import image_mime_type
from deep_researcher_agent.knowledge.image_attachments import record_image_attachment


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_image_attachments()
    yield
    clear_image_attachments()


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [("cat.png", "image/png"), ("cat.JPG", "image/jpeg"), ("cat.jpeg", "image/jpeg"), ("notes.txt", None)],
)
def test_image_mime_type(file_name, expected):
    assert image_mime_type(file_name) == expected


def test_claim_returns_recorded_images_once():
    assert record_image_attachment("conv-1", "/tmp/a.png", "a.png")
    assert record_image_attachment("conv-1", "/tmp/b.jpg", "b.jpg")

    assert claim_image_attachments("conv-1") == [
        ImageAttachment(file_path="/tmp/a.png", file_name="a.png", mime_type="image/png"),
        ImageAttachment(file_path="/tmp/b.jpg", file_name="b.jpg", mime_type="image/jpeg"),
    ]
    assert claim_image_attachments("conv-1") == []


def test_sessions_are_isolated():
    record_image_attachment("conv-1", "/tmp/a.png", "a.png")

    assert claim_image_attachments("conv-2") == []
    assert len(claim_image_attachments("conv-1")) == 1


def test_non_image_is_not_recorded():
    assert not record_image_attachment("conv-1", "/tmp/notes.txt", "notes.txt")
    assert claim_image_attachments("conv-1") == []


def test_pending_images_are_bounded(monkeypatch):
    monkeypatch.setattr(image_attachments, "MAX_PENDING_IMAGES_PER_COLLECTION", 2)
    for name in ("a.png", "b.png", "c.png"):
        record_image_attachment("conv-1", f"/tmp/{name}", name)

    assert [a.file_name for a in claim_image_attachments("conv-1")] == ["b.png", "c.png"]
