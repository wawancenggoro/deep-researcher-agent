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

"""Tests that the LlamaIndex ingestor records kept image uploads for the vision agent."""

from unittest.mock import patch

import pytest
from knowledge_layer.llamaindex import adapter as llamaindex_adapter

from deep_researcher_agent.knowledge.image_attachments import claim_image_attachments
from deep_researcher_agent.knowledge.image_attachments import clear_image_attachments


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_image_attachments()
    yield
    clear_image_attachments()


def test_submit_job_records_images_by_collection(tmp_path):
    image = tmp_path / "deep-researcher-upload-1.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    document = tmp_path / "deep-researcher-upload-2.txt"
    document.write_text("notes")

    with patch.object(llamaindex_adapter.LlamaIndexIngestor, "_start_ttl_cleanup_task"):
        ingestor = llamaindex_adapter.LlamaIndexIngestor(config={"persist_dir": str(tmp_path / "chroma")})
    # The background ingestion thread is not needed to record the image.
    with patch.object(llamaindex_adapter.threading, "Thread"):
        ingestor.submit_job(
            [str(image), str(document)],
            "conv-1",
            {"original_filenames": ["cat.png", "notes.txt"]},
        )

    attachments = claim_image_attachments("conv-1")
    assert [(a.file_path, a.file_name, a.mime_type) for a in attachments] == [(str(image), "cat.png", "image/png")]
