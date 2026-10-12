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

"""Pending image attachments per chat session.

Standalone image uploads are kept on disk instead of being ingested into the
knowledge base. Backends record them here by collection name (the chat
conversation id), and the next chat turn in that session claims them for the
vision agent.

The registry is process-local: the chat turn must run in the same process that
accepted the upload, which holds for single-replica deployments.
"""

import logging
import threading
from pathlib import Path

from pydantic import BaseModel

from deep_researcher_agent.common.logging_utils import log_identifier_ref

logger = logging.getLogger(__name__)

IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

# Bounds memory and per-turn VLM payload when images are uploaded but never sent.
MAX_PENDING_IMAGES_PER_COLLECTION = 8


class ImageAttachment(BaseModel):
    """An uploaded image waiting to be answered by the vision agent.

    Attributes:
        file_path: Server-side path of the kept upload.
        file_name: Original filename shown to the user.
        mime_type: Image MIME type derived from the filename extension.
    """

    file_path: str
    file_name: str
    mime_type: str


_lock = threading.Lock()
_pending: dict[str, list[ImageAttachment]] = {}


def image_mime_type(file_name: str) -> str | None:
    """Return the MIME type for a supported image filename, or None."""
    return IMAGE_MIME_TYPES.get(Path(file_name).suffix.lower())


def record_image_attachment(collection_name: str, file_path: str, file_name: str) -> bool:
    """Record a kept image upload for the next chat turn in ``collection_name``.

    Returns:
        True when recorded; False when the filename is not a supported image.
    """
    mime_type = image_mime_type(file_name)
    if mime_type is None:
        return False
    attachment = ImageAttachment(file_path=file_path, file_name=file_name, mime_type=mime_type)
    with _lock:
        pending = _pending.setdefault(collection_name, [])
        pending.append(attachment)
        dropped = len(pending) - MAX_PENDING_IMAGES_PER_COLLECTION
        if dropped > 0:
            del pending[:dropped]
    if dropped > 0:
        logger.warning(
            "Dropped %d oldest pending image(s) for collection_ref=%s",
            dropped,
            log_identifier_ref(collection_name),
        )
    return True


def claim_image_attachments(collection_name: str) -> list[ImageAttachment]:
    """Remove and return every pending image for ``collection_name``, oldest first."""
    with _lock:
        return _pending.pop(collection_name, [])


def clear_image_attachments() -> None:
    """Forget all pending images (used by tests)."""
    with _lock:
        _pending.clear()
