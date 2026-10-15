<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adding a Vision Agent

This guide explains how the vision agent was added, from the UI changes that let
users attach images to the agent that answers them. Use it to maintain the
feature, or as a pattern for another agent that is triggered by an uploaded file
rather than by the intent classifier.

When a user attaches an image in the web UI and sends a message, the turn goes
to the vision agent. The agent sends the user's text and the image, base64
encoded, to a vision-language model (VLM),
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`, instead of the default
Nemotron Lightning LLM.

This guide builds on [Adding a New Agent](./adding-a-new-agent.md), which
covers the basics of creating, registering, and routing an agent. It focuses on
what is different for a file-triggered agent.

---

## How It Works

The image and the chat message reach the backend as **two separate requests**.
The image is uploaded first, through the same documents endpoint as other
files. The chat message follows over the WebSocket and carries no image data.
The backend connects the two by session: the collection name used for the upload
is the chat's conversation ID.

```mermaid
sequenceDiagram
    participant UI as Web UI
    participant API as Documents API
    participant Ing as LlamaIndex ingestor
    participant Reg as Image registry
    participant Chat as Chat workflow
    participant VA as Vision agent
    participant VLM as Omni VLM

    UI->>API: POST /v1/collections/{session}/documents (image.png)
    API->>Ing: submit_job(temp paths, session)
    Ing->>Reg: record_image_attachment(session, path, name)
    API-->>UI: 202 Accepted (thumbnail marked "uploaded")
    Note over Ing: Image is kept on disk, not ingested
    UI->>Chat: Chat message (conversation-id = session)
    Chat->>Reg: claim_image_attachments(session)
    Chat->>VA: Skip intent classifier, read and base64-encode image
    VA->>VLM: Text + data:image/png;base64,...
    VLM-->>UI: Answer
```

### Design decisions

| Decision | Choice | Why |
|---|---|---|
| How the agent gets the image | **Server-side lookup.** Uploads are recorded by session, and the next chat turn claims them. | Builds on the existing upload flow with no change to the chat request format. The alternative, sending base64 inside the chat request, needs frontend payload changes and larger WebSocket messages. |
| How turns are routed | **Deterministic.** Any turn with a claimed image goes to the vision agent and skips the intent classifier. | Predictable, saves one LLM call, and keeps image references out of the routing prompt. |
| What goes into chat state | **File references only**, reset at every turn. Image bytes exist only inside the vision node and agent call. | Chat state is checkpointed; base64 images there would bloat the checkpoint database and persist user images. |

### Where the code lives

| Layer | Files |
|---|---|
| Upload settings | `deploy/.env` (`FILE_UPLOAD_ACCEPTED_TYPES`), `frontends/ui/src/shared/config/file-upload.ts` |
| Web UI | `frontends/ui/src/features/documents/` (`constants.ts`, `image-attachments-store.ts`, `image-thumbnail.ts`, `hooks/use-file-upload.ts`), `frontends/ui/src/features/layout/components/InputArea.tsx`, `frontends/ui/src/features/chat/` (`types.ts`, `store.ts`, `hooks/use-websocket-chat.ts`, `components/UserMessage.tsx`), `frontends/ui/src/features/layout/components/ChatArea.tsx` |
| Keeping images on the server | `frontends/deep_researcher_api/src/deep_researcher_api/routes/documents.py`, `sources/knowledge_layer/src/llamaindex/adapter.py`, `src/deep_researcher_agent/knowledge/image_attachments.py` |
| Vision agent | `src/deep_researcher_agent/agents/vision/` |
| Chat pipeline | `src/deep_researcher_agent/agents/chat_researcher/` (`models/intent.py`, `models/state.py`, `agent.py`, `register.py`) |
| Configuration | `configs/config_web_default_llamaindex.yml`, `pyproject.toml` |

---

## Part 1: Accept Image Uploads

### Step 1: Allow image file types in the UI and the API

The UI and the API share one extension allowlist, `FILE_UPLOAD_ACCEPTED_TYPES`.
Both must include the image types, or the file picker hides images and the API
rejects them with HTTP 415.

```bash
# deploy/.env
FILE_UPLOAD_ACCEPTED_TYPES=.pdf,.docx,.txt,.md,.png,.jpg,.jpeg
```

Set the same value wherever the UI and backend get their environment: in
`deploy/.env` for the CLI and Docker Compose, and under both apps' `env` sections
in `deploy/helm/deployment-k8s/values.yaml` for Helm. Refer to
[Knowledge Layer](../customization/knowledge-layer.md) for the upload settings.

Two lists map extensions to MIME types, and both already include PNG and JPEG:

- **UI:** `EXTENSION_TO_MIME` in `frontends/ui/src/shared/config/file-upload.ts`,
  used for drag-and-drop validation.
- **API:** `_DECLARED_CONTENT_TYPES` in
  `src/deep_researcher_agent/fastapi_extensions/upload_security.py`. The API also
  checks the file's magic bytes, so a renamed file (for example, a PDF named
  `.png`) is rejected.

To support another image format, add it to both lists and to the allowlist.

---

## Part 2: Show Images as Attachments in the UI

Ordinary uploads are added to the knowledge base and listed in the Files tab.
Images take a different path: they appear as thumbnails in the message composer,
move into the sent message, and are never listed as knowledge-base files.

### Step 2: Identify image files

`frontends/ui/src/features/documents/constants.ts` defines which uploads are
images and therefore not ingested:

```typescript
/** Extensions that are uploaded but not added to the knowledge base (images) */
export const IMAGE_FILE_TYPES = ['.png', '.jpg', '.jpeg']

/** True if the file is uploaded only and never appears in the knowledge base */
export const isImageFile = (fileName: string): boolean =>
  IMAGE_FILE_TYPES.some((ext) => fileName.toLowerCase().endsWith(ext))
```

Both are exported from `features/documents/index.ts`, together with the
attachments store from Step 3. Keep `IMAGE_FILE_TYPES` in sync with
`_IMAGE_EXTENSIONS` in the LlamaIndex adapter (Step 7) and `IMAGE_MIME_TYPES` in
the image registry (Step 8).

### Step 3: Track attachments in their own store

`frontends/ui/src/features/documents/image-attachments-store.ts` is a Zustand
store for images waiting in the composer. Each attachment records:

| Field | Purpose |
|---|---|
| `id`, `fileName` | Identity and label |
| `previewUrl` | `blob:` URL for the composer thumbnail, revoked when the attachment is removed |
| `thumbnailUrl` | Small JPEG data URL kept with the sent message, so it survives a page refresh |
| `status` | `uploading`, `uploaded`, or `failed` |
| `collectionName` | The session the image belongs to |

The store provides `addAttachment`, `updateAttachment`, `removeAttachment`, and
`clearAttachmentsForCollection`.

`frontends/ui/src/features/documents/image-thumbnail.ts` makes those
thumbnails. `createImageThumbnail` scales the image down to at most 512 px on its
longest side and returns a JPEG data URL. Chat history lives in `localStorage`,
which holds only a few MB, so full-size images can't be stored there. It paints
a white background first so transparent PNGs don't turn black.

### Step 4: Upload images separately from documents

In `frontends/ui/src/features/documents/hooks/use-file-upload.ts`, `uploadFiles`
splits the validated files and sends images to a new `uploadImages` function:

```typescript
const imageFiles = validationResult.validFiles.filter((f) => isImageFile(f.name))
const validFiles = validationResult.validFiles.filter((f) => !isImageFile(f.name))

if (imageFiles.length > 0) {
  void uploadImages(imageFiles, collectionName)
}
```

`uploadImages` adds each image to the attachments store as `uploading`, posts
the files to the same documents endpoint as other uploads, generates thumbnails
in parallel, and marks each attachment `uploaded` once both finish. It does not
poll for ingestion status, because images are never ingested: the upload request
completing means they are ready. On error the attachments are marked `failed`.

### Step 5: Show attachments in the composer

`frontends/ui/src/features/layout/components/InputArea.tsx`:

- Renders the current session's attachments as thumbnails above the text box,
  with a spinner while uploading, a warning icon on failure, and a remove button.
- **Disables Send while any image is still uploading.** The backend records the
  image before the upload request returns (Step 8), so a message sent after the
  upload finishes always finds its image.
- On send, converts the uploaded attachments into message images, clears them
  from the composer, and passes them to `sendMessage`:

  ```typescript
  const images = imageAttachments
    .filter((a) => a.status === 'uploaded' && a.thumbnailUrl)
    .map((a) => ({ id: a.id, fileName: a.fileName, url: a.thumbnailUrl as string }))
  setMessage('')
  if (currentConversation) clearImageAttachments(currentConversation.id)
  await sendMessage(currentMessage, { images })
  ```

- Opens the Files tab only when a non-image file is uploaded.

Sending still requires message text, as before.

### Step 6: Show images on sent messages

The sent message keeps its thumbnails for display:

| File | Change |
|---|---|
| `features/chat/types.ts` | New `MessageImage` type (`id`, `fileName`, `url`) and an optional `messageImages` field on `ChatMessage` |
| `features/chat/store.ts` | `addUserMessage` accepts and stores `messageImages` |
| `features/chat/hooks/use-websocket-chat.ts` | `sendMessage(content, options?)` accepts `{ images }` and passes them to `addUserMessage` |
| `features/layout/components/ChatArea.tsx` | Passes `message.messageImages` to `UserMessage` |
| `features/chat/components/UserMessage.tsx` | Renders the thumbnails above the message bubble |

```{important}
`messageImages` is display-only. The thumbnails are never sent to the backend,
and the chat request is unchanged. The backend finds the image through the
upload (Part 3).
```

The UI tests are in `InputArea.spec.tsx` and `UserMessage.spec.tsx`. Run them
from `frontends/ui/`:

```bash
npm run lint
npm run type-check
npm run test:ci
```

---

## Part 3: Keep Images on the Server

### Step 7: Keep uploaded files instead of deleting them

By default the upload's temporary file is deleted after ingestion. For images
there is nothing to ingest, so the file must survive until the chat turn reads
it.

Two changes keep it:

1. **Stop cleanup.** In
   `frontends/deep_researcher_api/src/deep_researcher_api/routes/documents.py`,
   the upload passes `"cleanup_files": False` to `submit_job`.
2. **Skip ingestion for images.** In `LlamaIndexIngestor._run_ingestion`
   (`sources/knowledge_layer/src/llamaindex/adapter.py`), image files are marked
   successful with zero chunks and dropped from file tracking, so they never
   appear in the collection:

   ```python
   # Standalone image uploads are kept on disk but not ingested
   _IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})
   ```

   ```python
   if Path(file_name).suffix.lower() in _IMAGE_EXTENSIONS:
       logger.info(f"Skipping ingestion for image {i + 1}/{len(file_paths)} (kept at {file_path})")
       self._update_file_status(job, i, FileStatus.SUCCESS, chunks_created=0)
       # Not part of the collection, so drop it from file tracking too
       with self._lock:
           if i < len(job.file_details):
               self._files.pop(job.file_details[i].file_id, None)
       continue
   ```

```{warning}
`"cleanup_files": False` applies to **every** upload through that route, not only
images, so ingested documents also stay in the temp directory. A narrower change
would keep only images, for example by deleting non-image files after ingestion
in the adapter. Kept images are also never deleted automatically; plan for
cleanup (such as a TTL sweep of `deep-researcher-upload-*` files) in long-running
deployments.
```

### Step 8: Record images by session

`src/deep_researcher_agent/knowledge/image_attachments.py` is a small
thread-safe registry of images waiting for a chat turn, keyed by collection name
(the conversation ID). It follows the same pattern as the document summary
store: backends write to it during upload, and the chat workflow reads from it.

| Function | Purpose |
|---|---|
| `record_image_attachment(collection, path, name)` | Store a kept image for the session. Returns `False` for non-image filenames. |
| `claim_image_attachments(collection)` | Remove and return all pending images for the session, oldest first. Each image is used once. |
| `image_mime_type(name)` | Map `.png`/`.jpg`/`.jpeg` to a MIME type. |
| `clear_image_attachments()` | Reset the registry (tests). |

Each entry is an `ImageAttachment` with `file_path`, `file_name`, and
`mime_type`. At most `MAX_PENDING_IMAGES_PER_COLLECTION` (8) images wait per
session; older ones are dropped with a warning. This bounds memory and the VLM
request size if images are uploaded but never sent.

The LlamaIndex adapter records images in `submit_job`, while it builds the job's
file list:

```python
# Kept images are answered by the vision agent on the next chat turn.
# Recorded here, before the upload request returns, so a quick send can't miss it.
if Path(file_name).suffix.lower() in _IMAGE_EXTENSIONS:
    record_image_attachment(collection_name, p, file_name)
```

Recording happens in `submit_job`, not in the background ingestion thread,
because `submit_job` runs before the upload request returns. The UI enables Send
only after that response (Step 5), so the image is always recorded before the
chat turn that uses it.

```{note}
The registry lives in process memory. The chat turn must run in the same
process that accepted the upload, which holds for a single-replica deployment.
Only the LlamaIndex backend keeps image files and records them; other knowledge
backends would need the same two changes (Steps 7 and 8).
```

---

## Part 4: Build the Vision Agent

The agent package follows Part 1 of [Adding a New Agent](./adding-a-new-agent.md).
Only the differences are described here.

### Step 9: Define state that carries images

`src/deep_researcher_agent/agents/vision/models/state.py` adds a model for one
base64 image and an `images` list on the agent state:

```python
class VisionImage(BaseModel):
    file_name: str
    mime_type: str
    data_base64: str = Field(repr=False)


class VisionAgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    images: list[VisionImage] = Field(default_factory=list)
```

`Field(repr=False)` matters. When a NeMo Agent Toolkit function call fails, NAT
logs the function's input. Without it, every failed vision call would write the
full base64 image into the server log.

`models/__init__.py` exports both classes.

### Step 10: Write the prompt

`prompts/vision.j2` tells the model to answer from what it can see: answer the
question directly or describe the image, transcribe visible text, read values
from charts and tables, refer to multiple images by filename, say when something
is unreadable, and not claim facts it can't see.

### Step 11: Send text and images to the model

`src/deep_researcher_agent/agents/vision/agent.py` contains two pieces.

`encode_image_file(file_path, file_name, mime_type)` reads an image from disk and
returns a `VisionImage` with base64 data.

`VisionAgent.build_messages(state)` builds the model input: the system prompt,
earlier text messages for context, and a final multimodal user message. The
image goes in as an OpenAI-style `image_url` content part with a base64
`data:` URL:

```python
content = [{"type": "text", "text": question or DEFAULT_QUESTION}]
for image in state.images:
    content.append({"type": "text", "text": f"Image: {image.file_name}"})
    content.append(
        {"type": "image_url", "image_url": {"url": f"data:{image.mime_type};base64,{image.data_base64}"}}
    )
return [SystemMessage(content=self.system_prompt), *context, HumanMessage(content=content)]
```

Details:

- An empty question becomes `"Describe this image."` (`DEFAULT_QUESTION`).
- A state with no images raises `ValueError`, which the orchestrator turns into a
  short error reply.
- `run` strips any inline `<think>...</think>` block. The NVIDIA endpoint normally
  returns reasoning separately in `additional_kwargs["reasoning_content"]`, so
  the reply is usually clean already.
- `run` returns the answer **without** the images, so base64 data never enters
  chat history.

### Step 12: Register the agent and its files

`register.py` registers `vision_agent` (config `VisionAgentConfig`: `llm` and
`max_history`). There is no standalone string workflow, because the agent needs
an image as well as text. `__init__.py` imports `register` and re-exports
`vision_agent`, as described in Step 6 of
[Adding a New Agent](./adding-a-new-agent.md).

Add the entry point and the prompt to `pyproject.toml`:

```toml
[tool.setuptools.package-data]
"deep_researcher_agent.agents.vision" = ["prompts/*.j2"]

[project.entry-points."nat.plugins"]
deep_researcher_vision = "deep_researcher_agent.agents.vision.register"
```

### Step 13: Configure the vision-language model

The `nim` LLM type (LangChain `ChatNVIDIA`) accepts `image_url` content parts
with base64 data URLs for this model, so no special client is needed:

```yaml
llms:
  # Vision-language model for the vision agent; must accept image inputs.
  # max_tokens covers reasoning tokens as well as the answer.
  nemotron_omni_vision_llm:
    _type: nim
    model_name: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
    base_url: "https://integrate.api.nvidia.com/v1"
    api_key: ${NVIDIA_API_KEY}
    temperature: 0.2
    max_tokens: 4096
    num_retries: 5
```

This is a reasoning model, and its reasoning tokens count toward `max_tokens`,
so keep the limit well above the expected answer length. To use a different VLM,
change `model_name`; the model must accept images.

---

## Part 5: Connect the Agent to the Chat Pipeline

All files in this part are under
`src/deep_researcher_agent/agents/chat_researcher/`. Unlike the joke and poem
routes in [Adding a New Agent](./adding-a-new-agent.md), this route bypasses
the intent classifier, so there is no prompt change and no availability flag.

### Step 14: Add the route target and per-turn image references

In `models/intent.py`, add `"vision"` to `RouteTarget`.

In `models/state.py`, add a field to `ChatResearcherState` and document it:

```python
attached_images: list[ImageAttachment] | None = None
```

It holds file references (`ImageAttachment` from Step 8), never image bytes.

### Step 15: Claim images when a turn starts

In `register.py`:

1. Add `vision_agent: FunctionRef | None` to `ChatDeepResearcherConfig`.
2. Resolve it next to the other optional agents:
   `vision_fn = await builder.get_function(config.vision_agent) if config.vision_agent else None`.
3. Pass `vision_fn=vision_fn.ainvoke if vision_fn else None` to `ChatResearcherAgent`.
4. In `_run_impl`, claim the session's images and put them on the turn's state:

   ```python
   attached_images = None
   if vision_fn is not None and client_conversation_id:
       from deep_researcher_agent.knowledge.image_attachments import claim_image_attachments

       attached_images = claim_image_attachments(client_conversation_id) or None
   ```

   ```python
   state = ChatResearcherState(
       # ...existing fields...
       attached_images=attached_images,
   )
   ```

Images are claimed only when a vision agent is configured, so deployments
without one leave them untouched.

### Step 16: Route image turns to the vision node

This step reads `state.attached_images`; it doesn't set it. The value travels
through these lines (all under `src/deep_researcher_agent/agents/chat_researcher/`;
line numbers are as of this writing and shift as the files change):

1. `register.py:843`: `_run_impl` calls `claim_image_attachments`, which only
   removes the session's images from the registry and returns them as a list. It
   does not touch the state. The list goes into a local variable:
   `attached_images = claim_image_attachments(client_conversation_id) or None`.
2. `register.py:865`: `_run_impl` passes that variable to the turn's state:
   `ChatResearcherState(..., attached_images=attached_images)`. This is where
   `state.attached_images` is set (Step 15).
3. `agent.py:790` (and `agent.py:773` for `dict` input): `run` copies the field
   into the graph's input state (item 6 below).
4. `agent.py:202`: `intent_classifier_node` reads it and routes to the vision
   node (item 2 below).

So when `intent_classifier_node` runs, the field holds either this turn's
images or `None`, never images from an earlier turn.

In `agent.py` (`ChatResearcherAgent`):

1. **Constructor:** add
   `vision_fn: Callable[[VisionAgentState], Awaitable[VisionAgentState]] | None = None`,
   document it, and store it as `self.vision_fn`.
2. **Bypass the classifier.** At the top of `intent_classifier_node`:

   ```python
   if state.attached_images and self.vision_fn is not None:
       return {"user_intent": IntentResult(intent="research", target="vision")}
   ```

3. **Add `vision_node`.** It reads and encodes the images off the event loop,
   then calls the agent:

   ```python
   images = await asyncio.gather(
       *(asyncio.to_thread(encode_image_file, a.file_path, a.file_name, a.mime_type) for a in attachments)
   )
   trimmed_messages = trim_message_history(state.messages, self.max_history)
   result = await self.vision_fn(VisionAgentState(messages=trimmed_messages, images=list(images)))
   return {"messages": [AIMessage(content=result.messages[-1].content)]}
   ```

   Failures are bounded. An `OSError` (for example, a deleted file) returns "I
   couldn't read the uploaded image. Please upload it again." Any other error
   returns "I couldn't analyze the image right now. Please try again." Both mark
   the workflow as failed, and neither puts file paths or exception text in the
   reply or the logs.
4. **Route.** In `route_after_orchestration`, return `"vision"` when
   `state.user_intent.target == "vision"`, before the `intent == "meta"` check.
5. **Wire the graph.** Add the node, add `"vision": "vision"` to the
   `intent_classifier` edge map, and add `graph.add_edge("vision", END)`.
6. **Reset images every turn.** In `run`, include `attached_images` in the input
   state for both input shapes (`dict` and `ChatResearcherState`). Without this,
   LangGraph's checkpoint would keep the previous turn's images, and the next
   text-only message would go to the vision agent again.

### Step 17: Turn it on in the web config

In `configs/config_web_default_llamaindex.yml`, add the LLM from Step 13, the
agent function, and the workflow setting:

```yaml
functions:
  vision_agent:
    _type: vision_agent
    llm: nemotron_omni_vision_llm

workflow:
  _type: chat_deepresearcher_agent
  # ...existing settings...
  vision_agent: vision_agent
```

Only the web config needs it, because only the web UI uploads images. Document
the setting in the `workflow` table of the
[Configuration Reference](../customization/configuration-reference.md).

---

## Part 6: Test

### Step 18: Unit tests

| Test file | What it covers |
|---|---|
| `tests/deep_researcher_agent/knowledge/test_image_attachments.py` | MIME mapping, claim-once behavior, session isolation, non-image rejection, the pending limit |
| `tests/knowledge_layer_tests/test_llamaindex_image_attachments.py` | `submit_job` records images (not documents) under the session with their original filenames |
| `tests/deep_researcher_agent/agents/vision/test_agent.py` | Base64 encoding, message format, history, default question, missing-image error, `<think>` stripping, image data excluded from `repr` |
| `tests/deep_researcher_agent/agents/chat_researcher/test_agent.py` | Image turns skip the classifier; images don't carry over to the next turn; no vision agent means normal routing; unreadable files and agent errors are bounded |

```bash
uv run pytest tests/deep_researcher_agent/knowledge tests/deep_researcher_agent/agents/vision \
  tests/deep_researcher_agent/agents/chat_researcher tests/knowledge_layer_tests/test_llamaindex_image_attachments.py -q
uv run ruff check src/deep_researcher_agent sources/knowledge_layer/src tests
uv run ruff format --check src/deep_researcher_agent sources/knowledge_layer/src tests
```

### Step 19: End-to-end test without the UI

You can drive the same flow the UI uses with `curl`. Start the web backend with
throwaway databases so local data stays untouched:

```bash
tmp=$(mktemp -d)
DEEP_RESEARCHER_CHECKPOINT_DB=$tmp/checkpoints.db \
NAT_JOB_STORE_DB_URL=sqlite+aiosqlite:///$tmp/jobs.db \
DEEP_RESEARCHER_SUMMARY_DB=sqlite+aiosqlite:///$tmp/summaries.db \
DEEP_RESEARCHER_CHROMA_DIR=$tmp/chroma \
dotenv -f deploy/.env run nat serve --config_file configs/config_web_default_llamaindex.yml --port 8765
```

In another terminal, create a session, upload an image, and send a message with
the same session as the `conversation-id` header:

```bash
session=vision-test-1
curl -s -X POST http://127.0.0.1:8765/v1/collections \
  -H 'Content-Type: application/json' -d "{\"name\":\"$session\"}"
curl -s -X POST "http://127.0.0.1:8765/v1/collections/$session/documents" \
  -F "files=@/path/to/image.png;type=image/png"
curl -s -X POST http://127.0.0.1:8765/chat \
  -H 'Content-Type: application/json' -H "conversation-id: $session" \
  -d '{"messages":[{"role":"user","content":"What is in this image?"}]}'
```

The reply describes the image, and the server log shows
`Claimed 1 uploaded image(s) for collection_ref=...`. A second message in the
same session without an upload should get a normal reply through the intent
classifier.

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| The file picker doesn't offer images, or the upload fails with HTTP 415 | `FILE_UPLOAD_ACCEPTED_TYPES` lacks `.png,.jpg,.jpeg` for the UI, the backend, or both (Step 1). Restart both after changing it. |
| The image is uploaded but the reply ignores it | No image was claimed. Check that `vision_agent` is set under `workflow`, that the knowledge backend is LlamaIndex, that the log shows `Claimed ... uploaded image(s)`, and that the upload and chat used the same session ID. |
| Images go missing with more than one backend replica | The registry is per process. Route a session's requests to one replica, or move the registry and files to shared storage. |
| `I couldn't read the uploaded image. Please upload it again.` | The kept file was deleted before the turn read it, for example by a temp-directory cleanup or a restart that cleared `/tmp`. |
| `I couldn't analyze the image right now. Please try again.` | The VLM call failed. The log shows `Vision agent failed (error_type=...)`, and NAT logs the upstream error just before it. `503 ResourceExhausted` means the hosted model is at capacity; retry. |
| The answer is cut off | Reasoning tokens used up `max_tokens`. Raise it in the LLM config (Step 13). |

## Known Limitations

- **Removing a thumbnail doesn't cancel the image.** The composer's remove button
  only updates the UI store, so the backend still has the image and the next
  message goes to the vision agent with it. Fixing this needs a backend endpoint
  to discard a pending image and a UI call to it.
- **No follow-up questions about an image.** An image is used for one turn. A
  later text message ("what does the label say?") goes through normal routing
  without the image.
- **Single replica, LlamaIndex backend only.** See the note in Step 8.
- **VLM calls aren't in the relay token traces.** The agent calls the model
  directly rather than through `ainvoke_with_relay`. Check that relay tracing
  won't record the base64 image before changing this.

---

## Related

- [Adding a New Agent](./adding-a-new-agent.md) — The general agent and route pattern this builds on
- [Intent Classifier](../architecture/agents/intent-classifier.md) — Routing that image turns bypass
- [Knowledge Layer](../customization/knowledge-layer.md) — Upload settings and backends
- [Configuration Reference](../customization/configuration-reference.md) — The `vision_agent` workflow setting
