# Agentflow Multimodal Support — Master Plan

## Research Summary

### How ADK Handles Multimodal

Google ADK uses `google.genai.types.Part` as the universal content unit:
- **Inline data**: `types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")` — for files < 20MB
- **File API upload**: `client.files.upload(file="path.jpg")` → returns a `types.File` reference — for large/reusable files
- **PIL Image**: Pillow Image objects are auto-converted by the SDK
- **Artifacts**: Binary data stored via `ArtifactService` (in-memory or GCS), versioned, identified by filename + namespace (session or user scope), represented as `types.Part(inline_data=types.Blob(data=bytes, mime_type="..."))`
- **Supported formats**: PNG, JPEG, WEBP, HEIC, HEIF for images; PDF via document processing; audio via PCM blobs

### How LangChain Handles Multimodal

LangChain v1 uses standard content blocks in `HumanMessage.content`:
```python
# Image via URL
{"type": "image", "url": "https://example.com/image.jpg"}
# Image via base64
{"type": "image", "base64": "...", "mime_type": "image/jpeg"}
# Image via file_id
{"type": "image", "file_id": "file-abc123"}
# PDF document
{"type": "file", "base64": "...", "mime_type": "application/pdf"}
# Audio
{"type": "audio", "base64": "...", "mime_type": "audio/wav"}
```
Provider-native formats also supported (OpenAI's `image_url` type, etc.)

### How OpenAI API Expects Multimodal

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{base64_data}"}},
        # OR
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
    ]
}]
# PDFs: use file_search or pass base64 as image
# Audio input: {"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}}
```

### How Google GenAI API Expects Multimodal

```python
from google.genai import types
contents = [
    types.Content(role="user", parts=[
        types.Part(text="What is this?"),
        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
        # OR via File API
        types.Part(file_data=types.FileData(file_uri="...", mime_type="...")),
    ])
]
```

---

## Current State Analysis

### What Exists ✅
| Component | Status | Notes |
|-----------|--------|-------|
| `MessageBlock` types | ✅ Defined | `ImageBlock`, `AudioBlock`, `VideoBlock`, `DocumentBlock`, `DataBlock` all exist in `message_block.py` |
| `MediaRef` model | ✅ Defined | Supports `url`, `file_id`, `data_base64` with mime_type, size, dimensions |
| `Message.attach_media()` | ✅ Exists | Can append media blocks to message content |
| `ContentBlock` union | ✅ Defined | Discriminated union of all block types |
| Response converters | ✅ Partial | `OpenAIConverter` and `GoogleGenAIConverter` can extract images from responses |
| `TokenUsages.image_tokens` | ✅ Defined | Field exists for multimodal token tracking |

### What's Missing ❌
| Component | Gap | Impact |
|-----------|-----|--------|
| **`_convert_dict()` in converter.py** | Only extracts `.text()` — all media blocks are **silently dropped** | Images/audio/docs in messages are never sent to LLM |
| **`_handle_regular_message()` in google.py** | Only wraps text in `types.Part(text=...)` — no media | Google provider ignores all multimodal content |
| **OpenAI message format** | Messages only contain string `content` — no `content: [...]` array | OpenAI provider ignores all multimodal content |
| **File upload API endpoint** | No endpoint to upload files (images, PDFs, docs) | No way for clients to send files |
| **Document extraction** | No PDF/DOCX text extraction utilities | Can't read documents to pass as text |
| **Media storage backend** | No binary file storage service — blobs would be inlined in DB | Uploaded files have nowhere to persist |
| **Multimodal config** | No per-agent config for how to handle images (base64 vs PIL vs file_id) | No flexibility |
| **Document processing config** | No config for whether to extract text from PDFs or pass raw to AI | No flexibility |
| **Input message creation helpers** | No convenience API to create multimodal messages easily | Poor DX |
| **MediaStore → MediaRef pipeline** | No mechanism to store blobs externally and reference them in messages | Blobs would be inlined into state/checkpointer, bloating DB |

### Critical Problem: State & Checkpointer with Binary Data ⚠️

Current serialization paths that would be affected by naive inline base64:

```
Path 1: AgentState → PG states table (JSONB)
  state.model_dump() → json.dumps() → INSERT INTO states(state_data JSONB)
  ⚠️ AgentState.context = list[Message] → all messages serialized into ONE JSONB blob
  ⚠️ Each state save RE-SERIALIZES ALL messages including ALL images from ALL turns
  ⚠️ A 10-turn conversation with 3 images ≈ 30MB+ in a single JSONB cell, growing every turn

Path 2: Messages → PG messages table (TEXT)  
  [block.model_dump(mode="json") for block in message.content] → json.dumps() → INSERT INTO messages(content TEXT)
  ⚠️ Each image block with data_base64 ≈ 1-5MB stored as TEXT per message row

Path 3: AgentState → Redis cache
  state.model_dump() → json.dumps() → Redis SETEX
  ⚠️ Same massive state with all images goes into Redis, evicted by TTL = 24h

Path 4: InMemoryCheckpointer → Python dict
  self._states[key] = state  (holds full Python objects in memory)
  ⚠️ All images kept as Python objects in process memory forever
```

**Result**: One 1MB image creates 3+ copies across PG JSONB, PG TEXT, and Redis.
A realistic 20-message conversation with 5 images = 60-100MB database footprint, re-serialized on every state save.

---

## Architecture Design

### Design Principles
1. **Never store blobs in the database** — Binary data goes to `MediaStore`; only tiny `MediaRef` references live in messages/state
2. **Provider-agnostic content model** — `Message` with `ContentBlock` types is the universal format
3. **Configurable processing** — Developer controls image handling (base64/url/file_id) and document handling (extract text vs pass raw)
4. **Maximum flexibility** — Support all input methods: URL, base64, file path, PIL Image, bytes, file_id
5. **Lazy conversion** — Content stays as `ContentBlock` until the last mile (provider call), then converts to provider-specific format
6. **Checkpointer stays unchanged** — The fix is what goes INTO messages, not how they're stored

### The Reference Pattern: How State & Checkpointer Work

The `MediaRef` model already has the right design. The fix is a `BaseMediaStore` layer that stores blobs externally and converts them to lightweight references BEFORE they enter the message.

```
┌─────────────────────────────────────────────────────────────────────┐
│  INGEST BOUNDARY (where blobs enter the system)                     │
│                                                                     │
│  Option A: API upload                                               │
│    POST /v1/files/upload (multipart)                                │
│    → MediaStore.store(bytes, mime_type) → storage_key "abc123"      │
│    → MediaRef(kind="url", url="agentflow://media/abc123")           │
│                                                                     │
│  Option B: SDK usage                                                │
│    msg = Message.with_image(bytes, mime_type, media_store=store)    │
│    → MediaStore.store(bytes, mime_type) → storage_key "abc123"      │
│    → MediaRef(kind="url", url="agentflow://media/abc123")           │
│                                                                     │
│  Option C: External URL (no storage needed)                         │
│    MediaRef(kind="url", url="https://cdn.example.com/img.jpg")      │
│                                                                     │
│  Option D: Small inline (< threshold, e.g. 50KB)                   │
│    MediaRef(kind="data", data_base64="...", mime_type="image/png")  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
         Message.content = [TextBlock("describe this"), ImageBlock(media=ref)]
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
   ┌──────────────┐      ┌───────────────┐      ┌──────────────────┐
   │ PG states    │      │ PG messages   │      │ Redis cache      │
   │ table JSONB  │      │ table TEXT    │      │                  │
   │              │      │               │      │                  │
   │ MediaRef is  │      │ MediaRef is   │      │ MediaRef is      │
   │ ~100 bytes:  │      │ ~100 bytes    │      │ ~100 bytes       │
   │ {kind:"url", │      │               │      │                  │
   │  url:"ag://  │      │ NOT 1-5MB     │      │ NOT 1-5MB        │
   │  media/abc"} │      │ base64!       │      │ base64!          │
   └──────────────┘      └───────────────┘      └──────────────────┘
                                    │
                                    ▼
           ┌────────────────────────────────────────────────┐
           │  LLM BOUNDARY (converter resolves references)  │
           │                                                │
           │  MediaRef(kind="url", url="agentflow://...")   │
           │  → MediaStore.retrieve(key) → raw bytes        │
           │  → OpenAI: base64 data URL                     │
           │  → Google: types.Part.from_bytes(bytes)        │
           │                                                │
           │  MediaRef(kind="url", url="https://...")       │
           │  → OpenAI: pass URL directly                   │
           │  → Google: types.Part.from_uri(uri)            │
           │                                                │
           │  MediaRef(kind="data", data_base64="...")      │
           │  → OpenAI: data:mime;base64,{data}             │
           │  → Google: types.Part.from_bytes(decoded)      │
           └────────────────────────────────────────────────┘
```

### BaseMediaStore Interface

```python
class BaseMediaStore(ABC):
    """Abstract interface for storing binary media outside the message system."""

    @abstractmethod
    async def store(self, data: bytes, mime_type: str, metadata: dict | None = None) -> str:
        """Store binary data, return a storage_key (opaque string)."""

    @abstractmethod
    async def retrieve(self, storage_key: str) -> tuple[bytes, str]:
        """Retrieve binary data and mime_type by storage_key."""

    @abstractmethod
    async def delete(self, storage_key: str) -> bool:
        """Delete stored media. Returns True if deleted."""

    @abstractmethod
    async def exists(self, storage_key: str) -> bool:
        """Check if media exists in store."""

    def to_media_ref(self, storage_key: str, mime_type: str, **kwargs) -> MediaRef:
        """Convert a storage key to a MediaRef for embedding in messages."""
        return MediaRef(
            kind="url",
            url=f"agentflow://media/{storage_key}",
            mime_type=mime_type,
            **kwargs,
        )
```

### MediaStore Implementations

| Implementation | Backend | Use Case |
|----------------|---------|----------|
| `InMemoryMediaStore` | Python `dict[str, tuple[bytes, str]]` | Tests, ephemeral scripts |
| `LocalFileMediaStore` | Filesystem (configurable base path) | Dev, single-server deployments |
| `S3MediaStore` / `CloudMediaStore` | S3/MinIO/GCS compatible via `cloud-storage-manager` | Production, multi-instance |
| `PgBlobStore` | PostgreSQL `bytea` in separate `media` table | PG-only deployments (avoids S3) |

Even `PgBlobStore` stores blobs in a **separate `media` table** with `bytea` column — never inside the `states` or `messages` JSONB. This keeps the core tables lean and the media data separately manageable (can be vacuumed, archived, or migrated to S3 later).

### MediaRef Resolution Strategy

The converter layer resolves `MediaRef` → provider format at LLM call time:

```python
class MediaRefResolver:
    """Resolves MediaRef objects to actual binary data or URLs for provider APIs."""

    def __init__(self, media_store: BaseMediaStore | None = None):
        self.media_store = media_store

    async def resolve_for_openai(self, ref: MediaRef) -> dict:
        """Convert MediaRef to OpenAI content part."""
        if ref.kind == "url" and ref.url and ref.url.startswith("agentflow://media/"):
            key = ref.url.removeprefix("agentflow://media/")
            data, mime = await self.media_store.retrieve(key)
            b64 = base64.b64encode(data).decode()
            return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        elif ref.kind == "url" and ref.url:
            return {"type": "image_url", "image_url": {"url": ref.url}}
        elif ref.kind == "data" and ref.data_base64:
            return {"type": "image_url", "image_url": {"url": f"data:{ref.mime_type};base64,{ref.data_base64}"}}
        elif ref.kind == "file_id":
            # Provider-native file reference
            return {"type": "image_url", "image_url": {"url": ref.url or ref.file_id}}

    async def resolve_for_google(self, ref: MediaRef) -> Any:
        """Convert MediaRef to Google types.Part."""
        from google.genai import types
        if ref.kind == "url" and ref.url and ref.url.startswith("agentflow://media/"):
            key = ref.url.removeprefix("agentflow://media/")
            data, mime = await self.media_store.retrieve(key)
            return types.Part.from_bytes(data=data, mime_type=mime)
        elif ref.kind == "url" and ref.url:
            return types.Part.from_uri(file_uri=ref.url, mime_type=ref.mime_type)
        elif ref.kind == "data" and ref.data_base64:
            data = base64.b64decode(ref.data_base64)
            return types.Part.from_bytes(data=data, mime_type=ref.mime_type)
        elif ref.kind == "file_id":
            return types.Part(file_data=types.FileData(file_uri=ref.file_id, mime_type=ref.mime_type))
```

### Inline Data Guard (Optional Safety Net)

For safety, add an optional pre-save hook that prevents large inline base64 from accidentally entering the checkpointer:

```python
class MediaOffloadPolicy(str, Enum):
    NEVER = "never"           # Allow inline base64 (testing/small images)
    THRESHOLD = "threshold"   # Offload if > max_inline_bytes (default)
    ALWAYS = "always"         # Always offload to MediaStore

class MultimodalConfig(BaseModel):
    # ...existing fields...
    offload_policy: MediaOffloadPolicy = MediaOffloadPolicy.THRESHOLD
    max_inline_bytes: int = 50_000  # ~50KB — below this, inline is fine
```

If a message enters the system with a large `data_base64` and a `MediaStore` is configured, the system can:
1. Log a warning ("Large inline media detected, consider using MediaStore")
2. Optionally auto-offload: store to `MediaStore`, replace `MediaRef(kind="data")` with `MediaRef(kind="url")`

This is NOT done in the checkpointer itself — it happens at the message ingestion boundary.

### What Changes in Checkpointer: **Nothing**

| Checkpointer Component | Change Needed? | Why |
|------------------------|---------------|-----|
| `BaseCheckpointer` | ❌ No change | Interface stays the same |
| `InMemoryCheckpointer` | ❌ No change | Python objects, just holds references |
| `PgCheckpointer.aput_state()` | ❌ No change | `state.model_dump()` → JSONB still works; MediaRef is tiny |
| `PgCheckpointer.aput_messages()` | ❌ No change | `block.model_dump()` works; ImageBlock/MediaRef serialize fine |
| `PgCheckpointer._row_to_message()` | ❌ No change | Pydantic `model_validate` already deserializes ImageBlock/MediaRef |
| Redis cache | ❌ No change | Same tiny JSON references |
| DB schema | ❌ No change* | *Optional: add `media` table for metadata, not required |

The key insight: **fix the input, not the storage.** If only references enter messages, the existing serialization pipeline handles everything perfectly.

### Content Flow

```
Client Upload                    PyAgenity Core                    Provider API
─────────────                    ──────────────                    ────────────
                                                                  
image/pdf/docx  ──► API endpoint ──► MediaProcessor ──► Message   
                    (FastAPI)         - validate           │      
                    - file upload     - store binary       │      
                    - base64          - create MediaRef    │      
                    - URL                                  │      
                                                           ▼      
                                                    ContentBlock   
                                                    (ImageBlock,   
                                                     DocumentBlock,
                                                     etc.)         
                                                           │      
                                    ┌──────────────────────┤      
                                    ▼                      ▼      
                              convert_dict()         Google format  
                              (OpenAI format)        (types.Part)   
                                    │                      │      
                                    ▼                      ▼      
                              OpenAI API              Gemini API   
```

### Design Philosophy: Library vs API

| Layer | Extraction? | Reason |
|-------|-------------|--------|
| **PyAgenity (core library)** | ❌ No extraction | Keeps the library lightweight; developers choose their own tools for document handling when using the SDK directly |
| **pyagenity-api (platform)** | ✅ Auto-extracts | When developers use the hosted API, extraction is handled transparently using `textxtract`. They just upload files and get back AI responses. |

> **Rule**: If you're using PyAgenity as a library, you control how documents are converted to text and pass the result as a `TextBlock`. If you're using the API platform, upload the file and the API handles extraction automatically via `textxtract`.

### New Components

```
agentflow/                          # PyAgenity core — NO extraction logic
├── media/                          # NEW: Media processing & storage module
│   ├── __init__.py
│   ├── config.py                   # MultimodalConfig, ImageHandling, DocumentHandling, MediaOffloadPolicy
│   ├── processor.py                # MediaProcessor: validate mime type, size, resize images
│   ├── resolver.py                 # MediaRefResolver: resolve MediaRef → provider format at LLM call time
│   └── storage/                    # Binary blob storage backends (NOT in checkpointer DB)
│       ├── __init__.py
│       ├── base.py                 # BaseMediaStore: store/retrieve/delete/exists/to_media_ref
│       ├── memory_store.py         # InMemoryMediaStore — dict-based (dev/test)
│       ├── local_store.py          # LocalFileMediaStore — filesystem (dev/single-server)
│       ├── cloud_store.py          # CloudMediaStore — S3/GCS via cloud-storage-manager (production)
│       └── pg_store.py             # PgBlobStore — separate PG BYTEA table (PG-only deployments)

agentflow_cli/                      # pyagenity-api — document extraction lives HERE
├── media/                          # NEW: API-side media handling
│   ├── __init__.py
│   ├── extractor.py                # DocumentExtractor: wraps textxtract AsyncTextExtractor
│   └── pipeline.py                 # DocumentPipeline: upload → extract → inject into message
```

---

## Sprint Plan

### Sprint 1: Core Multimodal Pipeline (PyAgenity) — Foundation
**Goal**: Make images work end-to-end through the agent pipeline

- [x] **1.1** Create `agentflow/media/__init__.py` and `agentflow/media/config.py`
  - `MultimodalConfig` pydantic model:
    ```python
    class ImageHandling(str, Enum):
        BASE64 = "base64"           # Inline base64 in API call
        URL = "url"                 # Pass URL reference
        FILE_ID = "file_id"         # Use provider file upload API
    
    class DocumentHandling(str, Enum):
        EXTRACT_TEXT = "extract_text"   # Read PDF/DOCX → pass as text
        PASS_RAW = "pass_raw"           # Pass binary to AI (if provider supports)
        SKIP = "skip"                   # Don't send documents to AI
    
    class MultimodalConfig(BaseModel):
        image_handling: ImageHandling = ImageHandling.BASE64
        document_handling: DocumentHandling = DocumentHandling.EXTRACT_TEXT
        max_image_size_mb: float = 10.0
        max_image_dimension: int = 2048      # Resize if larger
        supported_image_types: set[str] = {"image/jpeg", "image/png", "image/webp", "image/gif"}
        supported_doc_types: set[str] = {"application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    ```

- [x] **1.2** Update `_convert_dict()` in `agentflow/utils/converter.py`
  - Convert `ImageBlock` → OpenAI's `{"type": "image_url", "image_url": {"url": "..."}}` format
  - Convert `AudioBlock` → `{"type": "input_audio", "input_audio": {"data": "...", "format": "..."}}`
  - Convert `DocumentBlock` → text block (if extract_text mode) or image_url (if pass_raw mode for PDF)
  - Return `content` as a **list of content parts** (not a string) when multimodal blocks are present
  - Keep backward compat: text-only messages still return `{"content": "string"}`

- [x] **1.3** Update `_handle_regular_message()` in `agentflow/graph/agent_internal/google.py`
  - When message `content` is a list of parts (from convert_dict), convert each to `types.Part`:
    - Text → `types.Part(text=...)`
    - Image URL → `types.Part.from_uri(file_uri=url, mime_type=...)`
    - Image base64 → `types.Part.from_bytes(data=decoded_bytes, mime_type=...)`
    - File ID → `types.Part(file_data=types.FileData(file_uri=..., mime_type=...))`
  - Handle the case where content comes as a list of dicts (multimodal)

- [x] **1.4** Update OpenAI message handling
  - In `_call_openai()` / `_call_openai_responses()`, messages with list content already work with OpenAI SDK
  - Ensure the content array format `[{"type": "text", "text": "..."}, {"type": "image_url", ...}]` is passed through correctly

- [x] **1.5** Add `multimodal_config` parameter to `Agent.__init__()`
  - Optional `MultimodalConfig` parameter on Agent
  - Pass config through to converter functions

- [x] **1.6** Add convenience constructors to `Message`
  - `Message.image_message(image_url=..., text=..., role="user")`
  - `Message.multimodal_message(content_blocks=[...], role="user")`
  - `Message.from_file(file_path, mime_type=None, text=None)` — auto-detect type, create appropriate blocks

- [x] **1.7** Write tests
  - Test `_convert_dict` with ImageBlock (base64, URL, file_id)
  - Test Google format conversion with images
  - Test OpenAI format conversion with images 
  - Test end-to-end: Message with image → provider call format
  - Test backward compatibility: text-only still works

### Sprint 2: Document Processing & Extraction (pyagenity-api)
**Goal**: Automatic PDF, DOCX, and other document extraction in the API platform using `textxtract`. The PyAgenity core library does **not** include any extraction logic — if developers use the SDK directly, they extract text themselves and pass it as a `TextBlock`.

**Library used**: [`textxtract`](https://10xhub.github.io/textxtract/) — supports async, works from file path or raw bytes, handles PDF, DOCX, DOC, RTF, HTML, CSV, JSON, XML, MD, TXT, ZIP.

```python
from textxtract import AsyncTextExtractor
from textxtract.core.exceptions import FileTypeNotSupportedError, ExtractionError

extractor = AsyncTextExtractor()
text = await extractor.extract(file_bytes, "document.pdf")
```

- [x] **2.1** Add `textxtract` to `pyagenity-api` dependencies
  - In `pyagenity-api/pyproject.toml`, add:
    - `textxtract[pdf]` → PyMuPDF for PDF support
    - `textxtract[docx]` → python-docx for Word support
    - `textxtract[html]` → beautifulsoup4 for HTML support
    - `textxtract[xml]` → lxml for XML support
    - `textxtract[md]` → markdown for Markdown support
  - Or use `textxtract[pdf,docx,html,xml,md]` combined extras
  - Text, CSV, JSON, ZIP are supported built-in (no extras needed)

- [x] **2.2** Create `DocumentExtractor` service in `agentflow_cli/media/extractor.py`
  - Wraps `AsyncTextExtractor` from `textxtract`
  - Single method: `async def extract(data: bytes, filename: str) -> str`
  - Maps MIME type → filename extension when only bytes + mime_type is known
  - Handles `FileTypeNotSupportedError` → returns `None` (unsupported → pass raw)
  - Handles `ExtractionError` → raises `400 Bad Request` with clear message
  - Example:
    ```python
    from textxtract import AsyncTextExtractor
    from textxtract.core.exceptions import FileTypeNotSupportedError, ExtractionError

    class DocumentExtractor:
        def __init__(self):
            self._extractor = AsyncTextExtractor()

        async def extract(self, data: bytes, filename: str) -> str | None:
            try:
                return await self._extractor.extract(data, filename)
            except FileTypeNotSupportedError:
                return None  # caller decides how to handle unsupported types
            except ExtractionError as e:
                raise ValueError(f"Failed to extract text from {filename}: {e}") from e
    ```

- [x] **2.3** Create `DocumentPipeline` in `agentflow_cli/media/pipeline.py`
  - Orchestrates: receive uploaded file → extract text via `DocumentExtractor` → return `TextBlock` or `DocumentBlock`
  - When extraction succeeds → returns `TextBlock(text=extracted_text)` with original filename as metadata
  - When file type not supported for extraction (e.g. images) → return `DocumentBlock` (raw, to be handled by provider)
  - Respects `DocumentHandling` config:
    - `EXTRACT_TEXT` → always attempt extraction, raise if fails
    - `PASS_RAW` → skip extraction, return `DocumentBlock` with base64/media_ref
    - `SKIP` → return `None` (drop document from message)

- [x] **2.4** Wire `DocumentPipeline` into the file upload endpoint (Sprint 4)
  - When `POST /v1/files/upload` receives a document (non-image, non-audio):
    - Store binary in `MediaStore`
    - Also run `DocumentExtractor.extract()` and cache the extracted text
    - Return both `file_id` and optionally `extracted_text` in the response
  - When a `DocumentBlock` arrives in a message at invoke/stream time:
    - If `file_id` references an already-extracted file → inject `TextBlock` with cached text
    - If inline bytes/base64 → run extraction on-the-fly

- [x] **2.5** Update `MediaProcessor` in PyAgenity (`agentflow/media/processor.py`)
  - `MediaProcessor` handles only **images**: validate mime type, check file size, optionally resize
  - **No document extraction logic** — documents are processed by `DocumentPipeline` in the API layer
  - Clearly document this in docstring:
    > `MediaProcessor` handles image validation and resizing only. Document text extraction is the responsibility of the caller (API layer uses `DocumentPipeline`; SDK users extract text themselves).

- [x] **2.6** Add optional image dependency to PyAgenity `pyproject.toml`
  - `pip install 10xscale-agentflow[images]` → `Pillow` (for image resizing/processing)
  - Remove any pdf/docx extras from PyAgenity — those belong in pyagenity-api
  - `pip install 10xscale-agentflow[all]` → `Pillow` only (no extraction deps)

- [x] **2.7** Write tests
  - `DocumentExtractor`: mock `AsyncTextExtractor`, test success, unsupported type, extraction error
  - `DocumentPipeline`: test all three `DocumentHandling` modes
  - Integration: upload PDF via API → text extracted → injected into agent message
  - Verify PyAgenity core has zero `textxtract` or extraction imports

### Sprint 3: Media Storage Layer (PyAgenity) — The Database Problem Solution
**Goal**: Binary data NEVER touches the checkpointer/state DB. Only lightweight `MediaRef` references are stored.

- [x] **3.1** Create `BaseMediaStore` abstract interface
  - `agentflow/media/storage/base.py`:
    ```python
    class BaseMediaStore(ABC):
        async def store(self, data: bytes, mime_type: str, metadata: dict | None = None) -> str  # returns storage_key
        async def retrieve(self, storage_key: str) -> tuple[bytes, str]  # returns (data, mime_type)
        async def delete(self, storage_key: str) -> bool
        async def exists(self, storage_key: str) -> bool
        def to_media_ref(self, storage_key: str, mime_type: str, **kwargs) -> MediaRef
    ```
  - `to_media_ref()` creates `MediaRef(kind="url", url="agentflow://media/{key}", mime_type=...)`
  - Storage keys are opaque strings (UUID-based), no user input in keys

- [x] **3.2** Implement `InMemoryMediaStore`
  - `agentflow/media/storage/memory_store.py`
  - `dict[str, tuple[bytes, str, dict]]` — key → (data, mime_type, metadata)
  - For testing and ephemeral scripts
  - Auto-cleanup via TTL or max size (optional)

- [x] **3.3** Implement `LocalFileMediaStore`
  - `agentflow/media/storage/local_store.py`
  - Configurable base directory (default `./agentflow_media/`)
  - Storage layout: `{base_dir}/{key[:2]}/{key[2:4]}/{key}.{ext}` (sharded to avoid too many files per dir)
  - Metadata stored in sidecar `{key}.json` file
  - Retrieve reads from disk; delete removes both files
  - Security: path traversal prevention, validate key format

- [x] **3.4** Implement `PgBlobStore` (for PG-only deployments)
  - `agentflow/media/storage/pg_store.py`
  - Uses **separate** `media_blobs` table (NOT in states/messages JSONB):
    ```sql
    CREATE TABLE media_blobs (
        storage_key VARCHAR(255) PRIMARY KEY,
        data BYTEA NOT NULL,
        mime_type VARCHAR(100) NOT NULL,
        size_bytes BIGINT,
        thread_id VARCHAR(255),  -- optional, for cleanup
        created_at TIMESTAMPTZ DEFAULT NOW(),
        metadata JSONB DEFAULT '{}'
    );
    CREATE INDEX idx_media_blobs_thread ON media_blobs(thread_id);
    ```
  - Stores actual bytes in `BYTEA` column — separate from message/state JSONB
  - Messages/state still only contain `MediaRef(kind="url", url="agentflow://media/key")`
  - Benefits: same PG infra, transactional consistency, no extra service
  - Trade-off: PG not ideal for large blobs; fine for <10MB typical images

- [x] **3.5** Create `MediaRefResolver` — resolves references at LLM call time
  - `agentflow/media/resolver.py`
  - `resolve_for_openai(ref: MediaRef) -> dict` — converts to OpenAI content part format
  - `resolve_for_google(ref: MediaRef) -> types.Part` — converts to Google Part
  - Handles all `MediaRef.kind` values: `"url"` (internal + external), `"data"`, `"file_id"`
  - For `agentflow://media/{key}` URLs: calls `MediaStore.retrieve()` to get bytes
  - For `https://` URLs: passes through to provider directly
  - For `data` kind: uses inline base64 directly (small payloads)
  - For `file_id` kind: uses provider's native file reference

- [x] **3.6** Add inline data guard / auto-offload hook
  - In `MediaProcessor` or as a standalone utility:
    ```python
    async def ensure_media_offloaded(message: Message, store: BaseMediaStore, max_inline: int = 50_000) -> Message:
        """Replace large inline data_base64 with MediaStore references."""
        for block in message.content:
            if hasattr(block, 'media') and block.media.kind == "data":
                raw_size = len(block.media.data_base64 or "") * 3 // 4  # approx decoded size
                if raw_size > max_inline:
                    data = base64.b64decode(block.media.data_base64)
                    key = await store.store(data, block.media.mime_type or "application/octet-stream")
                    block.media = store.to_media_ref(key, block.media.mime_type)
        return message
    ```
  - Callable at API ingestion boundary, NOT in checkpointer
  - Configurable via `MultimodalConfig.offload_policy` and `max_inline_bytes`

- [x] **3.7** Inject `MediaStore` into graph compilation
  - `graph.compile(media_store=LocalFileMediaStore("./uploads"))` or
  - `graph.compile(media_store=InMemoryMediaStore())`
  - Store reference available to Agent and converters during execution
  - Similar to how ADK's `ArtifactService` is injected

- [x] **3.8** Wire `MediaRefResolver` into converter pipeline
  - `_convert_dict()` (OpenAI) → uses resolver for ImageBlock/AudioBlock/DocumentBlock
  - `_handle_regular_message()` (Google) → uses resolver for same
  - Resolver is instantiated with the `MediaStore` from graph config

- [x] **3.9** Convenience helpers on `Message`
  - `Message.with_image(data: bytes, mime_type: str, store: BaseMediaStore) -> Message`
  - `Message.with_file(path: str, store: BaseMediaStore) -> Message`
  - These store-then-reference: `store.store(data) → to_media_ref() → ImageBlock(media=ref)`
  - Also support direct URL/file_id for cases where MediaStore isn't needed

- [x] **3.10** Write comprehensive tests
  - InMemoryMediaStore: store/retrieve/delete roundtrip
  - LocalFileMediaStore: same + path traversal prevention + cleanup
  - PgBlobStore: same + verify blobs NOT in states/messages tables
  - MediaRefResolver: all MediaRef kinds → correct OpenAI/Google format
  - Auto-offload: large inline → auto-replaced with store reference
  - End-to-end: upload image → store → message → checkpointer save → reload → resolve for LLM
  - Verify: after checkpointer roundtrip, states JSONB is small (no base64 blobs)

- [x] **3.11** Implement `CloudMediaStore` (S3/GCS) via `cloud-storage-manager`
  - `agentflow/media/storage/cloud_store.py`
  - Uses `cloud-storage-manager` package (`pip install cloud-storage-manager`)
  - Supports both AWS S3 and GCS through unified `CloudStorageFactory` interface
  - Blob + sidecar metadata JSON stored in bucket with sharded layout
  - Signed URL download via `httpx` (async) with `urllib` fallback
  - Temp file upload (bytes → tempfile → upload → cleanup)
  - `get_public_url()` bonus method for direct browser/client access
  - Added `cloud-storage` optional dependency: `pip install 10xscale-agentflow[cloud-storage]`

- [x] **3.12** Fix OpenAI Responses API multimodal input format
  - Added `_to_responses_content()` helper in `agent_internal/openai.py`
  - Converts Chat Completions content parts → Responses API format:
    - `text` → `input_text`, `image_url` → `input_image` (flattened URL), `input_audio` preserved
  - Wired into `_call_openai_responses()` for all message content

- [x] **3.13** Add multimodal response handling to OpenAI Responses converter
  - `_extract_media_from_message_item()`: handles `output_image` and `output_audio` entries
  - `_extract_image_generation()`: handles `image_generation_call` items (DALL-E etc.)
  - Both non-streaming `convert_response` and streaming paths updated

- [x] **3.14** Add Document & Video type support across all providers
  - Updated `_document_block_to_openai()` → proper `{"type": "document", "document": {...}}` format
  - Created `_video_block_to_openai()` → `{"type": "video", "video": {...}}`
  - Updated `_build_content()` to handle `VideoBlock` in multimodal branch
  - Updated `_to_responses_content()` → document (`input_text`/`input_file`) + video (`input_text` ref)
  - Updated `_content_parts_to_google()` → document (`Part(text=…)`/`Part.from_bytes()`/`Part.from_uri()`) + video

- [x] **3.15** Multi-agent media stripping for text-only agents
  - Created `strip_media_blocks()` in `converter.py` — removes non-text content parts from message dicts
  - Wired into `Agent.execute()` in `execution.py` — auto-strips when `multimodal_config is None`
  - Collapses single remaining text part back to plain string for maximum compatibility

- [x] **3.16** Streaming media extraction for OpenAI Responses converter
  - Added `output_item.done` handlers for `message` type → `_extract_media_from_message_item()`
  - Added `image_generation_call` / `image_generation` handlers → `_extract_image_generation()`
  - Both sync and async streaming paths updated

- [x] **3.17** Streaming media extraction for Google GenAI converter
  - Added `_process_inline_media_part()` and `_process_file_media_part()` calls in `_extract_delta_content_blocks()`
  - Streaming chunks with images/audio/video now extracted as `ContentBlock`s

- [x] **3.18** Comprehensive multimodal end-to-end tests (`tests/test_multimodal_e2e.py`)
  - 73 tests across 11 test classes covering:
    - `_build_content` with all media types (image, audio, document, video)
    - `strip_media_blocks` for multi-agent workflows
    - `_to_responses_content` (OpenAI Responses input)
    - `_content_parts_to_google` (Google GenAI input)
    - OpenAI Chat / Responses / Google GenAI converter output
    - Multi-agent image stripping integration
    - Edge cases and full pipeline integration
  - All 73 tests passing; full suite: 2279 passed, 0 failed

### How the Pieces Fit Together (State & Checkpointer Summary)

```
BEFORE (broken):
  Image bytes → base64 → MediaRef(kind="data", data_base64="...1MB...")
  → Message → AgentState.context → PgCheckpointer → 1MB in JSONB + 1MB in TEXT + 1MB in Redis

AFTER (fixed):
  Image bytes → MediaStore.store(bytes) → key "abc123"
  → MediaRef(kind="url", url="agentflow://media/abc123") ← ~100 bytes
  → Message → AgentState.context → PgCheckpointer → 100 bytes in JSONB + 100 bytes in TEXT + 100 bytes in Redis
  
  Actual binary stored ONCE in:
  - InMemoryMediaStore: Python dict (testing)
  - LocalFileMediaStore: filesystem (dev)
  - PgBlobStore: media_blobs BYTEA table (PG-only deployment)
  - CloudMediaStore: S3 / GCS bucket via cloud-storage-manager (production)
```

**Checkpointer itself: ZERO changes needed.**  
**DB schema for states/messages: ZERO changes needed.**  
**The fix is entirely at the ingestion boundary and the LLM conversion boundary.**

### Sprint 4: API Layer — File Upload Endpoints (pyagenity-api)
**Goal**: REST API support for multimodal messages. Document extraction (Sprint 2) is already wired in; this sprint adds the upload endpoints, invoke/stream multimodal support, and wires everything together.

- [ ] **4.1** Add file upload endpoint
  ```
  POST /v1/files/upload
  Content-Type: multipart/form-data
  
  Response: {
    "file_id": "file_abc123",
    "mime_type": "image/jpeg", 
    "size_bytes": 102400,
    "filename": "photo.jpg",
    "extracted_text": "... (populated for supported document types, null for images/binary)",
    "url": "/v1/files/file_abc123"  
  }
  ```
  - For images/audio: store binary in `MediaStore`, return `file_id`
  - For documents (PDF, DOCX, etc.): store binary **and** run `DocumentPipeline.extract()`, return both `file_id` and `extracted_text`
  - Enforce `MEDIA_MAX_SIZE_MB` limit

- [ ] **4.2** Add file retrieval endpoint
  ```
  GET /v1/files/{file_id}
  → Returns file binary with correct Content-Type
  
  GET /v1/files/{file_id}/info
  → Returns file metadata (filename, mime_type, size_bytes, extracted_text if available)
  ```

- [ ] **4.3** Update graph invoke/stream endpoints to accept multimodal messages
  - `GraphInputSchema.messages` already accepts `Message` with `ContentBlock` — no schema change needed
  - Ensure JSON deserialization of `ImageBlock`, `DocumentBlock` etc. works correctly in API request
  - When a `DocumentBlock` with `file_id` is present in an incoming message:
    - Look up cached extracted text (from upload in 4.1) and substitute `TextBlock`
    - If no cached extraction, run `DocumentPipeline` on-the-fly
  - Client sends:
    ```json
    {
      "messages": [{
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image", "media": {"kind": "url", "url": "https://..."}}
        ]
      }]
    }
    ```
  - Or with uploaded file:
    ```json
    {
      "messages": [{
        "role": "user", 
        "content": [
          {"type": "text", "text": "Analyze this PDF"},
          {"type": "document", "media": {"kind": "file_id", "file_id": "file_abc123", "mime_type": "application/pdf"}}
        ]
      }]
    }
    ```

- [ ] **4.4** Add multimodal config endpoint
  ```
  GET /v1/config/multimodal → returns current config
  PUT /v1/config/multimodal → update config (admin)
  ```

- [ ] **4.5** Wire up `MediaProcessor`, `MediaStore`, and `DocumentPipeline` in API server startup
  - Configure via environment variables / settings:
    - `MEDIA_STORAGE_TYPE=local|memory|s3|gcs|cloud`
    - `MEDIA_STORAGE_PATH=./uploads`
    - `MEDIA_MAX_SIZE_MB=25`
    - `DOCUMENT_HANDLING=extract_text|pass_raw|skip`
  - `DocumentPipeline` instantiated once at startup, injected via FastAPI dependency

- [ ] **4.6** Write API tests

### Sprint 5: Client SDK Support (agentflow-react)
**Goal**: TypeScript client support for multimodal

- [ ] **5.1** Update TypeScript message types
  - Add `ImageBlock`, `AudioBlock`, `DocumentBlock` types matching Python models
  - Update `ContentBlock` union type

- [ ] **5.2** Add file upload client methods
  ```typescript
  client.files.upload(file: File | Blob): Promise<FileRef>
  client.files.get(fileId: string): Promise<FileInfo>
  ```

- [ ] **5.3** Add multimodal message helpers
  ```typescript
  Message.withImage(text: string, imageUrl: string): Message
  Message.withFile(text: string, file: FileRef): Message
  ```

- [ ] **5.4** Update playground/UI components
  - File upload button in chat input
  - Image preview in message bubbles
  - Document icon/preview for PDFs
  - Drag & drop support

- [ ] **5.5** Write client tests

### Sprint 6: Advanced Features & Polish
**Goal**: Production readiness

- [ ] **6.1** Image processing utilities
  - Auto-resize large images before sending
  - Thumbnail generation for storage
  - PIL-based processing option (convert to JPEG, optimize)
  - EXIF rotation handling

- [ ] **6.2** Provider-specific optimizations
  - Google: Use File API for large files (>20MB)
  - OpenAI: Use file_search for PDFs when available
  - Caching: Don't re-upload same file to provider

- [ ] **6.3** Streaming support for multimodal
  - Ensure streaming responses with images work correctly
  - Handle image generation streaming (progressive)

- [ ] **6.4** Security hardening
  - File type validation (magic bytes, not just extension)
  - Max file size enforcement
  - Virus scanning hook (optional)
  - Rate limiting on uploads
  - Sanitize filenames

- [ ] **6.5** Documentation
  - Multimodal usage guide
  - API reference for file endpoints
  - Configuration guide
  - Examples: image analysis, document Q&A, multimodal agent

---

## Configuration Reference

### Agent-Level Config
```python
from agentflow.media.config import MultimodalConfig, ImageHandling, DocumentHandling

agent = Agent(
    model="gpt-4o",
    provider="openai",
    multimodal_config=MultimodalConfig(
        image_handling=ImageHandling.BASE64,          # base64 | url | file_id
        document_handling=DocumentHandling.EXTRACT_TEXT,  # extract_text | pass_raw | skip
        max_image_size_mb=10.0,
        max_image_dimension=2048,
    ),
)
```

### API-Level Config (pyagenity-api)
```python
# In API settings / .env
MULTIMODAL_IMAGE_HANDLING=base64          # How API stores/passes images
MULTIMODAL_DOCUMENT_HANDLING=extract_text  # How API handles documents
MEDIA_STORAGE_TYPE=local                   # local | memory | cloud (s3/gcs)
MEDIA_STORAGE_PATH=./uploads               # For local storage
MEDIA_MAX_SIZE_MB=25                       # Max file upload size
```

### Per-Request Override (via API)
```json
{
  "messages": [...],
  "config": {
    "multimodal": {
      "document_handling": "pass_raw"
    }
  }
}
```

---

## Provider Support Matrix

| Feature | OpenAI | Google Gemini | Notes |
|---------|--------|---------------|-------|
| Image (base64) | ✅ gpt-4o, gpt-4o-mini | ✅ All Gemini | Most universal |
| Image (URL) | ✅ All vision models | ✅ All Gemini | Requires public URL |
| Image (file_id) | ✅ Via Assistants API | ✅ File API | Provider-managed |
| PDF (raw) | ✅ gpt-4o (as images) | ✅ Gemini (native) | Google has better native PDF support |
| PDF (extract text) | ✅ All text models | ✅ All text models | Universal fallback |
| DOCX (extract text) | ✅ All text models | ✅ All text models | Always extract |
| Audio input | ✅ gpt-4o-audio | ✅ Gemini Live | Limited model support |
| Video input | ❌ Not supported | ✅ Gemini native | Google only |

---

## Priority & Dependencies

```
Sprint 1 (Foundation)     ← Start here, unblocks everything
    │
    ├─► Sprint 2 (Documents)    ← Independent from Sprint 3
    │
    ├─► Sprint 3 (MediaStore)   ← Independent from Sprint 2, CRITICAL for production
    │       │                      (without this, images bloat the DB)
    │       │
    └───┬───┘
        │
        └─► Sprint 4 (API)      ← Depends on Sprint 1 + Sprint 3 (file upload needs MediaStore)
                │
                └─► Sprint 5 (Client)  ← Depends on Sprint 4
                        │
                        └─► Sprint 6 (Polish)  ← After everything works

Sprint 1 is mandatory first.
Sprints 2 and 3 can run in parallel after Sprint 1.
Sprint 4 REQUIRES Sprint 3 (file uploads need MediaStore).
For dev/testing, Sprint 1 alone enables multimodal with inline base64 using InMemoryCheckpointer.
For production with PgCheckpointer, Sprint 3 is required BEFORE serving real traffic.
```

---

## Estimated Complexity

| Sprint | Files Changed | New Files | Complexity |
|--------|--------------|-----------|------------|
| Sprint 1 | 4-5 modified | 2 new | Medium — core pipeline changes |
| Sprint 2 | 2-3 modified | 5 new | Medium — new module, optional deps |
| Sprint 3 | 3-4 modified | 7 new | High — storage backends, resolver, auto-offload, wiring |
| Sprint 4 | 3-4 modified | 3 new | Medium — API endpoints + wiring |
| Sprint 5 | 5-6 modified | 2 new | Medium — TypeScript types + UI |
| Sprint 6 | 3-4 modified | 1-2 new | Low-Medium — polish & optimization |

## Key Architectural Decision: Why Checkpointer Doesn't Change

The temptation is to add media-aware serialization in the checkpointer (e.g., intercept `model_dump()`, detect large base64, store separately). **Don't do this.** Reasons:

1. **Violates single responsibility** — Checkpointer's job is state persistence, not media management
2. **Creates hidden coupling** — Checkpointer would need a MediaStore reference, complicating DI
3. **Breaks deserialization** — If checkpointer strips data on save, it needs to re-inject on load; this is fragile
4. **Not where the problem is** — The problem is at the *ingestion* boundary, not the *persistence* boundary

Instead, the fix follows the **"clean input" principle**: ensure that by the time data reaches `AgentState.context`, all large binary payloads have already been offloaded to `MediaStore` and replaced with lightweight `MediaRef` references. The checkpointer then serializes these references like any other Pydantic model — no special handling needed.

The only optional guard is the `ensure_media_offloaded()` function that can be called at the API boundary as a safety net. If somehow a large inline base64 slips through, it warns (or auto-offloads) before the message enters the state.
