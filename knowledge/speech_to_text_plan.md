# Speech-to-Text Integration Plan

## Overview
Add transcription service to convert audio/video files to text, save as .txt, and auto-process into RAG database.

## Requirements
- Synchronous processing (user waits)
- Save transcripts to Example-Files/
- Upload button in chat UI
- Show transcript with download link
- Extend RAG to support .txt files

## Implementation Steps

### 1. Add .txt File Support to Models
**File:** `backend/models/document.py` (line 8-12)
- Add `TXT = "txt"` to FileType enum

### 2. Create TranscriptionService
**File:** `backend/services/transcription_service.py` [NEW]
```python
class TranscriptionService:
    def __init__(self):
        greenpt_env = DeepgramClientEnvironment(
            base="https://api.greenpt.ai",
            production="wss://api.greenpt.ai",
            agent="wss://api.greenpt.ai"
        )
        self.client = DeepgramClient(api_key=settings.greenpt_api_key, environment=greenpt_env)

    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        response = self.client.listen.v1.media.transcribe_file(
            request=audio_bytes, model="green-s", smart_format=True
        )
        return response.model_dump()['results']['channels'][0]['alternatives'][0]['transcript']

    async def save_transcript(self, transcript: str, original_filename: str) -> Path:
        txt_path = settings.documents_path / f"{Path(original_filename).stem}.txt"
        txt_path.write_text(transcript, encoding='utf-8')
        return txt_path
```

### 3. Extend DocumentProcessor for .txt Files
**File:** `backend/services/document_processor.py` (line 41)
- Add after PPTX elif branch:
```python
elif file_path.suffix.lower() == ".txt":
    full_content = file_path.read_text(encoding='utf-8')
    page_texts = [full_content]
    author = None
    created_at = None
    file_type = FileType.TXT
```

**File:** `backend/main.py` (line 70)
- Add ".txt" to supported_extensions list

### 4. Add /api/transcribe Endpoint
**File:** `backend/api/routes.py` (after line 94)
```python
@router.post("/transcribe")
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    # Validate file type (.mp3, .mp4, .wav, .m4a)
    # Transcribe using TranscriptionService
    # Save to Example-Files
    # Auto-process into RAG (reuse main.py lines 84-96 pattern)
    # Return {filename, transcript, chunks_added}
```

### 5. Initialize TranscriptionService
**File:** `backend/main.py`
- Add after line 23: `transcription_service = TranscriptionService()`
- Add after line 35: `app.state.transcription_service = transcription_service`

### 6. Frontend Upload UI
**File:** `webpages/app.html`

**HTML (line 240):**
```html
<input type="file" id="audio-upload" accept=".mp3,.mp4,.wav,.m4a" style="display:none" />
<button id="upload-btn" onclick="document.getElementById('audio-upload').click()">🎤</button>
```

**CSS (after line 108):**
```css
#upload-btn { /* Same style as #prompt-btn */ }
```

**JavaScript (after line 296):**
```javascript
document.getElementById('audio-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Disable buttons, show loading
    // POST to /api/transcribe with FormData
    // Display transcript in chat with download link
    // Re-enable buttons
});
```

### 7. Enable Static File Downloads
**File:** `backend/main.py` (after line 132)
```python
app.mount("/Example-Files", StaticFiles(directory=str(settings.documents_path)), name="files")
```

## Critical Files
- `backend/services/transcription_service.py` [NEW]
- `backend/services/document_processor.py` [MODIFY]
- `backend/api/routes.py` [MODIFY]
- `backend/models/document.py` [MODIFY]
- `backend/main.py` [MODIFY]
- `webpages/app.html` [MODIFY]

## Validation
1. Upload audio file → transcribes synchronously
2. Check Example-Files → .txt file saved
3. Chat about content → retrieves from RAG
4. Download link → works
