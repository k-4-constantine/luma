# backend/services/transcription_service.py
from pathlib import Path
from backend.config import settings
from deepgram import DeepgramClient, DeepgramClientEnvironment

class TranscriptionService:
    def __init__(self):
        # Remove /v1 suffix from base URL for Deepgram SDK (it adds its own path)
        base_url = settings.greenpt_base_url.rstrip('/v1').rstrip('/')
        greenpt_env = DeepgramClientEnvironment(
            base=base_url,
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