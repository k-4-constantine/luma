import os
from pathlib import Path
from dotenv import load_dotenv
from deepgram import DeepgramClient
from deepgram.environment import DeepgramClientEnvironment

load_dotenv()
RAG_API_KEY = os.getenv("API_KEY")
RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL")
GREENPT_API_KEY = os.getenv("GREENPT_API")
GREENPT_BASE_URL = "https://api.greenpt.ai/v1"

data_dir = Path('Example-Files')
#transcripts = Path('transcripts')

audio_files = []
for audio_file_path in data_dir.glob("*.mp4"):
  audio_files.append(audio_file_path)

greenpt_env = DeepgramClientEnvironment(
    base="https://api.greenpt.ai",
    production="wss://api.greenpt.ai",
    agent="wss://api.greenpt.ai",
)

deepgram = DeepgramClient(api_key=GREENPT_API_KEY, environment=greenpt_env)
for audio_file_path in audio_files:
  with open(audio_file_path, "rb") as file:
    response = deepgram.listen.v1.media.transcribe_file(
      request=file.read(),
      model="green-s",
      smart_format=True
    )
  response_dict = response.model_dump()

  transcript = response_dict['results']['channels'][0]['alternatives'][0]['transcript']
  with open(data_dir / f"{audio_file_path.stem}.txt", "w", encoding='utf-8') as f:
    f.write(transcript)
  print("Transcription successful")


