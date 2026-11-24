import os
import sys

# Must be set BEFORE importing numpy or faster_whisper on Windows
os.environ["MKL_DISABLE_FAST_MM"] = "1"
os.environ["CT2_USE_MKL"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CT2_VERBOSE"] = "0"

import uuid
from datetime import datetime, timezone
import pika

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared_dto.transcription_dto import TranscriptionDTO
from whisper_transcriber.pipeline_step1 import ingest_audio

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def transcribe_and_publish(relative_audio_path: str, frequency: str = "91.8FM") -> None:
    full_audio_path = os.path.join(BASE_DIR, relative_audio_path)
    print(f"[Producer] Transcribing '{relative_audio_path}' with Whisper...")

    segments = ingest_audio(full_audio_path)
    raw_text = " ".join(seg["text"] for seg in segments).strip()

    dto = TranscriptionDTO(
        message_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        frequency=frequency,
        audio_file_path=relative_audio_path,
        original_text=raw_text,
        segments=segments,
    )

    credentials = pika.PlainCredentials("admin", "admin123")
    parameters = pika.ConnectionParameters(host="localhost", port=5672, credentials=credentials)

    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue="whisper_to_nlp", durable=True)

    channel.basic_publish(
        exchange="",
        routing_key="whisper_to_nlp",
        body=dto.model_dump_json(),
        properties=pika.BasicProperties(delivery_mode=2),
    )
    print(f"[Producer] Sent DTO {dto.message_id} ('{raw_text}') to 'whisper_to_nlp'")
    connection.close()


if __name__ == "__main__":
    transcribe_and_publish("nlp_corrector/test1.mp3", frequency="91.8FM")