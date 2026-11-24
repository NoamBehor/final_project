import sys
import os
import uuid
from datetime import datetime, timezone
import pika

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared_dto.transcription_dto import TranscriptionDTO


def publish_transcription(dto: TranscriptionDTO) -> None:
    credentials = pika.PlainCredentials('admin', 'admin123')
    parameters = pika.ConnectionParameters(host='localhost', port=5672, credentials=credentials)

    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(queue='whisper_to_nlp', durable=True)

    channel.basic_publish(
        exchange='',
        routing_key='whisper_to_nlp',
        body=dto.model_dump_json(),
        properties=pika.BasicProperties(delivery_mode=2)
    )
    print(f"Sent DTO {dto.message_id} to queue 'whisper_to_nlp'")
    connection.close()


if __name__ == "__main__":
    sample_dto = TranscriptionDTO(
        message_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        frequency="91.8FM",
        audio_file_path="whisper_transcriber/test1.mp3",
        original_text="שידור ניסיון לבדיקת תור ההודעות"
    )
    publish_transcription(sample_dto)