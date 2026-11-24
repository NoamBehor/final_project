import sys
import os
import time
import pika

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared_dto.transcription_dto import TranscriptionDTO


def run_nlp_correction(text: str) -> str:
    # Hook your existing AlephBERT + Stanza correction pipeline here
    return f"{text} [מתוקן]"


def process_message(ch, method, properties, body: bytes) -> None:
    dto = TranscriptionDTO.model_validate_json(body.decode('utf-8'))
    print(f"Received DTO {dto.message_id}: {dto.original_text}")

    dto.corrected_text = run_nlp_correction(dto.original_text)
    dto.is_corrected = True

    ch.basic_publish(
        exchange='',
        routing_key='nlp_to_search',
        body=dto.model_dump_json(),
        properties=pika.BasicProperties(delivery_mode=2)
    )
    print(f"Forwarded corrected DTO {dto.message_id} to 'nlp_to_search'")

    ch.basic_ack(delivery_tag=method.delivery_tag)


def start_worker() -> None:
    credentials = pika.PlainCredentials('admin', 'admin123')
    parameters = pika.ConnectionParameters(
        host='localhost',
        port=5672,
        credentials=credentials,
        heartbeat=600,
        blocked_connection_timeout=300
    )

    while True:
        try:
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()

            channel.queue_declare(queue='whisper_to_nlp', durable=True)
            channel.queue_declare(queue='nlp_to_search', durable=True)

            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue='whisper_to_nlp', on_message_callback=process_message)

            print("NLP Worker listening on 'whisper_to_nlp'. Press Ctrl+C to stop.")
            channel.start_consuming()
        except KeyboardInterrupt:
            print("Worker stopped by user.")
            break
        except (pika.exceptions.AMQPConnectionError, pika.exceptions.StreamLostError) as err:
            print(f"Connection lost ({err}), reconnecting in 3 seconds...")
            time.sleep(3)


if __name__ == "__main__":
    start_worker()