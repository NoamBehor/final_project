import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import torch
import pika

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared_dto.transcription_dto import TranscriptionDTO
from whisper_transcriber.pipeline_step2 import analyze_spans
from nlp_corrector.pipeline_step3_4 import (
    VALID_DICTIONARY,
    bert_masker,
    generate_candidates,
    score_candidates,
    make_decision,
)
from nlp_corrector.pipeline_step6 import crop_audio_for_review

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REVIEW_DIR = os.path.join(BASE_DIR, "review_snippets")
os.makedirs(REVIEW_DIR, exist_ok=True)

_embed_tokenizer = bert_masker.tokenizer
_embed_model = bert_masker.model
_embed_model.eval()


def compute_alephbert_embedding(text: str) -> list[float]:
    if not text.strip():
        return [0.0] * 768
    inputs = _embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = _embed_model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = (summed / counts).squeeze(0)
        normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=0)
    return normalized.tolist()


def run_ms2_correction(dto: TranscriptionDTO) -> tuple[str, list[float]]:
    corrected_sentence = dto.original_text
    step1_data = dto.segments

    if not step1_data:
        return corrected_sentence, compute_alephbert_embedding(corrected_sentence)

    audio_path = dto.audio_file_path
    if not os.path.isabs(audio_path):
        audio_path = os.path.join(BASE_DIR, audio_path)

    detected_errors = analyze_spans(
        step1_data,
        bert_masker=bert_masker,
        valid_words=set(VALID_DICTIONARY),
    )

    for error in detected_errors:
        wrong_word = error["word"]
        reason = error.get("reason", "")
        acoustic_confidence = 1.0
        start_sec, end_sec = 0.0, 0.0

        for seg in step1_data:
            for w in seg["words"]:
                if w["word"] == wrong_word:
                    acoustic_confidence = w["probability"]
                    start_sec, end_sec = w["start"], w["end"]
                    break

        if acoustic_confidence >= 0.85 and "OOV" not in reason and "Contextual Anomaly" not in reason:
            continue

        candidates = generate_candidates(wrong_word, VALID_DICTIONARY)
        scored_results = score_candidates(corrected_sentence, wrong_word, candidates)
        decision = make_decision(scored_results)

        if decision["action"] == "SUGGEST" and decision["word"]:
            print(f"  [FIXED] '{wrong_word}' -> '{decision['word']}'")
            corrected_sentence = corrected_sentence.replace(wrong_word, decision["word"])
        else:
            print(f"  [ABSTAIN] '{wrong_word}': {decision['reason']}")
            if end_sec > start_sec and os.path.exists(audio_path):
                safe_word = "".join(c for c in wrong_word if c.isalnum())
                snippet_path = os.path.join(REVIEW_DIR, f"{dto.message_id}_{safe_word}.wav")
                crop_audio_for_review(audio_path, start_sec, end_sec, snippet_path)

    embedding = compute_alephbert_embedding(corrected_sentence)
    return corrected_sentence, embedding


def process_message(ch, method, properties, body: bytes) -> None:
    dto = TranscriptionDTO.model_validate_json(body.decode("utf-8"))
    print(f"\n[NLP Worker] Processing DTO {dto.message_id}: '{dto.original_text}'")

    corrected_text, embedding = run_ms2_correction(dto)
    dto.corrected_text = corrected_text
    dto.is_corrected = True
    dto.embedding = embedding

    ch.basic_publish(
        exchange="",
        routing_key="nlp_to_search",
        body=dto.model_dump_json(),
        properties=pika.BasicProperties(delivery_mode=2),
    )
    print(f"[NLP Worker] Forwarded DTO {dto.message_id} (dim={len(dto.embedding)}) to 'nlp_to_search'")
    ch.basic_ack(delivery_tag=method.delivery_tag)


def start_worker() -> None:
    credentials = pika.PlainCredentials("admin", "admin123")
    parameters = pika.ConnectionParameters(
        host="localhost",
        port=5672,
        credentials=credentials,
        heartbeat=600,
        blocked_connection_timeout=300,
    )

    while True:
        try:
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()

            channel.queue_declare(queue="whisper_to_nlp", durable=True)
            channel.queue_declare(queue="nlp_to_search", durable=True)

            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue="whisper_to_nlp", on_message_callback=process_message)

            print("NLP Worker (MS2 + AlephBERT Embeddings) listening on 'whisper_to_nlp'...")
            channel.start_consuming()
        except KeyboardInterrupt:
            print("Worker stopped by user.")
            break
        except (pika.exceptions.AMQPConnectionError, pika.exceptions.StreamLostError) as err:
            print(f"Connection lost ({err}), reconnecting in 3 seconds...")
            time.sleep(3)


if __name__ == "__main__":
    start_worker()