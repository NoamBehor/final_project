import os
import json
from faster_whisper import WhisperModel

_model = None


def _get_whisper_model() -> WhisperModel:
    global _model
    if _model is None:
        preferred_model = os.getenv("WHISPER_MODEL", "medium")
        for model_name in [preferred_model, "small"]:
            try:
                print(f"Loading Whisper model ({model_name})...")
                _model = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=2,
                    num_workers=1
                )
                break
            except RuntimeError as err:
                print(f"[Warning] Could not allocate memory for '{model_name}' ({err}). Falling back...")
                if model_name == "small":
                    raise
    return _model


def ingest_audio(audio_path: str) -> list[dict]:
    whisper_model = _get_whisper_model()
    print(f"Transcribing {audio_path}...")

    segments, _ = whisper_model.transcribe(audio_path, word_timestamps=True, language="he")
    pipeline_data = []

    for segment in segments:
        # Skip background noise or silence segments
        if segment.no_speech_prob > 0.6 or segment.avg_logprob < -1.0:
            print(f"[Skipped Noise/Silence] avg_logprob: {segment.avg_logprob:.2f}")
            continue

        segment_dict = {
            "text": segment.text.strip(),
            "start": segment.start,
            "end": segment.end,
            "avg_logprob": segment.avg_logprob,
            "words": []
        }

        for word in segment.words:
            segment_dict["words"].append({
                "word": word.word.strip(),
                "start": word.start,
                "end": word.end,
                "probability": word.probability
            })

        pipeline_data.append(segment_dict)

    return pipeline_data


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_audio_file = os.path.join(base_dir, "nlp_corrector", "test1.mp3")

    try:
        results = ingest_audio(test_audio_file)
        print(json.dumps(results, indent=4, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")