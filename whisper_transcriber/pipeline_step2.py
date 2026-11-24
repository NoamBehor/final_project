import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_DICT_PATH = os.path.join(BASE_DIR, "he_50k.txt")

HEBREW_PREFIXES = ('ו', 'ה', 'ב', 'ל', 'מ', 'ש', 'כ')

_stanza_nlp = None
_stanza_unavailable = False
_local_bert_masker = None


def _get_stanza_ner():
    global _stanza_nlp, _stanza_unavailable
    if _stanza_unavailable:
        return None
    if _stanza_nlp is None:
        try:
            import stanza
            _stanza_nlp = stanza.Pipeline(
                lang='he',
                processors='tokenize,ner',
                use_gpu=False,
                verbose=False,
                download_method=None
            )
        except Exception:
            _stanza_unavailable = True
            return None
    return _stanza_nlp


def _get_bert_masker(external_masker=None):
    global _local_bert_masker
    if external_masker is not None:
        return external_masker
    if _local_bert_masker is None:
        from transformers import pipeline
        _local_bert_masker = pipeline("fill-mask", model="onlplab/alephbert-base")
    return _local_bert_masker


def _load_core_vocab(filepath=DEFAULT_DICT_PATH, max_words=35000):
    vocab = set()
    if not os.path.exists(filepath):
        return vocab
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                vocab.add(parts[0])
            if len(vocab) >= max_words:
                break
    return vocab


VALID_HEBREW_WORDS = _load_core_vocab()


def clean_punctuation(word):
    return ''.join(char for char in word if char.isalpha())


def is_known_hebrew_word(clean_word: str, vocab: set[str]) -> bool:
    # Step 1: Check full word first (handles root words starting with ה/ש like הנהלה, הציג)
    if clean_word in vocab:
        return True

    # Step 2: Check after stripping 1 proclitic prefix (e.g., הנסונים -> נסונים)
    if len(clean_word) > 3 and clean_word[0] in HEBREW_PREFIXES:
        if clean_word[1:] in vocab:
            return True

    # Step 3: Check after stripping 2 proclitic prefixes (e.g., שהידים -> הידים)
    if len(clean_word) > 4 and clean_word[0] in HEBREW_PREFIXES and clean_word[1] in HEBREW_PREFIXES:
        if clean_word[2:] in vocab:
            return True

    return False


def analyze_spans(step1_json_data, bert_masker=None, valid_words=None):
    suspicious_words = []
    vocab = valid_words if valid_words is not None else VALID_HEBREW_WORDS
    masker = _get_bert_masker(bert_masker)

    for segment in step1_json_data:
        full_text = segment["text"]

        for word_data in segment["words"]:
            original_word = word_data["word"]
            clean_word = clean_punctuation(original_word)
            acoustic_prob = word_data["probability"]

            # Filter 1: Acoustic confidence from Whisper
            if acoustic_prob < 0.75:
                suspicious_words.append({
                    "word": original_word,
                    "reason": "Acoustic (Low Confidence)",
                    "score": round(acoustic_prob, 2)
                })
                continue

            # Filter 2: Cascading OOV check with optional Stanza NER guard
            if clean_word and vocab and not is_known_hebrew_word(clean_word, vocab):
                is_named_entity = False
                nlp_ner = _get_stanza_ner()
                if nlp_ner is not None:
                    doc = nlp_ner(clean_word)
                    is_named_entity = any(ent.type in ['PERSON', 'ORG', 'LOC'] for ent in doc.ents)

                if not is_named_entity:
                    suspicious_words.append({
                        "word": original_word,
                        "reason": "Lexical (OOV / Not in Dictionary)",
                        "score": round(acoustic_prob, 2)
                    })
                    continue

                # Filter 3: Contextual anomaly check with AlephBERT (catches real-word errors like שהידים)
            if len(clean_word) > 3:
                masked_text = full_text.replace(original_word, "[MASK]", 1)
                predictions = masker(masked_text, top_k=250)
                
                if len(predictions) > 0 and isinstance(predictions[0], list):
                    predictions = predictions[0]

                top_predicted_words = [
                    pred['token_str'].replace("##", "").strip()
                    for pred in predictions
                ]

                match_found = False
                for pred_word in top_predicted_words:
                    if pred_word == clean_word:
                        match_found = True
                        break
                    if len(pred_word) >= 3 and (pred_word in clean_word or clean_word in pred_word):
                        match_found = True
                        break

                if not match_found:
                    suspicious_words.append({
                        "word": original_word,
                        "reason": "Contextual Anomaly (BERT doesn't match context)",
                        "score": round(acoustic_prob, 2)
                    })

    return suspicious_words


if __name__ == "__main__":
    from whisper_transcriber.pipeline_step1 import ingest_audio

    audio_file = os.path.join(BASE_DIR, "nlp_corrector", "test1.mp3")
    step1_output = ingest_audio(audio_file)
    detected_errors = analyze_spans(step1_output)
    print(json.dumps(detected_errors, indent=4, ensure_ascii=False))