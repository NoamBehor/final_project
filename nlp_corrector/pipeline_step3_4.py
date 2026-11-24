import os
import re
import math
import difflib
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DICT_PATH = os.path.join(BASE_DIR, "he_50k.txt")

print("Loading AlephBERT for Contextual Scoring...")
_MODEL_NAME = "onlplab/alephbert-base"

try:
    _config = AutoConfig.from_pretrained(_MODEL_NAME, local_files_only=True)
    _config.tie_word_embeddings = False
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, local_files_only=True)
    _mlm_model = AutoModelForMaskedLM.from_pretrained(
        _MODEL_NAME,
        config=_config,
        local_files_only=True,
    )
except Exception:
    _config = AutoConfig.from_pretrained(_MODEL_NAME)
    _config.tie_word_embeddings = False
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    _mlm_model = AutoModelForMaskedLM.from_pretrained(_MODEL_NAME, config=_config)

with torch.no_grad():
    _mlm_model.cls.predictions.decoder.weight.copy_(
        _mlm_model.bert.embeddings.word_embeddings.weight
    )
_mlm_model.eval()


class _AlephBertWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, masked_text, top_k=100):
        inputs = self.tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=256)
        mask_positions = (inputs["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            return []
        mask_idx = mask_positions[0].item()
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, mask_idx]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[0]))
        return [
            {"token_str": self.tokenizer.convert_ids_to_tokens(tid.item()), "score": prob.item()}
            for tid, prob in zip(top_ids, top_probs)
        ]


bert_masker = _AlephBertWrapper(_mlm_model, _tokenizer)


def load_frequency_dictionary(filepath=DICT_PATH, max_words=35000):
    words = []
    print(f"Loading top {max_words} most common words from {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    words.append(parts[0])
                if len(words) >= max_words:
                    break
        print(f"Successfully loaded {len(words)} core vocabulary words.")
        return words
    except FileNotFoundError:
        print(f"Error: Dictionary file {filepath} not found.")
        return []


VALID_DICTIONARY = load_frequency_dictionary(DICT_PATH, max_words=35000)
VALID_WORD_SET = set(VALID_DICTIONARY)
HEBREW_PREFIXES = ('ו', 'ה', 'ב', 'ל', 'מ', 'ש', 'כ')


def _strip_punctuation(token: str) -> tuple[str, str, str]:
    match = re.match(r'^([^\w\u0590-\u05FF]*)([\w\u0590-\u05FF]+)([^\w\u0590-\u05FF]*)$', token)
    if not match:
        return "", token, ""
    return match.group(1), match.group(2), match.group(3)


def generate_candidates(wrong_word, vocab, max_candidates=15):
    lead_punct, clean_word, trail_punct = _strip_punctuation(wrong_word)
    candidate_pool = []

    full_matches = difflib.get_close_matches(clean_word, vocab, n=12, cutoff=0.50)
    candidate_pool.extend(full_matches)

    if len(clean_word) > 3 and clean_word[0] in HEBREW_PREFIXES:
        p1, base1 = clean_word[0], clean_word[1:]
        base1_matches = difflib.get_close_matches(base1, vocab, n=10, cutoff=0.45)
        candidate_pool.extend(p1 + m for m in base1_matches)

    if len(clean_word) > 4 and clean_word[0] in HEBREW_PREFIXES and clean_word[1] in HEBREW_PREFIXES:
        p2, base2 = clean_word[:2], clean_word[2:]
        base2_matches = difflib.get_close_matches(base2, vocab, n=10, cutoff=0.45)
        candidate_pool.extend(p2 + m for m in base2_matches)

    unique_clean = [c for c in dict.fromkeys(candidate_pool) if c != clean_word][:max_candidates]
    return [f"{lead_punct}{c}{trail_punct}" for c in unique_clean]

    print(f"    [Debug] Proposed Candidates to BERT: {final_candidates}")
    return final_candidates


def _score_candidate_in_context(sentence: str, target_word: str, candidate_word: str) -> float:
    _, clean_cand, _ = _strip_punctuation(candidate_word)
    cand_token_ids = _tokenizer.encode(clean_cand, add_special_tokens=False)
    if not cand_token_ids:
        return 0.0

    # Replace target word with exact number of [MASK] tokens matching the candidate's subwords
    mask_placeholders = " ".join([_tokenizer.mask_token] * len(cand_token_ids))
    masked_sentence = sentence.replace(target_word, mask_placeholders, 1)

    inputs = _tokenizer(masked_sentence, return_tensors="pt", truncation=True, max_length=256)
    mask_positions = (inputs["input_ids"][0] == _tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) != len(cand_token_ids):
        return 0.0

    with torch.no_grad():
        logits = _mlm_model(**inputs).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)

    total_log_prob = 0.0
    for pos, tok_id in zip(mask_positions, cand_token_ids):
        total_log_prob += log_probs[pos, tok_id].item()

    # Length-normalized geometric mean probability so 1-token and 2-token words compete fairly
    avg_log_prob = total_log_prob / len(cand_token_ids)
    return math.exp(avg_log_prob)


def score_candidates(sentence, wrong_word, candidates):
    if not candidates:
        return []

    scored_candidates = []
    for candidate in candidates:
        score = _score_candidate_in_context(sentence, wrong_word, candidate)
        scored_candidates.append({
            'token_str': candidate,
            'score': score
        })

    ranked = sorted(scored_candidates, key=lambda x: x['score'], reverse=True)
    for item in ranked[:3]:
        print(f"      -> Candidate '{item['token_str']}': score={item['score']:.5f}")
    return ranked


def make_decision(scored_candidates, min_score_threshold=0.001, min_gap=0.0001):
    if not scored_candidates:
        return {"action": "ABSTAIN", "reason": "No candidates available", "word": None}

    top_candidate = scored_candidates[0]

    if top_candidate['score'] < min_score_threshold:
        return {
            "action": "ABSTAIN",
            "reason": f"Top score ({top_candidate['score']:.5f}) is below threshold",
            "word": None
        }

    if len(scored_candidates) > 1:
        second_candidate = scored_candidates[1]
        gap = top_candidate['score'] - second_candidate['score']
        if gap < min_gap:
            return {
                "action": "ABSTAIN",
                "reason": f"Gap ({gap:.5f}) is too small",
                "word": None
            }

    return {
        "action": "SUGGEST",
        "reason": "Passed all confidence thresholds",
        "word": top_candidate['token_str']
    }