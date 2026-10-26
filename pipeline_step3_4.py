import difflib
from transformers import pipeline

from pipeline_step1 import ingest_audio
from pipeline_step2 import analyze_spans
from pipeline_step6 import crop_audio_for_review

print("Loading AlephBERT for Contextual Scoring...")
bert_masker = pipeline("fill-mask", model="onlplab/alephbert-base")

def load_frequency_dictionary(filepath, max_words=10000):
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
        print(f"Error: Dictionary file {filepath} not found. Using empty dictionary.")
        return []

VALID_DICTIONARY = load_frequency_dictionary("he_50k.txt", max_words=10000)

def generate_candidates(wrong_word, vocab, max_candidates=3):
    return difflib.get_close_matches(wrong_word, vocab, n=max_candidates, cutoff=0.4)

def score_candidates(sentence, wrong_word, candidates):
    if not candidates:
        return []
        
    masked_sentence = sentence.replace(wrong_word, "[MASK]")
    predictions = bert_masker(masked_sentence, top_k=150)
    
    # תיקון הבאג של המסיכות הכפולות (כמו במילה "את" ו-"אתמול")
    if len(predictions) > 0 and isinstance(predictions[0], list):
        predictions = predictions[0]
        
    scored_candidates = []
    
    for candidate in candidates:
        best_score = 0.0
        for pred in predictions:
            pred_token = pred['token_str'].replace("##", "").strip()
            if len(pred_token) >= 3 and (pred_token in candidate or candidate in pred_token):
                if pred['score'] > best_score:
                    best_score = pred['score']
        
        scored_candidates.append({
            'token_str': candidate,
            'score': best_score
        })
        
    return sorted(scored_candidates, key=lambda x: x['score'], reverse=True)

def make_decision(scored_candidates, min_score_threshold=0.015, min_gap=0.005):
    if not scored_candidates:
        return {"action": "ABSTAIN", "reason": "No candidates available", "word": None}
        
    top_candidate = scored_candidates[0]
    
    if top_candidate['score'] < min_score_threshold:
        return {"action": "ABSTAIN", "reason": f"Top score ({top_candidate['score']:.4f}) is below threshold", "word": None}
        
    if len(scored_candidates) > 1:
        second_candidate = scored_candidates[1]
        gap = top_candidate['score'] - second_candidate['score']
        if gap < min_gap:
            return {"action": "ABSTAIN", "reason": f"Gap ({gap:.4f}) is too small", "word": None}
            
    return {"action": "SUGGEST", "reason": "Passed all confidence thresholds", "word": top_candidate['token_str']}

if __name__ == "__main__":
    audio_file = "test1.mp3" 
    
    print("\n" + "="*50)
    print(f"🚀 Starting MS2 Pipeline on: {audio_file}")
    print("="*50)

    # שלב 1: קליטה אקוסטית
    print("\n[Step 1] Transcribing audio with Whisper...")
    step1_data = ingest_audio(audio_file)
    original_sentence = " ".join([seg["text"] for seg in step1_data])
    
    # שלב 2: זיהוי שגיאות
    print("\n[Step 2] Running SpanDetector...")
    detected_errors = analyze_spans(step1_data)
    
    if not detected_errors:
        print("\n✅ No errors detected!")
    else:
        print(f"Found {len(detected_errors)} potential errors. Filtering by acoustic confidence...\n")
        
        for error in detected_errors:
            wrong_word = error["word"]
            
            # חילוץ ציון הביטחון המקורי של Whisper מהשלב הראשון
            acoustic_confidence = 1.0
            start_sec, end_sec = 0, 0
            for seg in step1_data:
                for w in seg["words"]:
                    if w["word"] == wrong_word:
                        acoustic_confidence = w["probability"]
                        start_sec, end_sec = w["start"], w["end"]
                        break
            
            # --- הפתרון החכם ---
            # אם Whisper בטוח ב-85% ומעלה שזה נכון, כנראה שזו מילה תקינה עם קידומת. מתעלמים!
            if acoustic_confidence >= 0.85:
                print(f"⏩ Skipping '{wrong_word}' (Whisper is confident: {acoustic_confidence:.2f})")
                continue
                
            print("-" * 40)
            print(f"🔍 Analyzing Real Error: '{wrong_word}' (Confidence: {acoustic_confidence:.2f})")
            
            candidates = generate_candidates(wrong_word, VALID_DICTIONARY)
            scored_results = score_candidates(original_sentence, wrong_word, candidates)
            decision = make_decision(scored_results)
            
            if decision["action"] == "SUGGEST":
                print(f"  [✅ FIXED]: Replaced with '{decision['word']}'")
                original_sentence = original_sentence.replace(wrong_word, decision['word'])
            else:
                print(f"  [⚠️ ABSTAIN]: {decision['reason']}")
                
                if start_sec > 0:
                    safe_word = wrong_word.replace(",", "").replace(".", "")
                    output_wav = f"review_{safe_word}.wav"
                    crop_audio_for_review(audio_file, start_sec, end_sec, output_wav)

    print("\n" + "="*50)
    print(f"🏁 Final Text: {original_sentence}")
    print("="*50)