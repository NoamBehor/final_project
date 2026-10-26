import json
import stanza
from transformers import pipeline
from pipeline_step1 import ingest_audio


print("Loading Models... (This might take a moment on the first run as it downloads AlephBERT and Stanza)")

# 1. טעינת מודל ההקשר (AlephBERT)
bert_masker = pipeline("fill-mask", model="onlplab/alephbert-base")

# 2. טעינת Stanza (מוריד את המודל העברי אם הוא לא קיים)
stanza.download('he', processors='tokenize,ner')
stanza_nlp = stanza.Pipeline(lang='he', processors='tokenize,ner', use_gpu=False)

# מילון עברי מדומה לצורך הבדיקה (במערכת אמיתית נחבר מילון מלא או ספריית spellchecker)
VALID_HEBREW_WORDS = ["שלום", "זה", "משפט", "בדיקה", "עבור", "המערכת", "אתמול", "הנוסע", "בצד", "תאונה", "להגיע", "בזמן", "לבית", "חולים"]

def clean_punctuation(word):
    """מסיר פסיקים ונקודות כדי לבדוק את המילה עצמה במילון"""
    return ''.join(char for char in word if char.isalpha())

def analyze_spans(step1_json_data):
    suspicious_words = []
    
    for segment in step1_json_data:
        full_text = segment["text"]
        
        for word_data in segment["words"]:
            original_word = word_data["word"]
            clean_word = clean_punctuation(original_word)
            acoustic_prob = word_data["probability"]
            
            # --- מסננת 1: ביטחון אקוסטי מ-Whisper ---
            if acoustic_prob < 0.75:
                suspicious_words.append({
                    "word": original_word,
                    "reason": "Acoustic (Low Confidence)",
                    "score": round(acoustic_prob, 2)
                })
                continue # המילה סומנה, עוברים למילה הבאה
                
            # --- מסננת 2: מילון (OOV) ---
            if clean_word and clean_word not in VALID_HEBREW_WORDS:
                # רגע לפני שאנחנו פוסלים אותה, נבדוק עם Stanza שזה לא שם פרטי
                doc = stanza_nlp(clean_word)
                is_named_entity = any(ent.type in ['PERSON', 'ORG', 'LOC'] for ent in doc.ents)
                
                if not is_named_entity:
                    suspicious_words.append({
                        "word": original_word,
                        "reason": "Lexical (OOV / Not in Dictionary)"
                    })
                    continue
            
            # --- מסננת 3: הקשר סמנטי עם AlephBERT ---
            # כאן המערכת בודקת מילים ש"עברו" את שני השלבים הראשונים כדי לתפוס התנגשויות (מילה תקינה במקום שגוי).
            # בשלב זה נדלג על מילות קישור קצרות כדי לא להעמיס על המודל
            # --- מסננת 3: הקשר סמנטי עם AlephBERT ---
            # מסננים לבדיקה רק מילים ש-Whisper לא היה בטוח בהן ב-100% (מתחת ל-0.96)
            if len(clean_word) > 2 and acoustic_prob < 0.96:
                masked_text = full_text.replace(original_word, "[MASK]")
                
                # מבקשים 100 אפשרויות כדי לכסות את כל ההסתברויות ההגיוניות
                predictions = bert_masker(masked_text, top_k=100)
                
                # ניקוי הטוקנים של BERT מסימנים מיוחדים כמו ##
                top_predicted_words = [pred['token_str'].replace("##", "").strip() for pred in predictions]
                
                # בדיקת התאמה חכמה שפותרת את בעיית הקידומות והטוקנים
                match_found = False
                for pred_word in top_predicted_words:
                    # התאמה מוחלטת
                    if pred_word == clean_word:
                        match_found = True
                        break
                    # התאמה חלקית (למשל, BERT חזה "מערכת" והמילה היא "המערכת")
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
    # נתיב לקובץ האודיו שלך
    audio_file = "test1.mp3" # או wav, תלוי מה שמרת
    
    print("--- Starting MS2 Pipeline ---")
    
    # הפעלת שלב 1: קליטה אקוסטית (מחזיר את ה-JSON/מילון ישירות למשתנה)
    print(f"\n[Step 1] Running Acoustic Ingestion on '{audio_file}'...")
    try:
        step1_output = ingest_audio(audio_file)
        print("[Step 1] Finished successfully.")
    except Exception as e:
        print(f"Error in Step 1: {e}")
        exit()
        
    # הפעלת שלב 2: העברת הנתונים החיים (step1_output) ישירות לסינון
    print("\n[Step 2] Running SpanDetector Analysis...")
    detected_errors = analyze_spans(step1_output)
    
    # הדפסת התוצאות הסופיות
    print("\n--- Detected Errors ---")
    print(json.dumps(detected_errors, indent=4, ensure_ascii=False))