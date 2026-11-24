from faster_whisper import WhisperModel
import json

# טעינת המודל (בפעם הראשונה זה יוריד את קובצי המודל)
# אנחנו מגדירים ריצה על ה-CPU עם קוונטיזציה ל-int8 כדי שירוץ מהר ויעיל
print("Loading Whisper model...")
model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")

def ingest_audio(audio_path):
    print(f"Transcribing {audio_path}...")
    # הפעלת התמלול עם דרישה לחותמות זמן וקיבוע השפה לעברית
    segments, info = model.transcribe(audio_path, word_timestamps=True, language="he")
    
    pipeline_data = []
    
    for segment in segments:
        # סינון אקוסטי: דילוג על מקטעים של רעש רקע או חוסר דיבור
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
        
        # מעבר על כל מילה בתוך המקטע ושמירת הנתונים הקריטיים
        for word in segment.words:
            word_dict = {
                "word": word.word.strip(),
                "start": word.start,
                "end": word.end,
                "probability": word.probability
            }
            segment_dict["words"].append(word_dict)
            
        pipeline_data.append(segment_dict)
        
    return pipeline_data

# --- אזור בדיקה (טסט) ---
if __name__ == "__main__":
    # החלף את הנתיב לקובץ הקלטה קצר שיש לך על המחשב (mp3 או wav)
    test_audio_file = "test1.mp3" 
    
    try:
        results = ingest_audio(test_audio_file)
        
        # הדפסת התוצאה בפורמט JSON קריא כדי שנוכל לראות את המבנה
        print(json.dumps(results, indent=4, ensure_ascii=False))
        
    except Exception as e:
        print(f"Error: {e}")