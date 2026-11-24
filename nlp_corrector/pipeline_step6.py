import wave
import numpy as np
from faster_whisper.audio import decode_audio

def crop_audio_for_review(audio_path, start_sec, end_sec, output_path, buffer_sec=1.0):
    """
    שלב 6: Human-in-the-Loop
    שימוש במפענח האודיו הפנימי של Whisper כדי לגזור קטע מדויק, 
    ללא צורך בספריות חיצוניות בעייתיות כמו pydub.
    """
    print(f"Loading original audio: {audio_path}...")
    
    try:
        # קריאת האודיו בעזרת המנוע של Whisper (מחזיר מערך נתונים בקצב של 16kHz)
        audio_array = decode_audio(audio_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
        
    sample_rate = 16000  # Whisper עובד תמיד ב-16,000Hz
    
    # המרת זמנים (שניות) לאינדקסים בתוך המערך המתמטי
    start_sample = max(0, int((start_sec - buffer_sec) * sample_rate))
    end_sample = min(len(audio_array), int((end_sec + buffer_sec) * sample_rate))
    
    # חיתוך הקטע הספציפי
    snippet = audio_array[start_sample:end_sample]
    
    # המרה מפורמט עשרוני לפורמט השלם (int16) שנדרש לקובצי שמע רגילים
    snippet_int16 = (snippet * 32767).astype(np.int16)
    
    # כתיבת קובץ ה-WAV החדש באמצעות ספריית wave המובנית בפייתון
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1) # מונו
        wav_file.setsampwidth(2) # 2 בתים לדגימה
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(snippet_int16.tobytes())
        
    print(f"[✂️ AUDIO CROPPED]: Saved snippet for human review at '{output_path}'")
    print(f"Snippet length: {len(snippet) / sample_rate:.2f} seconds")

if __name__ == "__main__":
    original_audio = "test1.mp3"  
    snippet_file = "review_barir.wav"
    
    print("\n--- MS2 System: ABSTAIN Triggered ---")
    print("Word 'בריר' requires human review. Cropping audio...")
    
    crop_audio_for_review(
        audio_path=original_audio, 
        start_sec=6.36, 
        end_sec=6.66, 
        output_path=snippet_file,
        buffer_sec=1.5  # הוספת שוליים של שניה וחצי מכל כיוון
    )