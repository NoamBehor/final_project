import wave
import av
import numpy as np


def _decode_audio_mono_16k(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    chunks = []
    with av.open(audio_path) as container:
        for frame in container.decode(audio=0):
            for resampled_frame in resampler.resample(frame):
                chunks.append(resampled_frame.to_ndarray().flatten())
    if not chunks:
        return np.array([], dtype=np.int16)
    return np.concatenate(chunks)


def crop_audio_for_review(audio_path: str, start_sec: float, end_sec: float, output_path: str, buffer_sec: float = 1.0) -> None:
    sample_rate = 16000
    print(f"Loading original audio: {audio_path}...")

    audio_int16 = _decode_audio_mono_16k(audio_path, sample_rate=sample_rate)
    if audio_int16.size == 0:
        print(f"[Warning] Could not decode audio frames from '{audio_path}'.")
        return

    start_idx = max(0, int((start_sec - buffer_sec) * sample_rate))
    end_idx = min(len(audio_int16), int((end_sec + buffer_sec) * sample_rate))
    snippet = audio_int16[start_idx:end_idx]

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(snippet.tobytes())

    print(f"[AUDIO CROPPED]: Saved snippet for human review at '{output_path}' ({len(snippet) / sample_rate:.2f}s)")