from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class TranscriptionDTO(BaseModel):
    message_id: str
    timestamp: datetime
    frequency: str
    audio_file_path: str
    original_text: str
    corrected_text: Optional[str] = None
    is_corrected: bool = False