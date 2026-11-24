from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any

class TranscriptionDTO(BaseModel):
    message_id: str
    timestamp: datetime
    frequency: str
    audio_file_path: str
    original_text: str
    corrected_text: Optional[str] = None
    is_corrected: bool = False
    segments: List[Dict[str, Any]] = Field(default_factory=list)
    embedding: List[float] = Field(default_factory=list)