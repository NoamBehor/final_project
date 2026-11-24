using System;
using System.Text.Json.Serialization;

namespace RadioNLP.Shared
{
    public class TranscriptionDTO
    {
        [JsonPropertyName("message_id")]
        public string MessageId { get; set; }

        [JsonPropertyName("timestamp")]
        public DateTime Timestamp { get; set; }

        [JsonPropertyName("frequency")]
        public string Frequency { get; set; }

        [JsonPropertyName("audio_file_path")]
        public string AudioFilePath { get; set; }

        [JsonPropertyName("original_text")]
        public string OriginalText { get; set; }

        [JsonPropertyName("corrected_text")]
        public string CorrectedText { get; set; }

        [JsonPropertyName("is_corrected")]
        public bool IsCorrected { get; set; }
    }
}



