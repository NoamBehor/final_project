using System.Collections.Concurrent;
using Hnsw.Core;
using Npgsql;
using Search.Engine.Core;

namespace SearchService;

public record HybridSearchHit(
    string MessageId,
    string Frequency,
    string CorrectedText,
    DateTime CreatedAt,
    double HybridScore,
    double Bm25Score,
    float? VectorDistance
);

public class HybridSearchManager
{
    private const string PgConnString = "Host=127.0.0.1;Port=5433;Username=admin;Password=admin123;Database=radio_nlp_db;";
    private const int VectorDim = 768;

    private readonly BM25 _bm25;
    private readonly HnswIndex _hnsw;
    private readonly object _indexLock = new();

    private readonly ConcurrentDictionary<string, TranscriptionDto> _docs = new();
    private readonly Dictionary<int, string> _intToMessageId = new();
    private readonly Dictionary<string, int> _messageIdToInt = new();
    private int _nextNodeId = 1;

    public HybridSearchManager()
    {
        string stopwordsPath = Path.Combine(AppContext.BaseDirectory, "hebrew_stopwords.txt");
        _bm25 = new BM25(stopwordsPath);
        _hnsw = new HnswIndex();
    }

    public async Task InitializeFromDatabaseAsync(CancellationToken ct = default)
    {
        await using var conn = new NpgsqlConnection(PgConnString);
        await conn.OpenAsync(ct);

        const string sql = "SELECT message_id, frequency, original_text, corrected_text, created_at, embedding FROM transcriptions ORDER BY created_at ASC;";
        await using var cmd = new NpgsqlCommand(sql, conn);
        await using var reader = await cmd.ExecuteReaderAsync(ct);

        int loaded = 0;
        while (await reader.ReadAsync(ct))
        {
            var dto = new TranscriptionDto
            {
                MessageId = reader.GetGuid(0).ToString(),
                Frequency = reader.GetString(1),
                OriginalText = reader.GetString(2),
                CorrectedText = reader.IsDBNull(3) ? null : reader.GetString(3),
                Timestamp = reader.GetDateTime(4),
                IsCorrected = !reader.IsDBNull(3),
                Embedding = reader.IsDBNull(5) ? Array.Empty<float>() : reader.GetFieldValue<float[]>(5)
            };

            IndexDocument(dto);
            loaded++;
        }

        Console.WriteLine($"[HybridSearch] Hydrated {loaded} documents from PostgreSQL into BM25 & HNSW (dim={VectorDim}).");
    }

    public void IndexDocument(TranscriptionDto dto)
    {
        string textToIndex = !string.IsNullOrWhiteSpace(dto.CorrectedText) ? dto.CorrectedText : dto.OriginalText;
        _docs[dto.MessageId] = dto;

        float[] embedding = dto.Embedding.Length == VectorDim
            ? dto.Embedding
            : ComputeFallbackEmbedding(textToIndex);

        lock (_indexLock)
        {
            _bm25.AddDocument(dto.MessageId, textToIndex);

            if (!_messageIdToInt.ContainsKey(dto.MessageId))
            {
                int nodeId = _nextNodeId++;
                _messageIdToInt[dto.MessageId] = nodeId;
                _intToMessageId[nodeId] = dto.MessageId;
                _hnsw.Insert(nodeId, embedding);
            }
        }
    }

    public List<HybridSearchHit> Search(string query, int topK = 5, float[]? queryEmbedding = null)
    {
        if (string.IsNullOrWhiteSpace(query) && (queryEmbedding == null || queryEmbedding.Length != VectorDim))
            return new List<HybridSearchHit>();

        float[] queryVector = (queryEmbedding != null && queryEmbedding.Length == VectorDim)
            ? queryEmbedding
            : ComputeFallbackEmbedding(query);

        List<Search.Engine.Core.SearchResult> bm25Results;
        List<Hnsw.Core.Search.SearchResult> hnswResults;

        lock (_indexLock)
        {
            bm25Results = _bm25.Search(query, topK * 2);
            hnswResults = _hnsw.Count > 0 ? _hnsw.Search(queryVector, Math.Min(topK * 2, _hnsw.Count)) : new();
        }

        const double rrfK = 60.0;
        var rrfScores = new Dictionary<string, double>();
        var bm25Map = bm25Results.ToDictionary(r => r.DocId, r => r.Score);
        var distMap = new Dictionary<string, float>();

        for (int i = 0; i < bm25Results.Count; i++)
        {
            string id = bm25Results[i].DocId;
            rrfScores[id] = rrfScores.GetValueOrDefault(id) + 1.0 / (rrfK + i + 1);
        }

        for (int i = 0; i < hnswResults.Count; i++)
        {
            if (_intToMessageId.TryGetValue(hnswResults[i].Id, out string? id))
            {
                distMap[id] = hnswResults[i].Distance;
                rrfScores[id] = rrfScores.GetValueOrDefault(id) + 1.0 / (rrfK + i + 1);
            }
        }

        return rrfScores
            .OrderByDescending(kv => kv.Value)
            .ThenBy(kv => distMap.GetValueOrDefault(kv.Key, float.MaxValue))
            .Take(topK)
            .Select(kv =>
            {
                var doc = _docs[kv.Key];
                return new HybridSearchHit(
                    MessageId: doc.MessageId,
                    Frequency: doc.Frequency,
                    CorrectedText: doc.CorrectedText ?? doc.OriginalText,
                    CreatedAt: doc.Timestamp,
                    HybridScore: Math.Round(kv.Value, 5),
                    Bm25Score: Math.Round(bm25Map.GetValueOrDefault(kv.Key, 0.0), 4),
                    VectorDistance: distMap.TryGetValue(kv.Key, out float d) ? MathF.Round(d, 4) : null
                );
            })
            .ToList();
    }

    private float[] ComputeFallbackEmbedding(string text)
    {
        var vec = new float[VectorDim];
        var tokens = _bm25.PreprocessHebrew(text);

        foreach (var token in tokens)
        {
            int tokenBucket = Math.Abs(token.GetHashCode()) % VectorDim;
            vec[tokenBucket] += 1.0f;

            for (int i = 0; i < token.Length - 2; i++)
            {
                string trigram = token.Substring(i, 3);
                int triBucket = Math.Abs(trigram.GetHashCode()) % VectorDim;
                vec[triBucket] += 0.5f;
            }
        }

        float norm = MathF.Sqrt(vec.Sum(x => x * x));
        if (norm > 0)
        {
            for (int i = 0; i < VectorDim; i++)
                vec[i] /= norm;
        }
        else
        {
            vec[0] = 1.0f;
        }

        return vec;
    }
}