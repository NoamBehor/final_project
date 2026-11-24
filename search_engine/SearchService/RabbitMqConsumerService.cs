using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Npgsql;
using RabbitMQ.Client;
using RabbitMQ.Client.Events;

namespace SearchService;

public class TranscriptionDto
{
    [JsonPropertyName("message_id")]
    public string MessageId { get; set; } = string.Empty;

    [JsonPropertyName("timestamp")]
    public DateTime Timestamp { get; set; }

    [JsonPropertyName("frequency")]
    public string Frequency { get; set; } = string.Empty;

    [JsonPropertyName("audio_file_path")]
    public string AudioFilePath { get; set; } = string.Empty;

    [JsonPropertyName("original_text")]
    public string OriginalText { get; set; } = string.Empty;

    [JsonPropertyName("corrected_text")]
    public string? CorrectedText { get; set; }

    [JsonPropertyName("is_corrected")]
    public bool IsCorrected { get; set; }
}

public class RabbitMqConsumerService : BackgroundService
{
    private const string PgConnString = "Host=127.0.0.1;Port=5433;Username=admin;Password=admin123;Database=radio_nlp_db;";
    private const string QueueName = "nlp_to_search";
    private IConnection? _connection;
    private IModel? _channel;
    private readonly HybridSearchManager _searchManager;

public RabbitMqConsumerService(HybridSearchManager searchManager)
{
    _searchManager = searchManager;
}

    protected override Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var factory = new ConnectionFactory
        {
            HostName = "127.0.0.1",
            Port = 5672,
            UserName = "admin",
            Password = "admin123",
            AutomaticRecoveryEnabled = true
        };

        _connection = factory.CreateConnection();
        _channel = _connection.CreateModel();

        _channel.QueueDeclare(queue: QueueName, durable: true, exclusive: false, autoDelete: false, arguments: null);
        _channel.BasicQos(prefetchSize: 0, prefetchCount: 1, global: false);

        var consumer = new EventingBasicConsumer(_channel);
        consumer.Received += async (_, ea) =>
        {
            try
            {
                var json = Encoding.UTF8.GetString(ea.Body.ToArray());
                var dto = JsonSerializer.Deserialize<TranscriptionDto>(json);

                if (dto != null)
                {
                    await SaveToPostgresAsync(dto, stoppingToken);
                    _searchManager.IndexDocument(dto);
                    Console.WriteLine($"[C# SearchService] Saved DTO {dto.MessageId} ('{dto.CorrectedText}') to PostgreSQL.");
                }

                _channel.BasicAck(deliveryTag: ea.DeliveryTag, multiple: false);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[C# SearchService] Error processing message: {ex.Message}");
                _channel.BasicNack(deliveryTag: ea.DeliveryTag, multiple: false, requeue: true);
            }
        };

        _channel.BasicConsume(queue: QueueName, autoAck: false, consumer: consumer);
        Console.WriteLine("[C# SearchService] Listening on 'nlp_to_search'...");

        return Task.CompletedTask;
    }

    private static async Task SaveToPostgresAsync(TranscriptionDto dto, CancellationToken ct)
    {
        byte[] audioBytes = File.Exists(dto.AudioFilePath)
            ? await File.ReadAllBytesAsync(dto.AudioFilePath, ct)
            : Array.Empty<byte>();

        await using var conn = new NpgsqlConnection(PgConnString);
        await conn.OpenAsync(ct);

        const string sql = @"
            INSERT INTO transcriptions (message_id, frequency, original_text, corrected_text, audio_file, created_at)
            VALUES (@id, @freq, @orig, @corr, @audio, @created)
            ON CONFLICT (message_id) DO NOTHING;";

        await using var cmd = new NpgsqlCommand(sql, conn);
        cmd.Parameters.AddWithValue("id", Guid.Parse(dto.MessageId));
        cmd.Parameters.AddWithValue("freq", dto.Frequency);
        cmd.Parameters.AddWithValue("orig", dto.OriginalText);
        cmd.Parameters.AddWithValue("corr", (object?)dto.CorrectedText ?? DBNull.Value);
        cmd.Parameters.AddWithValue("audio", audioBytes);
        cmd.Parameters.AddWithValue("created", dto.Timestamp.ToUniversalTime());

        await cmd.ExecuteNonQueryAsync(ct);
    }

    public override void Dispose()
    {
        _channel?.Close();
        _connection?.Close();
        base.Dispose();
    }
}