using SearchService;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new() { Title = "SearchService", Version = "v1" });
    c.CustomSchemaIds(type => type.FullName);
});
builder.Services.AddSingleton<HybridSearchManager>();
builder.Services.AddHostedService<RabbitMqConsumerService>();

var app = builder.Build();

var searchManager = app.Services.GetRequiredService<HybridSearchManager>();
await searchManager.InitializeFromDatabaseAsync();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger(c =>
    {
        c.SerializeAsV2 = true;
    });
    app.UseSwaggerUI();
}

app.MapGet("/", () => Results.Redirect("/swagger")).ExcludeFromDescription();

app.MapGet("/api/search", (string q, int topK, HybridSearchManager manager) =>
{
    var results = manager.Search(q, topK > 0 ? topK : 5);
    return Results.Ok(results);
});

app.Run();