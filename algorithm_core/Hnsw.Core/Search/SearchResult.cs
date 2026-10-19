// This file represents the search result containing an id and distance.


namespace Hnsw.Core.Search;

public sealed record SearchResult(int Id, float Distance);