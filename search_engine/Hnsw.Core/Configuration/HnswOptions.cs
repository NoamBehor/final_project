// This file stores the configuration options for the HNSW index like M and Ef.

using System;

namespace Hnsw.Core.Configuration;

public sealed class HnswOptions
{
    public int M { get; init; }
    public int EfConstruction { get; init; }
    public int EfSearch { get; init; }
    public int MaxLevel { get; init; }
    public int? RandomSeed { get; init; }

    public HnswOptions(
        int m = 16,
        int efConstruction = 200,
        int efSearch = 50,
        int maxLevel = 16,
        int? randomSeed = null)
    {
        M = m;
        EfConstruction = efConstruction;
        EfSearch = efSearch;
        MaxLevel = maxLevel;
        RandomSeed = randomSeed;
    }

    public void Validate()
    {
        if (M < 2)
            throw new ArgumentOutOfRangeException(nameof(M), "M must be at least 2.");

        if (EfConstruction < M)
            throw new ArgumentOutOfRangeException(nameof(EfConstruction), "EfConstruction must be greater than or equal to M.");

        if (EfSearch < 1)
            throw new ArgumentOutOfRangeException(nameof(EfSearch), "EfSearch must be positive.");

        if (MaxLevel < 0)
            throw new ArgumentOutOfRangeException(nameof(MaxLevel), "MaxLevel cannot be negative.");
    }
}