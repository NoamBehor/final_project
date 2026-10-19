// This file serves as the main facade for the HNSW index handling insert and search operations.


using System;
using System.Collections.Generic;
using System.Linq;
using Hnsw.Core.Configuration;
using Hnsw.Core.Distance;
using Hnsw.Core.Graph;
using Hnsw.Core.Models;
using Hnsw.Core.Search;

namespace Hnsw.Core;

public sealed class HnswIndex
{
    private readonly HnswOptions _options;
    private readonly IVectorDistance _distance;
    private readonly HnswGraph _graph;
    private readonly HnswSearch _search;
    private readonly NeighborSelector _selector;
    private readonly Random _random;

    private int? _entryPointId;
    private int _maxLevel = -1;
    private int? _vectorDimension;

    public int Count => _graph.Count;
    public int HighestLevel => _maxLevel;

    public HnswIndex(HnswOptions? options = null, IVectorDistance? distance = null)
    {
        _options = options ?? new HnswOptions();
        _options.Validate();

        _distance = distance ?? new CosineDistance();
        _graph = new HnswGraph(_distance, _options.M);
        _search = new HnswSearch(_graph, _distance);
        _selector = new NeighborSelector();

        _random = _options.RandomSeed.HasValue
            ? new Random(_options.RandomSeed.Value)
            : new Random();
    }

    public void Insert(int id, float[] vector)
    {
        ArgumentNullException.ThrowIfNull(vector);

        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty.");

        if (_graph.Contains(id))
            throw new ArgumentException($"Node {id} already exists.");

        if (_vectorDimension.HasValue && vector.Length != _vectorDimension.Value)
            throw new ArgumentException("All vectors must have the same dimension.");

        int level = RandomLevel();
        var node = new HnswNode(id, vector, level);

        if (_entryPointId is null)
        {
            _graph.AddNode(node);
            _vectorDimension = vector.Length;
            _entryPointId = id;
            _maxLevel = level;
            return;
        }

        HnswNode current = _graph.GetNode(_entryPointId.Value);
        _graph.AddNode(node);
        _vectorDimension ??= vector.Length;

        for (int layer = _maxLevel; layer > level; layer--)
        {
            current = _search.GreedySearch(node.Vector, current, layer);
        }

        int insertionTopLevel = Math.Min(level, _maxLevel);

        for (int layer = insertionTopLevel; layer >= 0; layer--)
        {
            List<SearchResult> candidates = _search.SearchLayer(node.Vector, current, _options.EfConstruction, layer);
            int maxNeighbors = layer == 0 ? 2 * _options.M : _options.M;

            List<int> selected = _selector.Select(
                candidates.Select(c => (c.Id, c.Distance)),
                maxNeighbors);

            foreach (int neighborId in selected)
            {
                _graph.ConnectBidirectionally(node.Id, neighborId, layer);
            }

            if (candidates.Count > 0)
            {
                current = _graph.GetNode(candidates[0].Id);
            }
        }

        if (level > _maxLevel)
        {
            _entryPointId = id;
            _maxLevel = level;
        }
    }

    public List<SearchResult> Search(float[] query, int k)
    {
        ArgumentNullException.ThrowIfNull(query);

        if (query.Length == 0)
            throw new ArgumentException("Query cannot be empty.");

        if (k <= 0)
            throw new ArgumentOutOfRangeException(nameof(k));

        if (_vectorDimension.HasValue && query.Length != _vectorDimension.Value)
            throw new ArgumentException("Query dimension does not match the index.");

        if (_entryPointId is null)
            return new List<SearchResult>();

        HnswNode current = _graph.GetNode(_entryPointId.Value);

        for (int layer = _maxLevel; layer > 0; layer--)
        {
            current = _search.GreedySearch(query, current, layer);
        }

        return _search.SearchLayer(query, current, Math.Max(_options.EfSearch, k), 0)
            .Take(k)
            .ToList();
    }

    private int RandomLevel()
    {
        int level = 0;
        double probability = 1.0 / _options.M;

        while (level < _options.MaxLevel && _random.NextDouble() < probability)
        {
            level++;
        }

        return level;
    }
}