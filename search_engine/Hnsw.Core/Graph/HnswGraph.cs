// This file manages all nodes and their connections across graph layers.

using System;
using System.Collections.Generic;
using System.Linq;
using Hnsw.Core.Distance;
using Hnsw.Core.Models;

namespace Hnsw.Core.Graph;

public sealed class HnswGraph
{
    private readonly Dictionary<int, HnswNode> _nodes = new();
    private readonly IVectorDistance _distance;
    private readonly int _m;
    private readonly int _maxM0;

    public HnswGraph(IVectorDistance distance, int m)
    {
        _distance = distance;
        _m = m;
        _maxM0 = 2 * m;
    }

    public int Count => _nodes.Count;
    public bool Contains(int id) => _nodes.ContainsKey(id);
    public HnswNode GetNode(int id) => _nodes[id];
    public IEnumerable<HnswNode> GetAllNodes() => _nodes.Values;

    public void AddNode(HnswNode node)
    {
        if (!_nodes.TryAdd(node.Id, node))
        {
            throw new ArgumentException($"Node {node.Id} already exists.");
        }

        if (_nodes.Count > 1)
        {
            int expectedDimension = _nodes.Values.First().Vector.Length;
            if (node.Vector.Length != expectedDimension)
            {
                _nodes.Remove(node.Id);
                throw new ArgumentException("All vectors must have the same dimension.");
            }
        }
    }

    public IReadOnlyList<int> GetNeighbors(int id, int level)
    {
        HnswNode node = GetNode(id);
        if (level < 0 || level >= node.Connections.Count)
            return Array.Empty<int>();

        return node.Connections[level];
    }

    public void ConnectBidirectionally(int firstId, int secondId, int level)
    {
        if (firstId == secondId)
            return;

        AddOneWayConnection(firstId, secondId, level);
        AddOneWayConnection(secondId, firstId, level);
    }

    private void AddOneWayConnection(int sourceId, int targetId, int level)
    {
        HnswNode source = GetNode(sourceId);
        HnswNode target = GetNode(targetId);

        if (level > source.Level || level > target.Level)
            return;

        List<int> neighbors = source.Connections[level];
        if (neighbors.Contains(targetId))
            return;

        neighbors.Add(targetId);
        int maxNeighbors = level == 0 ? _maxM0 : _m;

        if (neighbors.Count > maxNeighbors)
        {
            List<int> closest = neighbors
                .OrderBy(id => _distance.Calculate(source.Vector, GetNode(id).Vector))
                .Take(maxNeighbors)
                .ToList();

            neighbors.Clear();
            neighbors.AddRange(closest);
        }
    }
}