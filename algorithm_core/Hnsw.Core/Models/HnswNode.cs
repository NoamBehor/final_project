// This file represents a single node in the HNSW graph containing its vector and connections.


using System;
using System.Collections.Generic;
using System.Linq;

namespace Hnsw.Core.Models;

public sealed class HnswNode
{
    public int Id { get; }
    public float[] Vector { get; }
    public List<List<int>> Connections { get; }
    public int Level => Connections.Count - 1;

    public HnswNode(int id, float[] vector, int level)
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty.");

        if (level < 0)
            throw new ArgumentOutOfRangeException(nameof(level));

        Id = id;
        Vector = vector.ToArray();
        Connections = new List<List<int>>();

        for (int i = 0; i <= level; i++)
        {
            Connections.Add(new List<int>());
        }
    }
}