// This file implements greedy search and layer search algorithms.


using System;
using System.Collections.Generic;
using Hnsw.Core.Distance;
using Hnsw.Core.Graph;
using Hnsw.Core.Models;

namespace Hnsw.Core.Search;

public sealed class HnswSearch
{
    private readonly HnswGraph _graph;
    private readonly IVectorDistance _distance;

    public HnswSearch(HnswGraph graph, IVectorDistance distance)
    {
        _graph = graph;
        _distance = distance;
    }

    public HnswNode GreedySearch(float[] query, HnswNode entryPoint, int level)
    {
        HnswNode current = entryPoint;
        float currentDistance = _distance.Calculate(query, current.Vector);
        bool improved = true;

        while (improved)
        {
            improved = false;
            foreach (int neighborId in _graph.GetNeighbors(current.Id, level))
            {
                HnswNode neighbor = _graph.GetNode(neighborId);
                float neighborDistance = _distance.Calculate(query, neighbor.Vector);

                if (neighborDistance < currentDistance)
                {
                    current = neighbor;
                    currentDistance = neighborDistance;
                    improved = true;
                }
            }
        }

        return current;
    }

    public List<SearchResult> SearchLayer(float[] query, HnswNode entryPoint, int ef, int level)
    {
        if (ef < 1)
            throw new ArgumentOutOfRangeException(nameof(ef));

        var candidates = new List<SearchResult>();
        var best = new List<SearchResult>();
        var visited = new HashSet<int>();

        float entryDistance = _distance.Calculate(query, entryPoint.Vector);
        var entryResult = new SearchResult(entryPoint.Id, entryDistance);

        candidates.Add(entryResult);
        best.Add(entryResult);
        visited.Add(entryPoint.Id);

        while (candidates.Count > 0)
        {
            candidates.Sort((a, b) => a.Distance.CompareTo(b.Distance));
            SearchResult currentResult = candidates[0];
            candidates.RemoveAt(0);

            if (best.Count >= ef && currentResult.Distance > best[^1].Distance)
            {
                break;
            }

            HnswNode current = _graph.GetNode(currentResult.Id);

            foreach (int neighborId in _graph.GetNeighbors(current.Id, level))
            {
                if (!visited.Add(neighborId))
                    continue;

                HnswNode neighbor = _graph.GetNode(neighborId);
                float distance = _distance.Calculate(query, neighbor.Vector);

                if (best.Count < ef || distance < best[^1].Distance)
                {
                    var result = new SearchResult(neighborId, distance);
                    candidates.Add(result);
                    best.Add(result);

                    best.Sort((a, b) => a.Distance.CompareTo(b.Distance));

                    if (best.Count > ef)
                        best.RemoveAt(best.Count - 1);
                }
            }
        }

        return best;
    }
}