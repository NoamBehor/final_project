// This file selects the best neighbor candidates based on distance.


using System.Collections.Generic;
using System.Linq;

namespace Hnsw.Core.Graph;

public sealed class NeighborSelector
{
    public List<int> Select(IEnumerable<(int Id, float Distance)> candidates, int maxNeighbors)
    {
        if (maxNeighbors <= 0)
            return new List<int>();

        return candidates
            .OrderBy(candidate => candidate.Distance)
            .Select(candidate => candidate.Id)
            .Distinct()
            .Take(maxNeighbors)
            .ToList();
    }
}