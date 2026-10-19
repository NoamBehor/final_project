using System;
using System.Collections.Generic;
using System.Linq;

namespace HnswExample
{
    // Represents one item stored inside the HNSW graph.
    public class HnswNode
    {
        public int Id { get; }
        public float[] Vector { get; }

        // Each list is one HNSW layer.
        // Example:
        // Connections[0] = neighbors in layer 0
        // Connections[1] = neighbors in layer 1
        public List<List<int>> Connections { get; }

        public HnswNode(int id, float[] vector, int level)
        {
            Id = id;
            Vector = vector;

            Connections = new List<List<int>>();

            for (int i = 0; i <= level; i++)
            {
                Connections.Add(new List<int>());
            }
        }
    }

    public class HnswIndex
    {
        // Maximum number of neighbors for normal layers.
        private readonly int M;

        // Maximum number of neighbors during index construction.
        private readonly int MaxM0;

        // Controls how many candidates we check while inserting.
        private readonly int EfConstruction;

        // Controls how many candidates we check during search.
        private readonly int EfSearch;

        // Used to create random levels.
        private readonly Random Random;

        // All nodes stored in the graph.
        private readonly Dictionary<int, HnswNode> Nodes;

        // The node from which we start searching.
        private int EntryPointId = -1;

        // Highest layer currently existing in the graph.
        private int MaxLevel = -1;

        public HnswIndex(
            int m = 16,
            int efConstruction = 200,
            int efSearch = 50)
        {
            if (m <= 0)
                throw new ArgumentException("M must be greater than 0.");

            if (efConstruction <= 0)
                throw new ArgumentException("efConstruction must be greater than 0.");

            if (efSearch <= 0)
                throw new ArgumentException("efSearch must be greater than 0.");

            M = m;

            // Layer 0 normally has more connections.
            MaxM0 = m * 2;

            EfConstruction = efConstruction;
            EfSearch = efSearch;

            Random = new Random();
            Nodes = new Dictionary<int, HnswNode>();
        }

        // ---------------------------------------------------------
        // INSERT
        // ---------------------------------------------------------

        public void Insert(int id, float[] vector)
        {
            if (vector == null || vector.Length == 0)
                throw new ArgumentException("Vector cannot be empty.");

            if (Nodes.ContainsKey(id))
                throw new ArgumentException($"Node with id {id} already exists.");

            // Create a random level for the new node.
            int newLevel = RandomLevel();

            HnswNode newNode = new HnswNode(
                id,
                Normalize(vector),
                newLevel
            );

            Nodes.Add(id, newNode);

            // First node becomes the entry point.
            if (EntryPointId == -1)
            {
                EntryPointId = id;
                MaxLevel = newLevel;
                return;
            }

            HnswNode currentNode = Nodes[EntryPointId];

            // -----------------------------------------------------
            // STEP 1:
            // Search from the highest layer down to newLevel + 1.
            //
            // At these layers we only need the closest single node.
            // -----------------------------------------------------

            for (int level = MaxLevel; level > newLevel; level--)
            {
                currentNode = GreedySearch(
                    newNode.Vector,
                    currentNode,
                    level
                );
            }

            // -----------------------------------------------------
            // STEP 2:
            // Insert into each layer from newLevel down to layer 0.
            // -----------------------------------------------------

            int maxInsertLevel = Math.Min(newLevel, MaxLevel);

            for (int level = maxInsertLevel; level >= 0; level--)
            {
                List<(int id, float distance)> candidates =
                    SearchLayer(
                        newNode.Vector,
                        currentNode,
                        EfConstruction,
                        level
                    );

                int maxNeighbors = level == 0 ? MaxM0 : M;

                List<int> selectedNeighbors =
                    SelectNeighbors(candidates, maxNeighbors);

                // Add connections from the new node to its neighbors.
                foreach (int neighborId in selectedNeighbors)
                {
                    AddConnection(newNode.Id, neighborId, level);

                    // Also connect the neighbor back to the new node.
                    AddConnection(neighborId, newNode.Id, level);
                }

                // The closest result becomes the entry point for
                // the next lower layer.
                if (candidates.Count > 0)
                {
                    currentNode = Nodes[candidates[0].id];
                }
            }

            // -----------------------------------------------------
            // STEP 3:
            // If the new node has a higher level than the old graph,
            // it becomes the new entry point.
            // -----------------------------------------------------

            if (newLevel > MaxLevel)
            {
                EntryPointId = id;
                MaxLevel = newLevel;
            }
        }

        // ---------------------------------------------------------
        // SEARCH
        // ---------------------------------------------------------

        public List<(int id, float distance)> Search(
            float[] queryVector,
            int k)
        {
            if (queryVector == null || queryVector.Length == 0)
                throw new ArgumentException("Query vector cannot be empty.");

            if (k <= 0)
                throw new ArgumentException("K must be greater than 0.");

            if (EntryPointId == -1)
                return new List<(int id, float distance)>();

            float[] query = Normalize(queryVector);

            HnswNode currentNode = Nodes[EntryPointId];

            // -----------------------------------------------------
            // STEP 1:
            // Start at the highest layer.
            // Move greedily until we reach layer 0.
            // -----------------------------------------------------

            for (int level = MaxLevel; level > 0; level--)
            {
                currentNode = GreedySearch(
                    query,
                    currentNode,
                    level
                );
            }

            // -----------------------------------------------------
            // STEP 2:
            // Search layer 0 using efSearch candidates.
            // -----------------------------------------------------

            List<(int id, float distance)> results =
                SearchLayer(
                    query,
                    currentNode,
                    Math.Max(EfSearch, k),
                    0
                );

            // Return only the K closest items.
            return results
                .OrderBy(x => x.distance)
                .Take(k)
                .ToList();
        }

        // ---------------------------------------------------------
        // GREEDY SEARCH
        // ---------------------------------------------------------

        private HnswNode GreedySearch(
            float[] query,
            HnswNode entryPoint,
            int level)
        {
            HnswNode current = entryPoint;

            float currentDistance =
                CosineDistance(query, current.Vector);

            bool changed = true;

            while (changed)
            {
                changed = false;

                foreach (int neighborId in current.Connections[level])
                {
                    HnswNode neighbor = Nodes[neighborId];

                    float neighborDistance =
                        CosineDistance(query, neighbor.Vector);

                    // If neighbor is closer, move to it.
                    if (neighborDistance < currentDistance)
                    {
                        current = neighbor;
                        currentDistance = neighborDistance;
                        changed = true;
                    }
                }
            }

            return current;
        }

        // ---------------------------------------------------------
        // SEARCH LAYER
        // ---------------------------------------------------------

        private List<(int id, float distance)> SearchLayer(
            float[] query,
            HnswNode entryPoint,
            int ef,
            int level)
        {
            // Candidate nodes still waiting to be processed.
            //
            // Smaller distance = higher priority.
            PriorityQueue<int, float> candidates =
                new PriorityQueue<int, float>();

            // Best nodes found so far.
            //
            // We use negative distance because PriorityQueue is
            // a MIN heap. Negative distance makes it behave
            // like a MAX heap for the worst result.
            PriorityQueue<int, float> results =
                new PriorityQueue<int, float>();

            HashSet<int> visited = new HashSet<int>();

            float entryDistance =
                CosineDistance(query, entryPoint.Vector);

            candidates.Enqueue(entryPoint.Id, entryDistance);

            // Negative distance.
            results.Enqueue(entryPoint.Id, -entryDistance);

            visited.Add(entryPoint.Id);

            while (candidates.Count > 0)
            {
                int currentId = candidates.Dequeue();

                HnswNode current = Nodes[currentId];

                float currentDistance =
                    CosineDistance(query, current.Vector);

                // Distance of the worst node currently stored.
                int worstResultId = results.Peek();

                float worstDistance =
                    CosineDistance(
                        query,
                        Nodes[worstResultId].Vector
                    );

                // If current node is already farther than the
                // worst result, no better candidates should be found
                // from this path.
                if (currentDistance > worstDistance &&
                    results.Count >= ef)
                {
                    break;
                }

                foreach (int neighborId in current.Connections[level])
                {
                    if (visited.Contains(neighborId))
                        continue;

                    visited.Add(neighborId);

                    HnswNode neighbor = Nodes[neighborId];

                    float neighborDistance =
                        CosineDistance(query, neighbor.Vector);

                    int currentWorstId = results.Peek();

                    float currentWorstDistance =
                        CosineDistance(
                            query,
                            Nodes[currentWorstId].Vector
                        );

                    // Add this neighbor if:
                    // 1. We still have room
                    // OR
                    // 2. It is better than the worst current result.
                    if (results.Count < ef ||
                        neighborDistance < currentWorstDistance)
                    {
                        candidates.Enqueue(
                            neighborId,
                            neighborDistance
                        );

                        // Negative distance makes the farthest
                        // result appear at the top of the heap.
                        results.Enqueue(
                            neighborId,
                            -neighborDistance
                        );

                        // Keep only the best EF results.
                        if (results.Count > ef)
                        {
                            results.Dequeue();
                        }
                    }
                }
            }

            // Convert the result heap into a normal list.
            List<(int id, float distance)> finalResults =
                new List<(int id, float distance)>();

            while (results.Count > 0)
            {
                int id = results.Dequeue();

                finalResults.Add(
                    (
                        id,
                        CosineDistance(
                            query,
                            Nodes[id].Vector
                        )
                    )
                );
            }

            // Closest first.
            finalResults.Sort(
                (a, b) => a.distance.CompareTo(b.distance)
            );

            return finalResults;
        }

        // ---------------------------------------------------------
        // ADD CONNECTION
        // ---------------------------------------------------------

        private void AddConnection(
            int sourceId,
            int targetId,
            int level)
        {
            HnswNode source = Nodes[sourceId];

            // Make sure the layer exists.
            if (level >= source.Connections.Count)
                return;

            List<int> neighbors =
                source.Connections[level];

            // Do not add duplicates.
            if (neighbors.Contains(targetId))
                return;

            neighbors.Add(targetId);

            int maxNeighbors =
                level == 0 ? MaxM0 : M;

            // If there are too many neighbors, keep only the closest.
            if (neighbors.Count > maxNeighbors)
            {
                neighbors.Sort(
                    (a, b) =>
                    {
                        float da =
                            CosineDistance(
                                source.Vector,
                                Nodes[a].Vector
                            );

                        float db =
                            CosineDistance(
                                source.Vector,
                                Nodes[b].Vector
                            );

                        return da.CompareTo(db);
                    }
                );

                while (neighbors.Count > maxNeighbors)
                {
                    neighbors.RemoveAt(neighbors.Count - 1);
                }
            }
        }

        // ---------------------------------------------------------
        // SELECT NEIGHBORS
        // ---------------------------------------------------------

        private List<int> SelectNeighbors(
            List<(int id, float distance)> candidates,
            int maxNeighbors)
        {
            // Simple version:
            // Just select the closest nodes.
            //
            // A more advanced HNSW implementation uses a
            // diversity heuristic here.

            return candidates
                .OrderBy(x => x.distance)
                .Take(maxNeighbors)
                .Select(x => x.id)
                .ToList();
        }

        // ---------------------------------------------------------
        // RANDOM LEVEL
        // ---------------------------------------------------------

        private int RandomLevel()
        {
            int level = 0;

            // Higher levels should be much less common.
            //
            // Example:
            // many nodes -> level 0
            // fewer nodes -> level 1
            // even fewer -> level 2
            // etc.

            while (Random.NextDouble() < 0.5 &&
                   level < 32)
            {
                level++;
            }

            return level;
        }

        // ---------------------------------------------------------
        // VECTOR NORMALIZATION
        // ---------------------------------------------------------

        private float[] Normalize(float[] vector)
        {
            double length = 0;

            for (int i = 0; i < vector.Length; i++)
            {
                length += vector[i] * vector[i];
            }

            length = Math.Sqrt(length);

            if (length == 0)
                return vector.ToArray();

            float[] normalized =
                new float[vector.Length];

            for (int i = 0; i < vector.Length; i++)
            {
                normalized[i] =
                    (float)(vector[i] / length);
            }

            return normalized;
        }

        // ---------------------------------------------------------
        // COSINE DISTANCE
        // ---------------------------------------------------------

        private float CosineDistance(
            float[] a,
            float[] b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException(
                    "Vectors must have the same size."
                );

            float dot = 0;

            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
            }

            // Because vectors are normalized:
            //
            // cosine similarity = dot product
            //
            // distance = 1 - similarity

            return 1.0f - dot;
        }

        // ---------------------------------------------------------
        // DEBUG INFORMATION
        // ---------------------------------------------------------

        public int Count
        {
            get { return Nodes.Count; }
        }

        public int HighestLevel
        {
            get { return MaxLevel; }
        }
    }

    // -------------------------------------------------------------
    // EXAMPLE
    // -------------------------------------------------------------

    public static class Program
    {
        public static void Main()
        {
            HnswIndex index = new HnswIndex(
                m: 16,
                efConstruction: 200,
                efSearch: 50
            );

            // Add some vectors.
            index.Insert(
                1,
                new float[] { 0.10f, 0.20f, 0.30f }
            );

            index.Insert(
                2,
                new float[] { 0.11f, 0.21f, 0.29f }
            );

            index.Insert(
                3,
                new float[] { 0.80f, 0.70f, 0.60f }
            );

            index.Insert(
                4,
                new float[] { 0.12f, 0.19f, 0.31f }
            );

            index.Insert(
                5,
                new float[] { 0.75f, 0.72f, 0.65f }
            );

            // Query vector.
            float[] query =
            {
                0.10f,
                0.20f,
                0.31f
            };

            // Search for the 3 closest vectors.
            List<(int id, float distance)> results =
                index.Search(query, 3);

            Console.WriteLine("Nearest neighbors:");

            foreach (var result in results)
            {
                Console.WriteLine(
                    $"ID: {result.id}, Distance: {result.distance:F4}"
                );
            }

            Console.WriteLine();
            Console.WriteLine($"Nodes: {index.Count}");
            Console.WriteLine($"Highest level: {index.HighestLevel}");
        }
    }
}