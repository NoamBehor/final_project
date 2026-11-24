// This file calculates the cosine distance between vectors.


using System;

namespace Hnsw.Core.Distance;

public sealed class CosineDistance : IVectorDistance
{
    public float Calculate(float[] a, float[] b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vectors must have the same dimension.");
        }

        double dotProduct = 0;
        double normA = 0;
        double normB = 0;

        for (int i = 0; i < a.Length; i++)
        {
            dotProduct += (double)a[i] * b[i];
            normA += (double)a[i] * a[i];
            normB += (double)b[i] * b[i];
        }

        if (normA == 0 || normB == 0)
        {
            throw new ArgumentException("Cosine distance is undefined for zero vectors.");
        }

        double similarity = dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB));
        similarity = Math.Clamp(similarity, -1.0, 1.0);

        return (float)(1.0 - similarity);
    }
}