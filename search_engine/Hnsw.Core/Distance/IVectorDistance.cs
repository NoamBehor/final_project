// This interface defines a method to calculate the distance between two vectors.

namespace Hnsw.Core.Distance;

public interface IVectorDistance
{
    float Calculate(float[] a, float[] b);
}