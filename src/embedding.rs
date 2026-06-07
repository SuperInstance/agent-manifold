//! Agent embedding: low-dimensional manifold embeddings of agent parameters.
//!
//! High-dimensional agent parameters can be mapped to a low-dimensional manifold
//! for visualization, clustering, and similarity search. This module provides
//! t-SNE-like neighbor-preserving embeddings and manifold-based distance metrics.

use serde::{Deserialize, Serialize};

use crate::policy::{Policy, PolicyManifold};

/// An agent's parameterization represented as a point in a high-dimensional space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPoint {
    /// Flat parameter vector.
    params: Vec<f64>,
}

impl AgentPoint {
    /// Create a new agent point from a parameter vector.
    pub fn new(params: Vec<f64>) -> Self {
        Self { params }
    }

    /// Create from a policy.
    pub fn from_policy(policy: &Policy) -> Self {
        Self {
            params: policy.probs().to_vec(),
        }
    }

    /// Dimension of the parameter space.
    pub fn dim(&self) -> usize {
        self.params.len()
    }

    /// Parameter vector.
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// Euclidean norm.
    pub fn euclidean_norm(&self) -> f64 {
        self.params.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Euclidean distance to another point.
    pub fn euclidean_distance(&self, other: &AgentPoint) -> f64 {
        self.params
            .iter()
            .zip(other.params.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Low-dimensional embedding of agent parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEmbedding {
    /// Embedding dimension.
    dim: usize,
    /// Perplexity for neighbor preservation (t-SNE-like parameter).
    perplexity: f64,
    /// Number of optimization iterations.
    n_iter: usize,
}

impl AgentEmbedding {
    /// Create a new embedding configuration.
    pub fn new(dim: usize, perplexity: f64, n_iter: usize) -> Self {
        Self {
            dim,
            perplexity,
            n_iter,
        }
    }

    /// Default 2D embedding for visualization.
    pub fn default_2d() -> Self {
        Self::new(2, 30.0, 500)
    }

    /// Default 3D embedding.
    pub fn default_3d() -> Self {
        Self::new(3, 30.0, 500)
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Compute pairwise distances between agents using manifold distance.
    ///
    /// For policies, uses symmetric KL-divergence.
    pub fn pairwise_distances(&self, agents: &[Policy]) -> Vec<Vec<f64>> {
        let manifold = PolicyManifold::new(agents.first().map(|p| p.n_actions()).unwrap_or(0));
        let n = agents.len();
        let mut dists = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = manifold.distance(&agents[i], &agents[j]);
                dists[i][j] = d;
                dists[j][i] = d;
            }
        }
        dists
    }

    /// Compute a simple t-SNE-like embedding using gradient descent.
    ///
    /// Minimizes KL divergence between pairwise similarity distributions
    /// in high-dimensional and low-dimensional spaces.
    pub fn embed(&self, agents: &[Policy], seed: u64) -> Vec<EmbeddedPoint> {
        let n = agents.len();
        if n == 0 {
            return vec![];
        }

        let dists = self.pairwise_distances(agents);

        // Compute pairwise affinities in high-dimensional space (Gaussian kernel)
        let sigma = self.compute_sigma(&dists);
        let p_affinity = self.high_dim_affinities(&dists, sigma);

        // Initialize embedding randomly (simple LCG for reproducibility)
        let mut embedded = self.random_init(n, seed);

        // Gradient descent to minimize KL(P || Q)
        let learning_rate = 100.0;
        let momentum = 0.8;
        let mut gains: Vec<Vec<f64>> = vec![vec![1.0; self.dim]; n];
        let mut velocities: Vec<Vec<f64>> = vec![vec![0.0; self.dim]; n];

        for _ in 0..self.n_iter {
            // Compute low-dimensional affinities (t-distribution kernel)
            let q_affinity = self.low_dim_affinities(&embedded);

            // Compute gradients
            let gradients = self.compute_gradients(&p_affinity, &q_affinity, &embedded, n);

            // Update with momentum and adaptive gains
            for i in 0..n {
                for d in 0..self.dim {
                    if gradients[i][d] * velocities[i][d] < 0.0 {
                        gains[i][d] += 0.2;
                    } else {
                        gains[i][d] *= 0.8;
                    }
                    gains[i][d] = gains[i][d].max(0.01);

                    velocities[i][d] =
                        momentum * velocities[i][d] - learning_rate * gains[i][d] * gradients[i][d];
                    embedded[i].coords[d] += velocities[i][d];
                }
            }

            // Center the embedding
            let center: Vec<f64> = (0..self.dim)
                .map(|d| embedded.iter().map(|p| p.coords[d]).sum::<f64>() / n as f64)
                .collect();
            for p in &mut embedded {
                for d in 0..self.dim {
                    p.coords[d] -= center[d];
                }
            }
        }

        embedded
    }

    fn compute_sigma(&self, dists: &[Vec<f64>]) -> f64 {
        let n = dists.len();
        if n <= 1 {
            return 1.0;
        }
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                total += dists[i][j];
                count += 1;
            }
        }
        if count > 0 { total / count as f64 } else { 1.0 }
    }

    fn high_dim_affinities(&self, dists: &[Vec<f64>], sigma: f64) -> Vec<Vec<f64>> {
        let n = dists.len();
        let sigma2 = 2.0 * sigma * sigma;
        let mut p = vec![vec![0.0; n]; n];
        for i in 0..n {
            let row_sum: f64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| (-dists[i][j].powi(2) / sigma2).exp())
                .sum();
            for j in 0..n {
                if i != j && row_sum > crate::EPS {
                    p[i][j] = (-dists[i][j].powi(2) / sigma2).exp() / (2.0 * row_sum);
                }
            }
        }
        // Symmetrize
        for i in 0..n {
            for j in (i + 1)..n {
                let sym = (p[i][j] + p[j][i]) / (2.0 * n as f64);
                p[i][j] = sym;
                p[j][i] = sym;
            }
        }
        p
    }

    fn low_dim_affinities(&self, embedded: &[EmbeddedPoint]) -> Vec<Vec<f64>> {
        let n = embedded.len();
        let mut q = vec![vec![0.0; n]; n];
        let mut total = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let dist_sq: f64 = embedded[i]
                    .coords
                    .iter()
                    .zip(embedded[j].coords.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                let val = 1.0 / (1.0 + dist_sq);
                q[i][j] = val;
                q[j][i] = val;
                total += 2.0 * val;
            }
        }
        if total > crate::EPS {
            for row in &mut q {
                for val in row.iter_mut() {
                    *val /= total;
                }
            }
        }
        q
    }

    fn compute_gradients(
        &self,
        p: &[Vec<f64>],
        q: &[Vec<f64>],
        embedded: &[EmbeddedPoint],
        n: usize,
    ) -> Vec<Vec<f64>> {
        let mut grads = vec![vec![0.0; self.dim]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dist_sq: f64 = embedded[i]
                    .coords
                    .iter()
                    .zip(embedded[j].coords.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                let factor = 4.0 * (p[i][j] - q[i][j]) / (1.0 + dist_sq);
                for d in 0..self.dim {
                    grads[i][d] += factor * (embedded[i].coords[d] - embedded[j].coords[d]);
                }
            }
        }
        grads
    }

    fn random_init(&self, n: usize, seed: u64) -> Vec<EmbeddedPoint> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                let coords: Vec<f64> = (0..self.dim)
                    .map(|_| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let r = (state >> 33) as f64 / (1u64 << 31) as f64 - 1.0;
                        r * 0.01
                    })
                    .collect();
                EmbeddedPoint { coords }
            })
            .collect()
    }

    /// Agent similarity via manifold distance (symmetric KL).
    pub fn similarity(&self, a: &Policy, b: &Policy) -> f64 {
        let m = PolicyManifold::new(a.n_actions());
        let d = m.distance(a, b);
        1.0 / (1.0 + d)
    }
}

/// A point in the low-dimensional embedding space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedPoint {
    /// Coordinates in the embedding space.
    pub coords: Vec<f64>,
}

impl EmbeddedPoint {
    /// Create a new embedded point.
    pub fn new(coords: Vec<f64>) -> Self {
        Self { coords }
    }

    /// Euclidean distance to another embedded point.
    pub fn distance(&self, other: &EmbeddedPoint) -> f64 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_point_from_policy() {
        let p = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let ap = AgentPoint::from_policy(&p);
        assert_eq!(ap.dim(), 3);
        assert!((ap.params()[0] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = AgentPoint::new(vec![0.0, 0.0]);
        let b = AgentPoint::new(vec![3.0, 4.0]);
        assert!((a.euclidean_distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_norm() {
        let a = AgentPoint::new(vec![3.0, 4.0]);
        assert!((a.euclidean_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_distances_symmetric() {
        let e = AgentEmbedding::default_2d();
        let p1 = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let p2 = Policy::new(vec![0.5, 0.3, 0.2]).unwrap();
        let p3 = Policy::uniform(3);
        let dists = e.pairwise_distances(&[p1, p2, p3]);
        assert_eq!(dists.len(), 3);
        for i in 0..3 {
            assert!(dists[i][i].abs() < 1e-10);
            for j in 0..3 {
                assert!((dists[i][j] - dists[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_embed_produces_correct_count() {
        let e = AgentEmbedding::new(2, 5.0, 100);
        let policies: Vec<Policy> = (0..5)
            .map(|i| {
                let mut p = vec![0.1; 4];
                p[i % 4] = 0.7;
                Policy::new(p).unwrap()
            })
            .collect();
        let embedded = e.embed(&policies, 42);
        assert_eq!(embedded.len(), 5);
        assert_eq!(embedded[0].coords.len(), 2);
    }

    #[test]
    fn test_embed_empty() {
        let e = AgentEmbedding::default_2d();
        let embedded = e.embed(&[], 42);
        assert!(embedded.is_empty());
    }

    #[test]
    fn test_similarity_identical() {
        let e = AgentEmbedding::default_2d();
        let p = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let sim = e.similarity(&p, &p);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_decreases_with_distance() {
        let e = AgentEmbedding::default_2d();
        let p = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let q1 = Policy::new(vec![0.25, 0.3, 0.45]).unwrap();
        let q2 = Policy::new(vec![0.8, 0.1, 0.1]).unwrap();
        let sim1 = e.similarity(&p, &q1);
        let sim2 = e.similarity(&p, &q2);
        assert!(sim1 > sim2);
    }

    #[test]
    fn test_embedded_point_distance() {
        let a = EmbeddedPoint::new(vec![0.0, 0.0]);
        let b = EmbeddedPoint::new(vec![1.0, 0.0]);
        assert!((a.distance(&b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_default_3d() {
        let e = AgentEmbedding::default_3d();
        assert_eq!(e.dim(), 3);
    }
}
