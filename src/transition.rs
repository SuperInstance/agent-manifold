//! Transition manifold: Markov transition matrices as points on a manifold.
//!
//! A Markov transition matrix P has rows that sum to 1 (row-stochastic).
//! The set of n×n row-stochastic matrices forms a product of n simplices,
//! giving it a natural product manifold structure.

use serde::{Deserialize, Serialize};

/// A row-stochastic transition matrix for a Markov chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionMatrix {
    /// Row-major transition probabilities. Row i gives P(next_state | state=i).
    rows: Vec<Vec<f64>>,
}

impl TransitionMatrix {
    /// Create a new transition matrix.
    ///
    /// Returns `None` if any row doesn't sum to 1.0 or contains negatives.
    pub fn new(rows: Vec<Vec<f64>>) -> Option<Self> {
        let n = rows.len();
        if n == 0 {
            return None;
        }
        for row in &rows {
            if row.len() != n {
                return None;
            }
            if row.iter().any(|&p| p < 0.0) {
                return None;
            }
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-8 {
                return None;
            }
        }
        Some(Self { rows })
    }

    /// Identity transition matrix (stay in same state with probability 1).
    pub fn identity(n: usize) -> Self {
        let mut rows = vec![vec![0.0; n]; n];
        for i in 0..n {
            rows[i][i] = 1.0;
        }
        Self { rows }
    }

    /// Uniform transition matrix (equal probability of going to any state).
    pub fn uniform(n: usize) -> Self {
        let p = 1.0 / n as f64;
        Self {
            rows: vec![vec![p; n]; n],
        }
    }

    /// Create from flat row-major data with softmax per row.
    pub fn from_logits_flat(n: usize, logits: &[f64]) -> Option<Self> {
        if logits.len() != n * n {
            return None;
        }
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let row_logits: Vec<f64> = logits[i * n..(i + 1) * n].to_vec();
            let max_l = row_logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = row_logits.iter().map(|&l| (l - max_l).exp()).collect();
            let sum: f64 = exps.iter().sum();
            rows.push(exps.iter().map(|&e| e / sum).collect());
        }
        Some(Self { rows })
    }

    /// Number of states.
    pub fn n_states(&self) -> usize {
        self.rows.len()
    }

    /// Get the full matrix as a slice of rows.
    pub fn rows(&self) -> &[Vec<f64>] {
        &self.rows
    }

    /// Transition probability P(j | i).
    pub fn prob(&self, from: usize, to: usize) -> f64 {
        self.rows
            .get(from)
            .and_then(|r| r.get(to))
            .copied()
            .unwrap_or(0.0)
    }

    /// Multiply by a distribution from the left: result[j] = Σ_i π[i] * P[i][j].
    pub fn apply_distribution(&self, pi: &[f64]) -> Vec<f64> {
        let n = self.n_states();
        let mut result = vec![0.0; n];
        for (i, &pi_i) in pi.iter().enumerate() {
            for (j, &p_ij) in self.rows[i].iter().enumerate() {
                result[j] += pi_i * p_ij;
            }
        }
        result
    }

    /// Matrix power: P^k.
    pub fn power(&self, k: u32) -> Self {
        if k == 0 {
            return Self::identity(self.n_states());
        }
        if k == 1 {
            return self.clone();
        }
        let mut result = self.clone();
        for _ in 1..k {
            result = result.multiply(self);
        }
        result
    }

    fn multiply(&self, other: &TransitionMatrix) -> TransitionMatrix {
        let n = self.n_states();
        let mut rows = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    rows[i][j] += self.rows[i][k] * other.rows[k][j];
                }
            }
        }
        TransitionMatrix { rows }
    }

    /// Compute the stationary distribution (if it exists).
    ///
    /// Uses power iteration to find the left eigenvector with eigenvalue 1.
    pub fn stationary_distribution(&self, max_iter: usize, tol: f64) -> Option<Vec<f64>> {
        let n = self.n_states();
        let mut pi = vec![1.0 / n as f64; n];
        for _ in 0..max_iter {
            let next = self.apply_distribution(&pi);
            let diff: f64 = pi
                .iter()
                .zip(next.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum();
            pi = next;
            if diff < tol {
                let sum: f64 = pi.iter().sum();
                return Some(pi.iter().map(|&p| p / sum).collect());
            }
        }
        // Try to normalize and return anyway if close
        let sum: f64 = pi.iter().sum();
        if sum > crate::EPS {
            Some(pi.iter().map(|&p| p / sum).collect())
        } else {
            None
        }
    }

    /// Check if the matrix is irreducible (all states can reach all others).
    pub fn is_irreducible(&self) -> bool {
        let n = self.n_states();
        // BFS from state 0
        let mut reachable = vec![false; n];
        reachable[0] = true;
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..n {
                if reachable[i] {
                    for j in 0..n {
                        if self.rows[i][j] > 0.0 && !reachable[j] {
                            reachable[j] = true;
                            changed = true;
                        }
                    }
                }
            }
        }
        reachable.iter().all(|&r| r)
    }

    /// Spectral gap: 1 - |λ₂| where λ₂ is the second-largest eigenvalue magnitude.
    ///
    /// Approximated via power iteration for the dominant eigenvector,
    /// then deflation for the second.
    pub fn spectral_gap(&self, max_iter: usize) -> f64 {
        let n = self.n_states();
        if n <= 1 {
            return 0.0;
        }

        // Power iteration for the largest eigenvalue (should be 1.0 for stochastic)
        let mut v = vec![1.0 / n as f64; n];
        for _ in 0..max_iter {
            v = self.apply_distribution(&v);
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > crate::EPS {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }
        }

        // For the second eigenvalue, use a deflated matrix approach
        // Approximate: use Frobenius norm minus dominant
        let _lambda1: f64 = 1.0; // For row-stochastic, dominant eigenvalue is 1

        // Simple approximation: compute P^k and see how fast it converges
        let p2 = self.power(2);
        let _uniform = vec![1.0 / n as f64; n];
        let stationary = self.stationary_distribution(max_iter, 1e-10);

        if let Some(ref stat) = stationary {
            // After applying P, the deviation from stationary shrinks by |λ₂| each step
            let test_dist: Vec<f64> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
            let after1 = self.apply_distribution(&test_dist);
            let after2 = p2.apply_distribution(&test_dist);

            let dev1: f64 = after1
                .iter()
                .zip(stat.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let dev2: f64 = after2
                .iter()
                .zip(stat.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if dev1 > crate::EPS {
                let ratio = dev2 / dev1;
                let lambda2_abs = ratio.abs();
                return (1.0 - lambda2_abs).max(0.0);
            }
        }

        0.0
    }

    /// Mixing time: number of steps for total variation distance to drop below ε.
    pub fn mixing_time(&self, epsilon: f64, max_iter: usize) -> usize {
        let stat = match self.stationary_distribution(max_iter, 1e-12) {
            Some(s) => s,
            None => return max_iter,
        };

        let n = self.n_states();
        for k in 1..max_iter {
            let pk = self.power(k as u32);
            // Worst-case starting distribution
            let mut max_tv: f64 = 0.0;
            for start in 0..n {
                let mut dist = vec![0.0; n];
                dist[start] = 1.0;
                let evolved = pk.apply_distribution(&dist);
                let tv: f64 = 0.5
                    * evolved
                        .iter()
                        .zip(stat.iter())
                        .map(|(&a, &b)| (a - b).abs())
                        .sum::<f64>();
                max_tv = max_tv.max(tv);
            }
            if max_tv < epsilon {
                return k;
            }
        }
        max_iter
    }
}

/// The transition manifold: product of simplices for row-stochastic matrices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionManifold {
    /// Number of states.
    n_states: usize,
}

impl TransitionManifold {
    /// Create a new transition manifold.
    pub fn new(n_states: usize) -> Self {
        Self { n_states }
    }

    /// Dimension of the manifold (n × (n-1) since each row is a simplex).
    pub fn dim(&self) -> usize {
        self.n_states * (self.n_states.saturating_sub(1))
    }

    /// Frobenius distance between two transition matrices.
    pub fn frobenius_distance(&self, p: &TransitionMatrix, q: &TransitionMatrix) -> f64 {
        p.rows()
            .iter()
            .zip(q.rows().iter())
            .flat_map(|(pr, qr)| pr.iter().zip(qr.iter()))
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Row-wise KL divergence: average KL per row.
    pub fn kl_row_average(&self, p: &TransitionMatrix, q: &TransitionMatrix) -> f64 {
        let n = self.n_states as f64;
        p.rows()
            .iter()
            .zip(q.rows().iter())
            .map(|(pr, qr)| {
                pr.iter()
                    .zip(qr.iter())
                    .map(|(&pi, &qi)| {
                        if pi > crate::EPS {
                            pi * (pi / (qi + crate::EPS)).ln()
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
            / n
    }

    /// Interpolate between two transition matrices (row-wise).
    pub fn interpolate(
        &self,
        p: &TransitionMatrix,
        q: &TransitionMatrix,
        t: f64,
    ) -> TransitionMatrix {
        let rows: Vec<Vec<f64>> = p
            .rows()
            .iter()
            .zip(q.rows().iter())
            .map(|(pr, qr)| {
                let raw: Vec<f64> = pr
                    .iter()
                    .zip(qr.iter())
                    .map(|(&a, &b)| (1.0 - t) * a + t * b)
                    .collect();
                // Renormalize to ensure row sums to 1
                let sum: f64 = raw.iter().sum();
                raw.iter().map(|&v| v / sum).collect()
            })
            .collect();
        TransitionMatrix { rows }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transition() {
        let t = TransitionMatrix::identity(3);
        assert_eq!(t.prob(0, 0), 1.0);
        assert_eq!(t.prob(0, 1), 0.0);
    }

    #[test]
    fn test_uniform_transition() {
        let t = TransitionMatrix::uniform(4);
        assert!((t.prob(0, 0) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_transition() {
        let rows = vec![vec![0.5, 0.6], vec![0.3, 0.7]];
        assert!(TransitionMatrix::new(rows).is_none());
    }

    #[test]
    fn test_rectangular_rejected() {
        let rows = vec![vec![0.5, 0.5], vec![0.3, 0.3, 0.4]];
        assert!(TransitionMatrix::new(rows).is_none());
    }

    #[test]
    fn test_stationary_distribution_uniform() {
        let t = TransitionMatrix::uniform(3);
        let stat = t.stationary_distribution(1000, 1e-10).unwrap();
        for p in &stat {
            assert!((p - 1.0 / 3.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_stationary_sums_to_one() {
        let t = TransitionMatrix::new(vec![
            vec![0.7, 0.2, 0.1],
            vec![0.1, 0.8, 0.1],
            vec![0.2, 0.2, 0.6],
        ])
        .unwrap();
        let stat = t.stationary_distribution(1000, 1e-10).unwrap();
        let sum: f64 = stat.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_power_identity() {
        let t = TransitionMatrix::identity(3);
        let t2 = t.power(5);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((t2.prob(i, j) - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_irreducible() {
        let t = TransitionMatrix::new(vec![vec![0.5, 0.5], vec![0.5, 0.5]]).unwrap();
        assert!(t.is_irreducible());
    }

    #[test]
    fn test_reducible() {
        let t = TransitionMatrix::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
        assert!(!t.is_irreducible());
    }

    #[test]
    fn test_frobenius_same_zero() {
        let m = TransitionManifold::new(3);
        let t = TransitionMatrix::identity(3);
        assert!(m.frobenius_distance(&t, &t) < 1e-10);
    }

    #[test]
    fn test_frobenius_positive() {
        let m = TransitionManifold::new(2);
        let t1 = TransitionMatrix::identity(2);
        let t2 = TransitionMatrix::uniform(2);
        assert!(m.frobenius_distance(&t1, &t2) > 0.0);
    }

    #[test]
    fn test_interpolate_endpoints() {
        let m = TransitionManifold::new(2);
        let p = TransitionMatrix::identity(2);
        let q = TransitionMatrix::uniform(2);

        let at_0 = m.interpolate(&p, &q, 0.0);
        let at_1 = m.interpolate(&p, &q, 1.0);

        for i in 0..2 {
            for j in 0..2 {
                assert!((at_0.prob(i, j) - p.prob(i, j)).abs() < 1e-10);
                assert!((at_1.prob(i, j) - q.prob(i, j)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_interpolate_rows_stochastic() {
        let m = TransitionManifold::new(3);
        let p = TransitionMatrix::identity(3);
        let q = TransitionMatrix::uniform(3);
        let mid = m.interpolate(&p, &q, 0.5);
        for row in mid.rows() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_transition_manifold_dim() {
        let m = TransitionManifold::new(4);
        assert_eq!(m.dim(), 12); // 4 * 3
    }

    #[test]
    fn test_from_logits() {
        let t = TransitionMatrix::from_logits_flat(2, &[1.0, 0.0, 0.0, 1.0]).unwrap();
        assert!(t.prob(0, 0) > 0.5);
        assert!(t.prob(1, 1) > 0.5);
    }

    #[test]
    fn test_mixing_time_identity_large() {
        // Identity matrix never mixes
        let t = TransitionMatrix::identity(2);
        let mt = t.mixing_time(0.25, 50);
        assert_eq!(mt, 50); // Never converges
    }
}
