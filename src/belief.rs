//! Belief manifold: probability distributions over states as points on the simplex.
//!
//! The probability simplex Δ^{n-1} = {p ∈ ℝ^n : p_i ≥ 0, Σp_i = 1} is the
//! natural manifold for belief states. We equip it with the Fisher information
//! metric, making it a Riemannian manifold with rich geometric structure.

use serde::{Deserialize, Serialize};

/// A belief state: a probability distribution over a finite state space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefState {
    /// State probabilities (must sum to 1.0, all non-negative).
    probs: Vec<f64>,
}

impl BeliefState {
    /// Create a new belief state from a probability vector.
    pub fn new(probs: Vec<f64>) -> Option<Self> {
        let sum: f64 = probs.iter().sum();
        if probs.iter().any(|&p| p < 0.0) || (sum - 1.0).abs() > 1e-8 {
            return None;
        }
        Some(Self { probs })
    }

    /// Uniform belief over n states.
    pub fn uniform(n: usize) -> Self {
        let p = 1.0 / n as f64;
        Self { probs: vec![p; n] }
    }

    /// Point mass belief (certain about a single state).
    pub fn point_mass(n: usize, state: usize) -> Option<Self> {
        if state >= n {
            return None;
        }
        let mut probs = vec![0.0; n];
        probs[state] = 1.0;
        Some(Self { probs })
    }

    /// Number of states.
    pub fn n_states(&self) -> usize {
        self.probs.len()
    }

    /// State probabilities.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// Probability of a specific state.
    pub fn prob(&self, state: usize) -> f64 {
        self.probs.get(state).copied().unwrap_or(0.0)
    }

    /// Shannon entropy of the belief.
    pub fn entropy(&self) -> f64 {
        self.probs
            .iter()
            .map(|&p| if p > crate::EPS { -p * p.ln() } else { 0.0 })
            .sum()
    }

    /// Is this belief concentrated at a single point (entropy ≈ 0)?
    pub fn is_deterministic(&self) -> bool {
        self.entropy() < 1e-8
    }

    /// Update belief using Bayes' rule with a likelihood vector.
    ///
    /// Returns `None` if the resulting distribution is invalid (all zeros).
    pub fn bayes_update(&self, likelihood: &[f64]) -> Option<Self> {
        let posterior: Vec<f64> = self
            .probs
            .iter()
            .zip(likelihood.iter())
            .map(|(&p, &l)| p * l)
            .collect();
        let sum: f64 = posterior.iter().sum();
        if sum < crate::EPS {
            return None;
        }
        Some(Self {
            probs: posterior.iter().map(|&p| p / sum).collect(),
        })
    }
}

/// The belief simplex manifold Δ^{n-1}.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefManifold {
    /// Number of states.
    n_states: usize,
}

impl BeliefManifold {
    /// Create a new belief manifold for n states.
    pub fn new(n_states: usize) -> Self {
        Self { n_states }
    }

    /// Dimension of the simplex (n-1).
    pub fn dim(&self) -> usize {
        self.n_states.saturating_sub(1)
    }

    /// Fisher information metric (inner product of tangent vectors at a belief).
    ///
    /// For the simplex with categorical distributions: ⟨u, v⟩_p = Σ u_i v_i / p_i
    pub fn fisher_inner(&self, p: &BeliefState, u: &[f64], v: &[f64]) -> f64 {
        u.iter()
            .zip(v.iter())
            .zip(p.probs().iter())
            .map(|((&ui, &vi), &pi)| ui * vi / (pi + crate::EPS))
            .sum()
    }

    /// Norm of a tangent vector at a belief point.
    pub fn fisher_norm(&self, p: &BeliefState, u: &[f64]) -> f64 {
        self.fisher_inner(p, u, u).sqrt()
    }

    /// Geodesic distance on the simplex (Fisher-Rao distance).
    ///
    /// For categorical distributions, this is 2 * arccos(Σ√(p_i q_i)).
    pub fn fisher_rao_distance(&self, p: &BeliefState, q: &BeliefState) -> f64 {
        let bc: f64 = p
            .probs()
            .iter()
            .zip(q.probs().iter())
            .map(|(&pi, &qi)| (pi * qi).sqrt())
            .sum();
        2.0 * bc.clamp(-1.0, 1.0).acos()
    }

    /// Interpolate between beliefs along the Fisher-Rao geodesic.
    ///
    /// Returns the belief at parameter t ∈ [0, 1] along the geodesic.
    pub fn geodesic_point(&self, p: &BeliefState, q: &BeliefState, t: f64) -> BeliefState {
        let sqrt_p: Vec<f64> = p.probs().iter().map(|&pi| pi.sqrt()).collect();
        let sqrt_q: Vec<f64> = q.probs().iter().map(|&qi| qi.sqrt()).collect();
        let interp: Vec<f64> = sqrt_p
            .iter()
            .zip(sqrt_q.iter())
            .map(|(&sp, &sq)| {
                let v = (1.0 - t) * sp + t * sq;
                v * v
            })
            .collect();
        let sum: f64 = interp.iter().sum();
        BeliefState {
            probs: interp.iter().map(|&v| v / sum).collect(),
        }
    }

    /// Total variation distance between two beliefs.
    pub fn total_variation(&self, p: &BeliefState, q: &BeliefState) -> f64 {
        0.5 * p
            .probs()
            .iter()
            .zip(q.probs().iter())
            .map(|(&pi, &qi)| (pi - qi).abs())
            .sum::<f64>()
    }
}

/// Dirichlet distribution parameterization for the belief simplex.
///
/// The Dirichlet distribution is a natural prior over the simplex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dirichlet {
    /// Concentration parameters (α_i > 0).
    alpha: Vec<f64>,
}

impl Dirichlet {
    /// Create a new Dirichlet distribution.
    ///
    /// Returns `None` if any α_i ≤ 0.
    pub fn new(alpha: Vec<f64>) -> Option<Self> {
        if alpha.iter().any(|&a| a <= 0.0) {
            return None;
        }
        Some(Self { alpha })
    }

    /// Symmetric Dirichlet with all α_i = α.
    pub fn symmetric(n: usize, alpha: f64) -> Option<Self> {
        if alpha <= 0.0 {
            return None;
        }
        Some(Self {
            alpha: vec![alpha; n],
        })
    }

    /// Concentration parameters.
    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }

    /// Sum of concentration parameters.
    pub fn alpha_sum(&self) -> f64 {
        self.alpha.iter().sum()
    }

    /// Mean of the Dirichlet distribution (a point on the simplex).
    pub fn mean(&self) -> BeliefState {
        let sum = self.alpha_sum();
        BeliefState {
            probs: self.alpha.iter().map(|&a| a / sum).collect(),
        }
    }

    /// Mode of the Dirichlet distribution.
    ///
    /// Only valid when all α_i > 1 (otherwise mode is on the boundary).
    pub fn mode(&self) -> Option<BeliefState> {
        if self.alpha.iter().any(|&a| a <= 1.0) {
            return None;
        }
        let sum: f64 = self.alpha.iter().sum::<f64>() - self.alpha.len() as f64;
        Some(BeliefState {
            probs: self.alpha.iter().map(|&a| (a - 1.0) / sum).collect(),
        })
    }

    /// Dimension of the Dirichlet.
    pub fn dim(&self) -> usize {
        self.alpha.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_creation() {
        let b = BeliefState::new(vec![0.3, 0.5, 0.2]).unwrap();
        assert_eq!(b.n_states(), 3);
        assert!((b.prob(1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_belief_invalid() {
        assert!(BeliefState::new(vec![0.5, 0.6]).is_none());
        assert!(BeliefState::new(vec![-0.1, 1.1]).is_none());
    }

    #[test]
    fn test_point_mass() {
        let b = BeliefState::point_mass(3, 1).unwrap();
        assert!(b.is_deterministic());
        assert_eq!(b.prob(1), 1.0);
    }

    #[test]
    fn test_uniform_entropy() {
        let b = BeliefState::uniform(4);
        let h = b.entropy();
        assert!((h - 4.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_bayes_update() {
        let prior = BeliefState::uniform(3);
        let likelihood = vec![0.9, 0.05, 0.05];
        let posterior = prior.bayes_update(&likelihood).unwrap();
        assert!(posterior.prob(0) > 0.5);
    }

    #[test]
    fn test_bayes_update_all_zero() {
        let prior = BeliefState::new(vec![0.5, 0.5]).unwrap();
        assert!(prior.bayes_update(&[0.0, 0.0]).is_none());
    }

    #[test]
    fn test_fisher_rao_distance_same() {
        let m = BeliefManifold::new(3);
        let b = BeliefState::new(vec![0.2, 0.3, 0.5]).unwrap();
        assert!(m.fisher_rao_distance(&b, &b) < 1e-10);
    }

    #[test]
    fn test_fisher_rao_distance_positive() {
        let m = BeliefManifold::new(3);
        let p = BeliefState::new(vec![0.5, 0.3, 0.2]).unwrap();
        let q = BeliefState::new(vec![0.1, 0.2, 0.7]).unwrap();
        let d = m.fisher_rao_distance(&p, &q);
        assert!(d > 0.0);
    }

    #[test]
    fn test_geodesic_endpoints() {
        let m = BeliefManifold::new(3);
        let p = BeliefState::new(vec![0.2, 0.3, 0.5]).unwrap();
        let q = BeliefState::new(vec![0.5, 0.3, 0.2]).unwrap();
        let at_0 = m.geodesic_point(&p, &q, 0.0);
        let at_1 = m.geodesic_point(&p, &q, 1.0);
        for i in 0..3 {
            assert!((at_0.prob(i) - p.prob(i)).abs() < 1e-8);
            assert!((at_1.prob(i) - q.prob(i)).abs() < 1e-8);
        }
    }

    #[test]
    fn test_geodesic_midpoint_valid() {
        let m = BeliefManifold::new(3);
        let p = BeliefState::new(vec![0.2, 0.3, 0.5]).unwrap();
        let q = BeliefState::new(vec![0.5, 0.3, 0.2]).unwrap();
        let mid = m.geodesic_point(&p, &q, 0.5);
        assert!((mid.probs().iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!(mid.probs().iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_total_variation() {
        let m = BeliefManifold::new(3);
        let p = BeliefState::new(vec![0.2, 0.3, 0.5]).unwrap();
        let q = BeliefState::new(vec![0.2, 0.3, 0.5]).unwrap();
        assert!(m.total_variation(&p, &q) < 1e-10);
    }

    #[test]
    fn test_total_variation_max() {
        let m = BeliefManifold::new(3);
        let p = BeliefState::point_mass(3, 0).unwrap();
        let q = BeliefState::point_mass(3, 2).unwrap();
        assert!((m.total_variation(&p, &q) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dirichlet_mean() {
        let d = Dirichlet::new(vec![2.0, 3.0, 5.0]).unwrap();
        let mean = d.mean();
        assert!((mean.prob(0) - 0.2).abs() < 1e-10);
        assert!((mean.prob(1) - 0.3).abs() < 1e-10);
        assert!((mean.prob(2) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_dirichlet_mode() {
        let d = Dirichlet::new(vec![3.0, 4.0, 6.0]).unwrap();
        let mode = d.mode().unwrap();
        assert!((mode.probs().iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dirichlet_mode_boundary() {
        let d = Dirichlet::new(vec![0.5, 2.0, 3.0]).unwrap();
        assert!(d.mode().is_none()); // α_1 ≤ 1
    }

    #[test]
    fn test_dirichlet_invalid() {
        assert!(Dirichlet::new(vec![0.0, 1.0]).is_none());
        assert!(Dirichlet::symmetric(3, -1.0).is_none());
    }

    #[test]
    fn test_simplex_dim() {
        let m = BeliefManifold::new(5);
        assert_eq!(m.dim(), 4);
    }
}
