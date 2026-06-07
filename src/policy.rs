//! Policy manifold: parameterized policies as points on a statistical manifold.
//!
//! Each policy is a categorical distribution over actions. The manifold is
//! equipped with the Fisher information metric, making KL-divergence the
//! natural squared distance. Policy gradients live in the tangent space.

use serde::{Deserialize, Serialize};

/// A categorical policy: a probability distribution over a finite action set.
///
/// Probabilities must be non-negative and sum to 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Action probabilities (must sum to 1.0).
    probs: Vec<f64>,
}

impl Policy {
    /// Create a new policy from a probability vector.
    ///
    /// Returns `None` if probabilities don't sum to approximately 1.0
    /// or any probability is negative.
    pub fn new(probs: Vec<f64>) -> Option<Self> {
        let sum: f64 = probs.iter().sum();
        if probs.iter().any(|&p| p < 0.0) || (sum - 1.0).abs() > 1e-8 {
            return None;
        }
        Some(Self { probs })
    }

    /// Create a uniform policy over `n` actions.
    pub fn uniform(n: usize) -> Self {
        let p = 1.0 / n as f64;
        Self { probs: vec![p; n] }
    }

    /// Create a deterministic (one-hot) policy.
    pub fn deterministic(n: usize, action: usize) -> Option<Self> {
        if action >= n {
            return None;
        }
        let mut probs = vec![0.0; n];
        probs[action] = 1.0;
        Some(Self { probs })
    }

    /// Softmax policy from logits.
    pub fn from_logits(logits: &[f64]) -> Self {
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let probs: Vec<f64> = exps.iter().map(|&e| e / sum).collect();
        Self { probs }
    }

    /// Number of actions.
    pub fn n_actions(&self) -> usize {
        self.probs.len()
    }

    /// Action probabilities.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// Probability of a specific action.
    pub fn prob(&self, action: usize) -> f64 {
        self.probs.get(action).copied().unwrap_or(0.0)
    }

    /// Shannon entropy of the policy.
    pub fn entropy(&self) -> f64 {
        self.probs
            .iter()
            .map(|&p| if p > crate::EPS { -p * p.ln() } else { 0.0 })
            .sum()
    }
}

/// The policy manifold: a statistical manifold of categorical distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyManifold {
    /// Number of actions (dimension of the ambient space).
    n_actions: usize,
}

impl PolicyManifold {
    /// Create a new policy manifold for the given number of actions.
    pub fn new(n_actions: usize) -> Self {
        Self { n_actions }
    }

    /// Dimension of the manifold (n_actions - 1, since probabilities sum to 1).
    pub fn dim(&self) -> usize {
        self.n_actions.saturating_sub(1)
    }

    /// Number of actions.
    pub fn n_actions(&self) -> usize {
        self.n_actions
    }

    /// KL-divergence D_KL(p || q) between two policies.
    ///
    /// This is the canonical (asymmetric) distance on the policy manifold.
    pub fn kl_divergence(&self, p: &Policy, q: &Policy) -> f64 {
        p.probs()
            .iter()
            .zip(q.probs().iter())
            .map(|(&pi, &qi)| {
                if pi > crate::EPS && qi > crate::EPS {
                    pi * (pi / qi).ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Symmetric KL-divergence (Jeffreys divergence) between two policies.
    pub fn symmetric_kl(&self, p: &Policy, q: &Policy) -> f64 {
        self.kl_divergence(p, q) + self.kl_divergence(q, p)
    }

    /// Policy distance: the square root of symmetric KL-divergence.
    ///
    /// This is a proper metric (symmetric, positive-definite, triangle inequality).
    pub fn distance(&self, p: &Policy, q: &Policy) -> f64 {
        self.symmetric_kl(p, q).sqrt()
    }

    /// Fisher information matrix at a policy point (diagonal approximation).
    ///
    /// For a categorical distribution, the Fisher matrix has entries
    /// F_ij = δ_ij / p_i, restricted to the tangent space.
    pub fn fisher_diagonal(&self, p: &Policy) -> Vec<f64> {
        p.probs()
            .iter()
            .map(|&pi| 1.0 / (pi + crate::EPS))
            .collect()
    }

    /// Policy gradient as a tangent vector at point `p`.
    ///
    /// Given reward signal `r` for each action, computes the natural gradient
    /// ∇̃ J = F⁻¹ ∇J where F is the Fisher information matrix.
    pub fn natural_gradient(&self, p: &Policy, rewards: &[f64]) -> Vec<f64> {
        let baseline: f64 = p
            .probs()
            .iter()
            .zip(rewards.iter())
            .map(|(&pi, &ri)| pi * ri)
            .sum();
        let advantage: Vec<f64> = rewards.iter().map(|&r| r - baseline).collect();
        // Natural gradient: F^{-1} * gradient ≈ p * advantage for categorical
        p.probs()
            .iter()
            .zip(advantage.iter())
            .map(|(&pi, &ai)| pi * ai)
            .collect()
    }

    /// Compute the midpoint between two policies using geodesic interpolation.
    ///
    /// Uses the Fisher-Rao geodesic for categorical distributions (proportional
    /// to the square-root parameterization).
    pub fn midpoint(&self, p: &Policy, q: &Policy) -> Policy {
        let sqrt_p: Vec<f64> = p.probs().iter().map(|&pi| pi.sqrt()).collect();
        let sqrt_q: Vec<f64> = q.probs().iter().map(|&qi| qi.sqrt()).collect();
        let mid: Vec<f64> = sqrt_p
            .iter()
            .zip(sqrt_q.iter())
            .map(|(&sp, &sq)| {
                let m = (sp + sq) / 2.0;
                m * m
            })
            .collect();
        let sum: f64 = mid.iter().sum();
        Policy {
            probs: mid.iter().map(|&m| m / sum).collect(),
        }
    }

    /// Inner product of two tangent vectors at point p (Fisher metric).
    pub fn inner_product(&self, p: &Policy, u: &[f64], v: &[f64]) -> f64 {
        u.iter()
            .zip(v.iter())
            .zip(p.probs().iter())
            .map(|((&ui, &vi), &pi)| ui * vi / (pi + crate::EPS))
            .sum()
    }

    /// Norm of a tangent vector at point p.
    pub fn norm(&self, p: &Policy, u: &[f64]) -> f64 {
        self.inner_product(p, u, u).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_policy_sums_to_one() {
        let p = Policy::uniform(4);
        assert!((p.probs().iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert_eq!(p.n_actions(), 4);
    }

    #[test]
    fn test_deterministic_policy() {
        let p = Policy::deterministic(3, 1).unwrap();
        assert_eq!(p.prob(0), 0.0);
        assert_eq!(p.prob(1), 1.0);
        assert_eq!(p.prob(2), 0.0);
    }

    #[test]
    fn test_deterministic_out_of_bounds() {
        assert!(Policy::deterministic(3, 5).is_none());
    }

    #[test]
    fn test_policy_new_invalid() {
        assert!(Policy::new(vec![0.5, 0.3]).is_none()); // doesn't sum to 1
        assert!(Policy::new(vec![-0.5, 1.5]).is_none()); // negative
    }

    #[test]
    fn test_from_logits() {
        let p = Policy::from_logits(&[1.0, 2.0, 3.0]);
        assert!((p.probs().iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!(p.prob(2) > p.prob(1));
        assert!(p.prob(1) > p.prob(0));
    }

    #[test]
    fn test_entropy_uniform() {
        let p = Policy::uniform(2);
        let h = p.entropy();
        assert!((h - 2.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_deterministic_zero() {
        let p = Policy::deterministic(3, 0).unwrap();
        assert!(p.entropy().abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_same_policy() {
        let m = PolicyManifold::new(3);
        let p = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        assert!(m.kl_divergence(&p, &p).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_asymmetric() {
        let m = PolicyManifold::new(3);
        let p = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let q = Policy::new(vec![0.1, 0.4, 0.5]).unwrap();
        let kl_pq = m.kl_divergence(&p, &q);
        let kl_qp = m.kl_divergence(&q, &p);
        assert!((kl_pq - kl_qp).abs() > 1e-6);
    }

    #[test]
    fn test_symmetric_kl() {
        let m = PolicyManifold::new(3);
        let p = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let q = Policy::new(vec![0.1, 0.4, 0.5]).unwrap();
        let skl = m.symmetric_kl(&p, &q);
        assert!((skl - m.kl_divergence(&p, &q) - m.kl_divergence(&q, &p)).abs() < 1e-10);
    }

    #[test]
    fn test_distance_non_negative() {
        let m = PolicyManifold::new(4);
        let p = Policy::uniform(4);
        let q = Policy::new(vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        assert!(m.distance(&p, &q) >= 0.0);
    }

    #[test]
    fn test_distance_zero_same_point() {
        let m = PolicyManifold::new(3);
        let p = Policy::new(vec![0.3, 0.3, 0.4]).unwrap();
        assert!(m.distance(&p, &p) < 1e-10);
    }

    #[test]
    fn test_manifold_dim() {
        let m = PolicyManifold::new(5);
        assert_eq!(m.dim(), 4);
    }

    #[test]
    fn test_natural_gradient() {
        let m = PolicyManifold::new(3);
        let p = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let grad = m.natural_gradient(&p, &[1.0, 0.0, 0.0]);
        assert_eq!(grad.len(), 3);
        // Action 0 has highest reward above baseline, should have positive gradient
        let _baseline = 0.2 * 1.0;
        assert!(grad[0] > 0.0);
    }

    #[test]
    fn test_midpoint_between_same_is_same() {
        let m = PolicyManifold::new(3);
        let p = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let mid = m.midpoint(&p, &p);
        for i in 0..3 {
            assert!((mid.prob(i) - p.prob(i)).abs() < 1e-8);
        }
    }

    #[test]
    fn test_fisher_diagonal() {
        let m = PolicyManifold::new(3);
        let p = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let fisher = m.fisher_diagonal(&p);
        assert!(fisher[0] > fisher[2]); // lower prob → higher Fisher info
    }

    #[test]
    fn test_inner_product() {
        let m = PolicyManifold::new(3);
        let p = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let u = vec![1.0, 0.0, 0.0];
        let ip = m.inner_product(&p, &u, &u);
        assert!(ip > 0.0);
    }

    #[test]
    fn test_norm_positive() {
        let m = PolicyManifold::new(3);
        let p = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let u = vec![1.0, 2.0, -1.0];
        assert!(m.norm(&p, &u) > 0.0);
    }
}
