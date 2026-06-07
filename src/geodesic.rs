//! Geodesic interpolation between agent configurations.

use serde::{Deserialize, Serialize};

use crate::belief::BeliefState;
use crate::policy::Policy;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentGeodesic {
    start_policy: Policy,
    end_policy: Policy,
    start_belief: BeliefState,
    end_belief: BeliefState,
}

impl AgentGeodesic {
    pub fn new(
        start_policy: Policy,
        end_policy: Policy,
        start_belief: BeliefState,
        end_belief: BeliefState,
    ) -> Self {
        Self {
            start_policy,
            end_policy,
            start_belief,
            end_belief,
        }
    }

    pub fn policy_only(start: Policy, end: Policy) -> Option<Self> {
        let n = start.n_actions().max(end.n_actions());
        Some(Self {
            start_policy: start,
            end_policy: end,
            start_belief: BeliefState::uniform(n),
            end_belief: BeliefState::uniform(n),
        })
    }

    pub fn evaluate(&self, t: f64) -> (Policy, BeliefState) {
        let policy = self.interpolate_policy(t);
        let belief = self.interpolate_belief(t);
        (policy, belief)
    }

    fn interpolate_policy(&self, t: f64) -> Policy {
        let sp: Vec<f64> = self
            .start_policy
            .probs()
            .iter()
            .map(|&p| p.sqrt())
            .collect();
        let ep: Vec<f64> = self.end_policy.probs().iter().map(|&p| p.sqrt()).collect();
        let interp: Vec<f64> = sp
            .iter()
            .zip(ep.iter())
            .map(|(&a, &b)| {
                let v = (1.0 - t) * a + t * b;
                v * v
            })
            .collect();
        let sum: f64 = interp.iter().sum();
        Policy::new(interp.iter().map(|&v| v / sum).collect())
            .unwrap_or_else(|| Policy::uniform(interp.len()))
    }

    fn interpolate_belief(&self, t: f64) -> BeliefState {
        let sb: Vec<f64> = self
            .start_belief
            .probs()
            .iter()
            .map(|&p| p.sqrt())
            .collect();
        let eb: Vec<f64> = self.end_belief.probs().iter().map(|&p| p.sqrt()).collect();
        let interp: Vec<f64> = sb
            .iter()
            .zip(eb.iter())
            .map(|(&a, &b)| {
                let v = (1.0 - t) * a + t * b;
                v * v
            })
            .collect();
        let sum: f64 = interp.iter().sum();
        BeliefState::new(interp.iter().map(|&v| v / sum).collect())
            .unwrap_or_else(|| BeliefState::uniform(interp.len()))
    }

    pub fn length(&self, n_segments: usize) -> f64 {
        if n_segments == 0 {
            return 0.0;
        }
        let dt = 1.0 / n_segments as f64;
        let mut total = 0.0;
        let mut prev = self.evaluate(0.0);
        for i in 1..=n_segments {
            let t = i as f64 * dt;
            let curr = self.evaluate(t);
            total += self.euclidean_distance(&prev, &curr);
            prev = curr;
        }
        total
    }

    fn euclidean_distance(&self, a: &(Policy, BeliefState), b: &(Policy, BeliefState)) -> f64 {
        let pd: f64 =
            a.0.probs()
                .iter()
                .zip(b.0.probs().iter())
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt();
        let bd: f64 =
            a.1.probs()
                .iter()
                .zip(b.1.probs().iter())
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt();
        (pd.powi(2) + bd.powi(2)).sqrt()
    }

    pub fn sample(&self, n: usize) -> Vec<(Policy, BeliefState)> {
        (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1).max(1) as f64;
                self.evaluate(t)
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalPolicyUpdate {
    step_size: f64,
}

impl NaturalPolicyUpdate {
    pub fn new(step_size: f64) -> Self {
        Self { step_size }
    }

    pub fn update(&self, policy: &Policy, natural_gradient: &[f64]) -> Policy {
        let logits: Vec<f64> = policy
            .probs()
            .iter()
            .zip(natural_gradient.iter())
            .map(|(&p, &g)| {
                if p > crate::EPS {
                    p.ln() + self.step_size * g
                } else {
                    self.step_size * g
                }
            })
            .collect();
        Policy::from_logits(&logits)
    }

    pub fn step_size(&self) -> f64 {
        self.step_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::PolicyManifold;

    #[test]
    fn test_geodesic_endpoints() {
        let p1 = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let p2 = Policy::new(vec![0.5, 0.3, 0.2]).unwrap();
        let geo = AgentGeodesic::policy_only(p1.clone(), p2.clone()).unwrap();
        let (at_0, _) = geo.evaluate(0.0);
        let (at_1, _) = geo.evaluate(1.0);
        for i in 0..3 {
            assert!((at_0.prob(i) - p1.prob(i)).abs() < 1e-8);
            assert!((at_1.prob(i) - p2.prob(i)).abs() < 1e-8);
        }
    }

    #[test]
    fn test_geodesic_midpoint_valid() {
        let p1 = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let p2 = Policy::new(vec![0.5, 0.3, 0.2]).unwrap();
        let geo = AgentGeodesic::policy_only(p1, p2).unwrap();
        let (mid, _) = geo.evaluate(0.5);
        assert!((mid.probs().iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!(mid.probs().iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_geodesic_length_positive() {
        let p1 = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let p2 = Policy::new(vec![0.5, 0.3, 0.2]).unwrap();
        let geo = AgentGeodesic::policy_only(p1, p2).unwrap();
        assert!(geo.length(100) > 0.0);
    }

    #[test]
    fn test_geodesic_same_point_length_zero() {
        let p = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let geo = AgentGeodesic::policy_only(p.clone(), p).unwrap();
        assert!(geo.length(100) < 1e-10);
    }

    #[test]
    fn test_geodesic_sample_count() {
        let p1 = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let p2 = Policy::new(vec![0.5, 0.3, 0.2]).unwrap();
        let geo = AgentGeodesic::policy_only(p1, p2).unwrap();
        let samples = geo.sample(5);
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn test_natural_update_produces_valid_policy() {
        let p = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let updater = NaturalPolicyUpdate::new(0.1);
        let grad = vec![1.0, 0.0, -1.0];
        let updated = updater.update(&p, &grad);
        assert!((updated.probs().iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!(updated.probs().iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_natural_update_moves_toward_reward() {
        let p = Policy::new(vec![0.2, 0.3, 0.5]).unwrap();
        let m = PolicyManifold::new(3);
        let rewards = vec![10.0, 0.0, 0.0];
        let grad = m.natural_gradient(&p, &rewards);
        let updater = NaturalPolicyUpdate::new(0.5);
        let updated = updater.update(&p, &grad);
        assert!(updated.prob(0) > p.prob(0));
    }

    #[test]
    fn test_full_geodesic_with_belief() {
        let p1 = Policy::new(vec![0.2, 0.8]).unwrap();
        let p2 = Policy::new(vec![0.7, 0.3]).unwrap();
        let b1 = BeliefState::new(vec![0.9, 0.1]).unwrap();
        let b2 = BeliefState::new(vec![0.1, 0.9]).unwrap();
        let geo = AgentGeodesic::new(p1, p2, b1, b2);
        let (pol, bel) = geo.evaluate(0.5);
        assert!((pol.probs().iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!((bel.probs().iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }
}
