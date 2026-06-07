//! Projection onto constraint manifolds and Riemannian optimization.
//!
//! When optimizing agent parameters, we often need to project gradients onto
//! the tangent space of constraints (e.g., simplex) and retract points back
//! onto the manifold after Euclidean updates. This module provides these
//! operations for Riemannian optimization on agent manifolds.

use serde::{Deserialize, Serialize};

/// Projection operations for the probability simplex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplexProjection {
    /// Dimension of the simplex.
    dim: usize,
}

impl SimplexProjection {
    /// Create a new simplex projection for dimension n.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Project a point onto the probability simplex Δ^{n-1}.
    ///
    /// Uses the algorithm from Duchi et al. (2008): "Efficient Projections
    /// onto the l1-Ball for Learning in High Dimensions."
    pub fn project(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(
            v.len(),
            self.dim,
            "Vector dimension must match simplex dimension"
        );

        let mut u = v.to_vec();
        u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0;
        let mut rho = 0;
        for (i, &val) in u.iter().enumerate() {
            cumsum += val;
            let threshold = (cumsum - 1.0) / (i as f64 + 1.0);
            if val > threshold {
                rho = i;
            }
        }

        cumsum = u[..=rho].iter().sum();
        let theta = (cumsum - 1.0) / (rho as f64 + 1.0);

        v.iter().map(|&vi| (vi - theta).max(0.0)).collect()
    }

    /// Dimension of the simplex.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Check if a vector lies on the simplex.
    pub fn is_on_simplex(&self, v: &[f64], tol: f64) -> bool {
        if v.len() != self.dim {
            return false;
        }
        v.iter().all(|&x| x >= -tol) && (v.iter().sum::<f64>() - 1.0).abs() < tol
    }
}

/// Projection onto the tangent space of the simplex.
///
/// The tangent space at a point p on the simplex consists of vectors
/// whose components sum to zero: {u : Σ u_i = 0}.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TangentProjection {
    /// Dimension of the simplex.
    dim: usize,
}

impl TangentProjection {
    /// Create a new tangent projection for dimension n.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Project a vector onto the tangent space of the simplex.
    ///
    /// Subtracts the mean so the result sums to zero.
    pub fn project(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.dim, "Vector dimension must match");
        let mean = v.iter().sum::<f64>() / self.dim as f64;
        v.iter().map(|&vi| vi - mean).collect()
    }

    /// Check if a vector is in the tangent space (sums to zero).
    pub fn is_tangent(&self, v: &[f64], tol: f64) -> bool {
        v.iter().sum::<f64>().abs() < tol
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Riemannian optimization on the simplex using retractions.
///
/// A retraction maps a point on the manifold plus a tangent vector to a new
/// point on the manifold, providing a first-order approximation to the
/// exponential map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiemannianOptimizer {
    /// Dimension of the parameter space.
    dim: usize,
    /// Step size (learning rate).
    step_size: f64,
    /// Method for retraction.
    retraction: RetractionMethod,
}

/// Available retraction methods for the simplex.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RetractionMethod {
    /// Project onto simplex after Euclidean step.
    Projection,
    /// Multiplicative update (exponential map approximation).
    Exponential,
}

impl RiemannianOptimizer {
    /// Create a new Riemannian optimizer with projection retraction.
    pub fn new(dim: usize, step_size: f64) -> Self {
        Self {
            dim,
            step_size,
            retraction: RetractionMethod::Projection,
        }
    }

    /// Create with exponential retraction.
    pub fn with_exponential(dim: usize, step_size: f64) -> Self {
        Self {
            dim,
            step_size,
            retraction: RetractionMethod::Exponential,
        }
    }

    /// Set the retraction method.
    pub fn set_retraction(&mut self, method: RetractionMethod) {
        self.retraction = method;
    }

    /// Perform a Riemannian gradient step.
    ///
    /// Starting from point `x` on the simplex with gradient `grad`,
    /// projects the gradient onto the tangent space, takes a step,
    /// and retracts back onto the manifold.
    pub fn step(&self, x: &[f64], grad: &[f64]) -> Vec<f64> {
        let tangent_proj = TangentProjection::new(self.dim);
        let tangent_grad = tangent_proj.project(grad);

        let raw_update: Vec<f64> = x
            .iter()
            .zip(tangent_grad.iter())
            .map(|(&xi, &gi)| xi - self.step_size * gi)
            .collect();

        match self.retraction {
            RetractionMethod::Projection => {
                let simplex_proj = SimplexProjection::new(self.dim);
                simplex_proj.project(&raw_update)
            }
            RetractionMethod::Exponential => {
                // Exponential map: p_i * exp(-η * g_i) / Z
                let log_update: Vec<f64> = x
                    .iter()
                    .zip(tangent_grad.iter())
                    .map(|(&xi, &gi)| {
                        if xi > crate::EPS {
                            xi.ln() - self.step_size * gi
                        } else {
                            -self.step_size * gi
                        }
                    })
                    .collect();
                let max_val = log_update.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = log_update.iter().map(|&l| (l - max_val).exp()).collect();
                let sum: f64 = exps.iter().sum();
                exps.iter().map(|&e| e / sum).collect()
            }
        }
    }

    /// Run multiple optimization steps.
    pub fn optimize(&self, x: &[f64], gradients: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut trajectory = vec![x.to_vec()];
        let mut current = x.to_vec();
        for grad in gradients {
            current = self.step(&current, grad);
            trajectory.push(current.clone());
        }
        trajectory
    }

    /// Step size.
    pub fn step_size(&self) -> f64 {
        self.step_size
    }
}

/// Project onto the row-stochastic matrix manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticProjection {
    /// Number of rows (states).
    n_rows: usize,
    /// Number of columns (states).
    n_cols: usize,
}

impl StochasticProjection {
    /// Create a new stochastic projection for n×n matrices.
    pub fn new(n: usize) -> Self {
        Self {
            n_rows: n,
            n_cols: n,
        }
    }

    /// Project a flat matrix (row-major) onto the row-stochastic manifold.
    ///
    /// Each row is independently projected onto the simplex.
    pub fn project(&self, matrix: &[f64]) -> Vec<Vec<f64>> {
        let simplex_proj = SimplexProjection::new(self.n_cols);
        (0..self.n_rows)
            .map(|i| {
                let row = &matrix[i * self.n_cols..(i + 1) * self.n_cols];
                simplex_proj.project(row)
            })
            .collect()
    }

    /// Check if a matrix is row-stochastic.
    pub fn is_row_stochastic(&self, matrix: &[Vec<f64>], tol: f64) -> bool {
        if matrix.len() != self.n_rows {
            return false;
        }
        for row in matrix {
            if row.len() != self.n_cols {
                return false;
            }
            if row.iter().any(|&x| x < -tol) {
                return false;
            }
            if (row.iter().sum::<f64>() - 1.0).abs() > tol {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_projection_valid() {
        let sp = SimplexProjection::new(3);
        let projected = sp.project(&[0.5, 0.5, 0.5]);
        assert!(sp.is_on_simplex(&projected, 1e-8));
    }

    #[test]
    fn test_simplex_projection_negative() {
        let sp = SimplexProjection::new(3);
        let projected = sp.project(&[-1.0, 2.0, 0.0]);
        assert!(sp.is_on_simplex(&projected, 1e-8));
        assert!(projected.iter().all(|&x| x >= -1e-10));
    }

    #[test]
    fn test_simplex_projection_already_on() {
        let sp = SimplexProjection::new(3);
        let v = vec![0.2, 0.3, 0.5];
        let projected = sp.project(&v);
        for i in 0..3 {
            assert!((projected[i] - v[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn test_simplex_projection_sums_to_one() {
        let sp = SimplexProjection::new(5);
        let projected = sp.project(&[10.0, -3.0, 1.0, 0.5, -2.0]);
        let sum: f64 = projected.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_tangent_projection_sums_zero() {
        let tp = TangentProjection::new(4);
        let projected = tp.project(&[1.0, 2.0, 3.0, 4.0]);
        assert!(tp.is_tangent(&projected, 1e-10));
    }

    #[test]
    fn test_tangent_preserves_relative() {
        let tp = TangentProjection::new(3);
        let projected = tp.project(&[1.0, 2.0, 3.0]);
        assert!((projected[1] - projected[0] - 1.0).abs() < 1e-10);
        assert!((projected[2] - projected[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_riemannian_step_stays_on_simplex() {
        let opt = RiemannianOptimizer::new(3, 0.1);
        let x = vec![0.2, 0.3, 0.5];
        let grad = vec![1.0, 0.0, -1.0];
        let result = opt.step(&x, &grad);
        let sp = SimplexProjection::new(3);
        assert!(sp.is_on_simplex(&result, 1e-8));
    }

    #[test]
    fn test_riemannian_exponential_step() {
        let opt = RiemannianOptimizer::with_exponential(3, 0.1);
        let x = vec![0.2, 0.3, 0.5];
        let grad = vec![1.0, 0.0, -1.0];
        let result = opt.step(&x, &grad);
        let sp = SimplexProjection::new(3);
        assert!(sp.is_on_simplex(&result, 1e-8));
    }

    #[test]
    fn test_riemannian_optimize_trajectory() {
        let opt = RiemannianOptimizer::new(3, 0.1);
        let x = vec![0.2, 0.3, 0.5];
        let grads = vec![
            vec![1.0, 0.0, -1.0],
            vec![0.5, 0.5, -1.0],
            vec![0.0, 1.0, -1.0],
        ];
        let trajectory = opt.optimize(&x, &grads);
        assert_eq!(trajectory.len(), 4); // initial + 3 steps
        let sp = SimplexProjection::new(3);
        for point in &trajectory {
            assert!(sp.is_on_simplex(point, 1e-8));
        }
    }

    #[test]
    fn test_stochastic_projection() {
        let sp = StochasticProjection::new(2);
        let flat = vec![1.0, 2.0, 3.0, 4.0];
        let projected = sp.project(&flat);
        assert!(sp.is_row_stochastic(&projected, 1e-8));
    }

    #[test]
    fn test_stochastic_projection_negative() {
        let sp = StochasticProjection::new(3);
        let flat = vec![-1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0, 8.0, -9.0];
        let projected = sp.project(&flat);
        assert!(sp.is_row_stochastic(&projected, 1e-8));
    }

    #[test]
    fn test_is_on_simplex_false() {
        let sp = SimplexProjection::new(3);
        assert!(!sp.is_on_simplex(&[0.5, 0.5, 0.5], 1e-8)); // sums to 1.5
        assert!(!sp.is_on_simplex(&[-0.1, 0.6, 0.5], 1e-8)); // negative
    }
}
