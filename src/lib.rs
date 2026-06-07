//! # agent-manifold
//!
//! Agent parameter spaces as differentiable manifolds.
//!
//! This crate treats the parameters of AI agents — policies, belief states,
//! transition models — as points on curved manifolds rather than flat Euclidean
//! spaces. Operations like distance, interpolation, and optimization respect
//! the intrinsic geometry of these spaces.
//!
//! ## Modules
//!
//! - [`policy`] — Policy manifolds with KL-divergence geometry
//! - [`belief`] — Belief states on the probability simplex
//! - [`transition`] — Row-stochastic transition matrices
//! - [`geodesic`] — Geodesic interpolation between agent configurations
//! - [`embedding`] — Low-dimensional manifold embeddings of agent parameters
//! - [`projection`] — Tangent-space projections and Riemannian optimization

#![allow(clippy::needless_range_loop)]

pub mod belief;
pub mod embedding;
pub mod geodesic;
pub mod policy;
pub mod projection;
pub mod transition;

/// A small epsilon for numerical stability near boundaries.
pub const EPS: f64 = 1e-12;
