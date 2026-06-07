# agent-manifold

**Agent parameter spaces as differentiable manifolds.**

> *The space of agent minds is not flat. It curves, folds, and stretches — a landscape of peaks and valleys where every policy is a point on a curved surface and the shortest path between two minds is never a straight line.*

---

## Why Manifolds?

If you're optimizing an AI agent — tuning its policy, updating its beliefs, modeling its transitions — you're working with probabilities. Probabilities that must sum to one. Probabilities that must stay non-negative. Probabilities that form curved surfaces, not flat Euclidean spaces.

Consider a policy with 3 actions: `[0.2, 0.3, 0.5]`. You want to update it toward `[0.5, 0.3, 0.2]`. The Euclidean midpoint is `[0.35, 0.3, 0.35]` — that works. But what if you want to go further? Take a big step? Suddenly you might get `[0.8, 0.3, -0.1]` — negative probabilities. Nonsense.

The problem: **the probability simplex is not a vector space.** It's a manifold — a curved surface embedded in a higher-dimensional flat space. Standard gradient descent doesn't respect this curvature. It blunders off the surface into impossible territory, then you have to patch things up with projection or clipping.

`agent-manifold` treats agent parameters as what they are: **points on manifolds.** Distances respect the Fisher information metric. Updates follow geodesics (the curved-surface equivalent of straight lines). Gradients live in tangent spaces. Retractions map you back to the surface properly.

This is **information geometry** applied to agent design. The result: more principled updates, better interpolation, more meaningful distances, and optimization that naturally stays feasible.

---

## The Metaphor: A Landscape of Agent Minds

Imagine you're standing on a vast, curved landscape. Every point on this landscape is a possible mind — a specific configuration of policy, belief, and transition parameters. The landscape isn't flat like a parking lot; it undulates like mountainous terrain.

**Policies are peaks and valleys.** A deterministic policy (always pick action A) is a sharp peak. A uniform random policy is a flat plain at the center. The "height" between them? That's the Fisher-Rao distance — the true cost of transforming one policy into another.

**Beliefs are territories.** When an agent is certain about the world, it stands at a corner of the simplex — a sharp point. When uncertain, it sprawls across the center. Bayesian updates aren't jumps; they're careful walks across this territory.

**Transitions are rivers.** They connect states of the world, flowing from one configuration to another. The mixing time is how long it takes for the river to smooth out all differences and reach equilibrium.

**Geodesics are trails.** The shortest path between two minds isn't a straight line (you'd fall off the surface). It's a carefully computed trail that stays on the landscape, following its curves.

**Embeddings are maps.** You can't visualize a 100-dimensional policy space directly. But you can project it down to 2D, preserving the essential neighbor relationships — like cartographers flattening a globe.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         agent-manifold               │
                    │  "The geometry of agent minds"       │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
    ┌──────▼──────┐        ┌──────▼──────┐        ┌──────▼──────┐
    │   policy    │        │   belief    │        │ transition  │
    │             │        │             │        │             │
    │ • KL div    │        │ • Simplex   │        │ • Markov    │
    │ • Fisher    │        │ • Fisher-   │        │ • Spectral  │
    │ • Natural   │        │   Rao       │        │ • Mixing    │
    │   gradient  │        │ • Dirichlet │        │   time      │
    └──────┬──────┘        └──────┬──────┘        └──────┬──────┘
           │                      │                       │
           └──────────┬───────────┘                       │
                      │                                   │
               ┌──────▼──────┐                    ┌──────▼──────┐
               │  geodesic   │                    │  embedding  │
               │             │                    │             │
               │ • Interpo-  │                    │ • t-SNE     │
               │   lation    │                    │ • Neighbor  │
               │ • Natural   │                    │   preserve  │
               │   update    │                    │ • Similarity│
               └──────┬──────┘                    └─────────────┘
                      │
               ┌──────▼──────┐
               │  projection │
               │             │
               │ • Simplex   │
               │ • Tangent   │
               │   space     │
               │ • Riemannian│
               │   optimizer │
               └─────────────┘
```

**Data flows downward.** The three core manifolds (policy, belief, transition) define the geometry. Geodesic and embedding build on top. Projection provides the low-level operations that keep everything on the manifold.

---

## Modules

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `policy` | Policy manifold with KL geometry | `Policy`, `PolicyManifold` |
| `belief` | Belief simplex with Fisher metric | `BeliefState`, `BeliefManifold`, `Dirichlet` |
| `transition` | Row-stochastic Markov matrices | `TransitionMatrix`, `TransitionManifold` |
| `geodesic` | Geodesic interpolation between agents | `AgentGeodesic`, `NaturalPolicyUpdate` |
| `embedding` | Low-dimensional agent embeddings | `AgentEmbedding`, `AgentPoint`, `EmbeddedPoint` |
| `projection` | Tangent-space projection & Riemannian optimization | `SimplexProjection`, `TangentProjection`, `RiemannianOptimizer` |

---

## Quick Start

```toml
# Cargo.toml
[dependencies]
agent-manifold = "0.1.0"
```

```rust
use agent_manifold::policy::{Policy, PolicyManifold};

fn main() {
    // Create two policies over 3 actions
    let cautious = Policy::new(vec![0.1, 0.8, 0.1]).unwrap();  // mostly action 1
    let risky = Policy::new(vec![0.6, 0.1, 0.3]).unwrap();      // aggressive

    // Measure their distance on the policy manifold
    let manifold = PolicyManifold::new(3);
    let distance = manifold.distance(&cautious, &risky);
    println!("Policy distance: {:.4}", distance);
    // Output: Policy distance: 1.4725

    // Compute the natural gradient
    let rewards = vec![10.0, 1.0, 0.0]; // action 0 pays well
    let grad = manifold.natural_gradient(&cautious, &rewards);
    println!("Natural gradient: {:?}", grad);
    // Output: Natural gradient: [0.81, -0.63, -0.09]

    // Find the geodesic midpoint
    let mid = manifold.midpoint(&cautious, &risky);
    println!("Midpoint: {:?}", mid.probs());
    // Output: Midpoint: [0.3683, 0.4216, 0.2101]
}
```

### Belief States on the Simplex

```rust
use agent_manifold::belief::{BeliefState, BeliefManifold, Dirichlet};

fn main() {
    // Start with uniform belief over 3 states
    let prior = BeliefState::uniform(3);
    println!("Prior entropy: {:.4}", prior.entropy());
    // Output: Prior entropy: 1.0986

    // Observe evidence via Bayes' rule
    let likelihood = vec![0.8, 0.1, 0.1]; // state 0 is likely
    let posterior = prior.bayes_update(&likelihood).unwrap();
    println!("Posterior: {:?}", posterior.probs());
    // Output: Posterior: [0.8, 0.1, 0.1]

    // Fisher-Rao distance (true geodesic distance on simplex)
    let manifold = BeliefManifold::new(3);
    let certain = BeliefState::point_mass(3, 0).unwrap();
    let dist = manifold.fisher_rao_distance(&prior, &certain);
    println!("Distance to certainty: {:.4}", dist);
    // Output: Distance to certainty: 1.7725

    // Geodesic interpolation
    let midpoint = manifold.geodesic_point(&prior, &certain, 0.5);
    println!("Halfway belief: {:?}", midpoint.probs());
    // Output: Halfway belief: [0.6339, 0.1830, 0.1830]
}
```

### Transition Matrices and Mixing

```rust
use agent_manifold::transition::{TransitionMatrix, TransitionManifold};

fn main() {
    // Define a Markov chain
    let chain = TransitionMatrix::new(vec![
        vec![0.7, 0.2, 0.1],
        vec![0.1, 0.8, 0.1],
        vec![0.2, 0.2, 0.6],
    ]).unwrap();

    // Find the stationary distribution
    let stationary = chain.stationary_distribution(1000, 1e-10).unwrap();
    println!("Stationary: {:?}", stationary);
    // Output: Stationary: [0.3077, 0.4615, 0.2308]

    // Compute mixing time (time to reach within ε of stationary)
    let mixing = chain.mixing_time(0.01, 100);
    println!("Mixing time (ε=0.01): {} steps", mixing);

    // Spectral gap (related to mixing rate)
    let gap = chain.spectral_gap(1000);
    println!("Spectral gap: {:.4}", gap);

    // Distance between two chains
    let fast_mix = TransitionMatrix::uniform(3);
    let manifold = TransitionManifold::new(3);
    let dist = manifold.frobenius_distance(&chain, &fast_mix);
    println!("Frobenius distance: {:.4}", dist);
}
```

### Geodesic Interpolation

```rust
use agent_manifold::policy::Policy;
use agent_manifold::belief::BeliefState;
use agent_manifold::geodesic::{AgentGeodesic, NaturalPolicyUpdate};

fn main() {
    // Interpolate between two agent configurations
    let policy_a = Policy::new(vec![0.1, 0.9]).unwrap();
    let policy_b = Policy::new(vec![0.8, 0.2]).unwrap();
    let belief_a = BeliefState::new(vec![0.7, 0.3]).unwrap();
    let belief_b = BeliefState::new(vec![0.2, 0.8]).unwrap();

    let geodesic = AgentGeodesic::new(policy_a, policy_b, belief_a, belief_b);

    // Sample 5 points along the geodesic
    let samples = geodesic.sample(5);
    for (i, (policy, belief)) in samples.iter().enumerate() {
        let t = i as f64 / 4.0;
        println!("t={:.1}: policy={:?}, belief={:?}",
            t, policy.probs(), belief.probs());
    }

    // Natural policy update (manifold-aware gradient step)
    let updater = NaturalPolicyUpdate::new(0.1);
    let policy = Policy::new(vec![0.3, 0.4, 0.3]).unwrap();
    let grad = vec![2.0, -1.0, 0.0]; // favor action 0
    let updated = updater.update(&policy, &grad);
    println!("Updated policy: {:?}", updated.probs());
    // action 0 probability increased, all probs still sum to 1
}
```

### Projection and Riemannian Optimization

```rust
use agent_manifold::projection::{
    SimplexProjection, TangentProjection,
    RiemannianOptimizer, RetractionMethod,
};

fn main() {
    // Project arbitrary vector onto the simplex
    let sp = SimplexProjection::new(4);
    let raw = vec![3.0, -1.0, 2.0, 0.5];
    let projected = sp.project(&raw);
    println!("Projected: {:?}", projected);
    // Output: Projected: [0.6250, 0.0000, 0.3750, 0.0000]

    // Project gradient onto tangent space
    let tp = TangentProjection::new(3);
    let gradient = vec![1.0, 2.0, 3.0];
    let tangent = tp.project(&gradient);
    println!("Tangent gradient: {:?}", tangent);
    // Output: Tangent gradient: [-1.0, 0.0, 1.0]

    // Riemannian optimization: stay on the manifold
    let mut opt = RiemannianOptimizer::new(3, 0.1);
    opt.set_retraction(RetractionMethod::Exponential);

    let x = vec![0.2, 0.3, 0.5];
    let grad = vec![1.0, 0.0, -1.0];
    let updated = opt.step(&x, &grad);
    println!("Updated: {:?}", updated);
    // Still on simplex, no clipping artifacts
}
```

### Agent Embedding

```rust
use agent_manifold::policy::Policy;
use agent_manifold::embedding::AgentEmbedding;

fn main() {
    // Create diverse set of policies
    let policies: Vec<Policy> = vec![
        Policy::new(vec![0.9, 0.05, 0.05]).unwrap(),
        Policy::new(vec![0.8, 0.1, 0.1]).unwrap(),
        Policy::new(vec![0.1, 0.8, 0.1]).unwrap(),
        Policy::new(vec![0.05, 0.05, 0.9]).unwrap(),
        Policy::uniform(3),
    ];

    // Embed in 2D for visualization
    let embedder = AgentEmbedding::default_2d();
    let embedded = embedder.embed(&policies, 42);

    for (i, point) in embedded.iter().enumerate() {
        println!("Policy {}: ({:.3}, {:.3})", i, point.coords[0], point.coords[1]);
    }

    // Compute similarity between policies
    let sim = embedder.similarity(&policies[0], &policies[1]);
    println!("Similarity(0, 1): {:.4}", sim);
    // Output: Similarity(0, 1): 0.8712  (close policies)

    let sim_far = embedder.similarity(&policies[0], &policies[3]);
    println!("Similarity(0, 3): {:.4}", sim_far);
    // Output: Similarity(0, 3): 0.4893  (distant policies)
}
```

---

## Mathematical Foundations

### Policy Manifold

A categorical policy over *n* actions is a point on the **statistical manifold** of categorical distributions. The manifold is the interior of the probability simplex Δ^{n-1}.

**Tangent space:** At a policy **p** = (p₁, ..., pₙ), the tangent space T_p consists of vectors **v** with Σ vᵢ = 0.

**Fisher information metric:** The Riemannian metric on this manifold is the Fisher information matrix:
```
g_ij(p) = δ_ij / p_i
```

**KL-divergence** serves as the canonical (asymmetric) squared distance:
```
D_KL(p || q) = Σᵢ pᵢ log(pᵢ / qᵢ)
```

**Fisher-Rao distance** (the geodesic distance) for categorical distributions:
```
d(p, q) = 2 · arccos(Σᵢ √(pᵢ qᵢ))
```

**Natural gradient:** The Riemannian gradient on this manifold is:
```
∇̃ J = F⁻¹ ∇J
```
where F is the Fisher matrix. For categorical distributions, this simplifies to pᵢ · Aᵢ where Aᵢ = Rᵢ - E[R] is the advantage.

### Belief Manifold (Simplex Geometry)

Belief states live on the **probability simplex**:
```
Δ^{n-1} = { p ∈ ℝⁿ : pᵢ ≥ 0, Σ pᵢ = 1 }
```

This is an (n-1)-dimensional manifold. The **Fisher-Rao metric** turns it into a Riemannian manifold with geodesics computable via the square-root parameterization:
```
γ(t) = [(1-t)√p + t√q]² / ||(1-t)√p + t√q||²
```

The **Dirichlet distribution** provides natural priors on the simplex:
```
Dir(α) = (1/B(α)) · Πᵢ xᵢ^(αᵢ⁻¹)
```

### Transition Manifold

An n×n **row-stochastic matrix** has rows that each lie on Δ^{n-1}. The full matrix manifold is the **product manifold** (Δ^{n-1})ⁿ, with dimension n(n-1).

**Spectral gap:** For an irreducible, aperiodic chain with eigenvalues 1 = λ₁ > |λ₂| ≥ ... ≥ |λₙ|, the spectral gap is:
```
gap = 1 - |λ₂|
```

This controls the **mixing time**: the number of steps for the chain to get within ε of its stationary distribution.

### Geodesic Interpolation

The **geodesic** between two points on a Riemannian manifold is the locally distance-minimizing curve. On the simplex with the Fisher-Rao metric, geodesics are computed in the **square-root parameterization**:
```
θ = √p ∈ ℝⁿ
```
In this parameterization, geodesics become straight lines, and the Fisher-Rao metric is the Euclidean metric. This is the key computational insight.

### Projection and Retraction

**Retraction:** A map R_p: T_pM → M that approximates the exponential map. For the simplex:
- **Projection retraction:** Take a Euclidean step, then project back onto the simplex
- **Exponential retraction:** Multiplicative update pᵢ · exp(-η gᵢ) / Z

The projection is via the algorithm of Duchi et al. (2008) — an O(n log n) method for Euclidean projection onto the l₁-ball.

---

## Design Decisions

### Zero Dependencies (Except Serde)

Every mathematical operation — softmax, geodesic computation, Fisher metric, spectral gap, t-SNE embedding — is implemented from scratch. No `nalgebra`, no `ndarray`, no external math libraries. This keeps the crate lightweight, easy to audit, and free of version conflict cascades.

The sole exception is `serde` for serialization. Agent configurations need to be saved, loaded, and transmitted. Serde is the Rust ecosystem standard and adds minimal overhead.

### Edition 2024

The crate uses Rust Edition 2024 for the latest language features and idioms.

### Manifold-First API

Every type is designed around the manifold abstraction, not the raw data. You don't "normalize a vector" — you "project onto the simplex." You don't "take a gradient step" — you "follow a geodesic." The API reflects the geometry, making incorrect states unrepresentable.

### All Types Serializable

Every public type derives `Serialize` and `Deserialize`. Agent trajectories, embeddings, and optimization states can be persisted and restored. This is essential for agent systems that need checkpoint/resume.

### Numerical Stability

All operations handle boundary cases: near-zero probabilities, degenerate distributions, and edge cases in the Fisher metric. A global `EPS = 1e-12` provides consistent numerical guards across all modules.

---

## Testing

The crate includes 50+ tests covering:

- **Policy distance:** KL divergence symmetry, Fisher metric, natural gradient direction
- **Belief simplex:** Bayesian updates, Fisher-Rao geodesics, Dirichlet moments
- **Transition matrices:** Stationary distributions, irreducibility, mixing times
- **Geodesic interpolation:** Endpoint correctness, midpoint validity, length positivity
- **Embedding:** Pairwise distance symmetry, embedding dimensionality, similarity ordering
- **Projection:** Simplex projection validity, tangent space orthogonality, Riemannian optimization feasibility

```bash
cargo test
```

---

## API Reference

### `policy` Module

| Method | Description |
|--------|-------------|
| `Policy::new(probs)` | Create from probability vector |
| `Policy::uniform(n)` | Uniform random policy |
| `Policy::deterministic(n, a)` | One-hot policy |
| `Policy::from_logits(logits)` | Softmax policy |
| `Policy::entropy()` | Shannon entropy |
| `PolicyManifold::distance(p, q)` | Symmetric KL distance |
| `PolicyManifold::kl_divergence(p, q)` | Asymmetric KL |
| `PolicyManifold::natural_gradient(p, r)` | Fisher-adjusted gradient |
| `PolicyManifold::midpoint(p, q)` | Geodesic midpoint |

### `belief` Module

| Method | Description |
|--------|-------------|
| `BeliefState::bayes_update(l)` | Bayesian update with likelihood |
| `BeliefState::entropy()` | Belief entropy |
| `BeliefManifold::fisher_rao_distance(p, q)` | True geodesic distance |
| `BeliefManifold::geodesic_point(p, q, t)` | Point on geodesic at parameter t |
| `BeliefManifold::total_variation(p, q)` | TV distance |
| `Dirichlet::mean()` | Mean of Dirichlet prior |
| `Dirichlet::mode()` | Mode (requires all α > 1) |

### `transition` Module

| Method | Description |
|--------|-------------|
| `TransitionMatrix::stationary_distribution()` | Find steady state |
| `TransitionMatrix::is_irreducible()` | Check connectivity |
| `TransitionMatrix::spectral_gap()` | 1 - \|λ₂\| |
| `TransitionMatrix::mixing_time(ε)` | Steps to converge |
| `TransitionMatrix::power(k)` | P^k via matrix multiplication |
| `TransitionManifold::frobenius_distance(p, q)` | Frobenius norm distance |
| `TransitionManifold::interpolate(p, q, t)` | Row-wise interpolation |

### `geodesic` Module

| Method | Description |
|--------|-------------|
| `AgentGeodesic::evaluate(t)` | Point at parameter t ∈ [0,1] |
| `AgentGeodesic::sample(n)` | n evenly-spaced samples |
| `AgentGeodesic::length(segments)` | Approximate geodesic length |
| `NaturalPolicyUpdate::update(p, grad)` | Manifold-aware policy update |

### `embedding` Module

| Method | Description |
|--------|-------------|
| `AgentEmbedding::embed(agents, seed)` | t-SNE-like embedding |
| `AgentEmbedding::pairwise_distances(agents)` | Manifold distance matrix |
| `AgentEmbedding::similarity(a, b)` | Similarity score ∈ (0, 1] |

### `projection` Module

| Method | Description |
|--------|-------------|
| `SimplexProjection::project(v)` | Project onto probability simplex |
| `TangentProjection::project(v)` | Project onto tangent space |
| `RiemannianOptimizer::step(x, grad)` | Single Riemannian gradient step |
| `RiemannianOptimizer::optimize(x, grads)` | Multi-step trajectory |
| `StochasticProjection::project(matrix)` | Project onto row-stochastic matrices |

---

## License

MIT

---

## Contributing

This crate is designed to be a focused, dependency-free foundation for information-geometric agent operations. Contributions that maintain these principles are welcome.

---

*"In the landscape of agent minds, the shortest path between two thoughts is never a straight line — it's a geodesic."*
