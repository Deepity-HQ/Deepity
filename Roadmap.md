===============================================================================
ROADMAP TO A MODERN HIGH-PERFORMANCE PREDICTIVE CODING NETWORK
===============================================================================

Goal:
Bring the current discriminative PCN closer to the current research frontier
without fundamentally changing its architecture.

-------------------------------------------------------------------------------
1. PRECISION-WEIGHTED ERRORS (Highest Priority)
-------------------------------------------------------------------------------

Research motivation:
Predictive coding is fundamentally Bayesian. Prediction errors should be weighted
by their precision (inverse variance), allowing the model to trust reliable
signals more than noisy ones.

Current:
    e = z - μ

Target:
    e = Λ(z - μ)

where Λ is the precision.

Implementation:

Start simple:
    float layerPrecision;

Later:
    float* neuronPrecision;

Modify CalculateState():

Current:
    e = z - mu

New:
    e = (z - mu)
    cblas_sscal(N, layerPrecision, e, 1);

Eventually:
    e[i] *= precision[i];

Also multiply precision into:

    UpdateState()
    UpdateWeights()

so all inference and learning become precision-weighted.

Research:
- Friston's predictive processing / free-energy framework
- 2024 survey:
  "Predictive Coding Networks and Applications: A Survey"
  https://arxiv.org/abs/2407.04117

-------------------------------------------------------------------------------
2. ADAPTIVE CONVERGENCE (Very High Priority)
-------------------------------------------------------------------------------

Research motivation:
Most PCNs use a fixed number of inference iterations purely for convenience.

Your code already computes energy.

Use it.

Current:

for(i=0;i<10;i++)
{
    CalculateState();
    UpdateState();
}

New:

previousEnergy = INF;

while(true)
{
    energy = CalculateState();
    UpdateState();

    if(abs(previousEnergy-energy)<epsilon)
        break;

    previousEnergy = energy;
}

Benefits:

• Easy samples converge quickly.
• Hard samples receive more computation.
• Eliminates wasted inference.

-------------------------------------------------------------------------------
3. FEEDFORWARD INITIALIZATION (Amortized Inference)
-------------------------------------------------------------------------------

Research motivation:

Zero initialization wastes inference steps.

Instead initialize latent states using one feedforward pass.

Current:

ResetState()

↓

all z = 0

↓

many inference iterations

New:

Feedforward()

↓

initialize every z

↓

few inference iterations

Implementation:

Replace ResetState() with something like

InitializeFeedforward()

which simply computes

z1 = f(W0x)

z2 = f(W1z1)

...

before iterative inference begins.

Research:
Predictive coding + amortized inference literature
Variational inference literature

-------------------------------------------------------------------------------
4. ERROR NORMALIZATION
-------------------------------------------------------------------------------

Research motivation:

Deep PCNs become unstable because prediction errors have different magnitudes.

Normalize errors before using them.

Current:

e = z - μ

New:

e = Normalize(z - μ)

Implementation ideas:

LayerNorm(e)

or

RMSNorm(e)

or

e /= RMS(e)

Location:

Immediately after CalculateState computes e.

Research:

Deep predictive coding papers
μPC papers
Recent OpenReview work on stable deep PCNs

-------------------------------------------------------------------------------
5. LEARNABLE INFERENCE RATE
-------------------------------------------------------------------------------

Research motivation:

Every layer converges differently.

Current:

z += ir * dz

New:

z += ir[layer] * dz

Eventually:

z += ir[i] * dz

Implementation:

Replace

float ir;

with

std::vector<float> layerIR;

or

float* neuronIR;

Very localized code change.

-------------------------------------------------------------------------------
6. RESIDUAL PREDICTIONS
-------------------------------------------------------------------------------

Research motivation:

Residual connections stabilize very deep networks.

Current:

μ = Wz + b

New:

μ = Wz + b + Residual(z)

Initially:

Residual(z)=z

Later:

Residual(z)=Linear(z)

or

Residual(z)=SmallMLP(z)

Location:

CalculateState()

after SGEMM

before activation

-------------------------------------------------------------------------------
7. SPARSE PREDICTIONS / ATTENTION
-------------------------------------------------------------------------------

Research motivation:

Prediction need not be fully connected.

Instead of

μ = Wz

Eventually:

μ = SparseLinear(z)

or

μ = Attention(z)

Benefits:

• Less unnecessary computation
• Better scalability
• Better long-range dependencies

Requires larger architectural changes.

Low priority until dense implementation is mature.

-------------------------------------------------------------------------------
8. PERSISTENT MEMORY
-------------------------------------------------------------------------------

Research motivation:

Current PCNs only predict using the current latent state.

Future:

prediction =
    f(current_state,
      retrieved_memory)

Memory options:

• episodic memory
• semantic memory
• key-value memory
• vector retrieval

Useful for:

• continual learning
• long-term consistency
• AGI architectures

Not required for image classification.

-------------------------------------------------------------------------------
9. MULTI-TIMESCALE LATENT STATES
-------------------------------------------------------------------------------

Research motivation:

Not every layer should evolve at the same speed.

Current:

single inference rate

Future:

bottom layers:
    fast

middle layers:
    medium

top layers:
    slow

Implementation:

Different ir values per layer.

Simple change.

Potentially improves stability.

-------------------------------------------------------------------------------
IMPLEMENTATION ORDER
-------------------------------------------------------------------------------

1. Adaptive convergence
2. Feedforward initialization
3. Precision weighting
4. Error normalization
5. Layer-wise inference rates
6. Residual predictions
7. Sparse prediction / attention
8. Persistent memory

-------------------------------------------------------------------------------
MAIN REFERENCES
-------------------------------------------------------------------------------

Friston, K.
"The free-energy principle"
(and related predictive processing papers)

Millidge et al.
"Predictive Coding: A Theoretical and Experimental Review"

2024:
"Predictive Coding Networks and Applications: A Survey"
https://arxiv.org/abs/2407.04117

Recent work on deep/stable PCNs (e.g., μPC and related parameterizations):
- https://openreview.net/forum?id=FwdN0KovFp
- Search terms: "μPC predictive coding", "stable deep predictive coding",
  "deep predictive coding networks"

===============================================================================