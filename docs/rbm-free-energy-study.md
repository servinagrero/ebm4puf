---
icon: lucide/activity
---

# RBM free-energy study

This document describes the Restricted Boltzmann Machine (RBM) experiments
implemented in `scripts/analysis_free_energy.py` and visualized by
`scripts/plot_free_energy.py`. It records the data transformations, models,
test scenarios, metrics, output artifacts, and interpretation required to
reproduce the study.

The study has two related objectives:

1. Determine whether an RBM can generate synthetic PUF bit-flip vectors that
   preserve marginal reliability and contemporaneous dependencies.
2. Determine whether RBM free energy can authenticate genuine readouts or
   detect readouts that do not follow the enrolled device's reliability
   structure.

The current results indicate that the RBM is more successful as a
dependency-sensitive discriminator than as a faithful generator.

## Data model

Each PUF family in `data/puf_artifact.h5` contains:

- A reference response matrix \(R\) with shape \(D \times B\).
- A readout tensor \(X\) with shape \(D \times S \times B\).

Here:

- \(D\) is the number of devices.
- \(S\) is the number of repeated readouts per device.
- \(B=256\) is the number of response bits.

The number of devices and readouts can differ between PUF families.

Only room-temperature datasets are included by default. The current
high-readout study focuses on:

- `SRAM_XMC4500`
- `RO_Spartan7_roomT`
- `RO_Maiti_Spartan3E`

## Bit-flip representation

The RBM is trained on bit-flip vectors rather than raw PUF responses.

For device \(d\), sample \(s\), and bit \(i\), the error matrix is

\[
E_{d,s,i} = X_{d,s,i} \oplus R_{d,i}.
\]

Therefore:

- \(E_{d,s,i}=1\) denotes a bit flip.
- \(E_{d,s,i}=0\) denotes agreement with the reference response.

For one device, the resulting matrix has shape \(S \times B\). Every device
receives its own RBM.

The default split is:

- 60% enrollment/training readouts.
- 20% calibration readouts.
- 20% held-out test readouts.

The split is deterministic for a fixed seed. Training, calibration, and test
readouts remain separate throughout the experiment.

!!! warning "Limited held-out data"

    A family with 100 readouts per device has only about 20 calibration and
    20 test vectors under the default split. Sequence-level results obtained
    by bootstrap resampling estimate repeated draws from these empirical
    distributions. They are not equivalent to collecting many independent
    groups of new hardware measurements.

## Marginal reliability model

For a device, the training marginal bit-flip probability is

\[
\hat p_i = \frac{1}{S_{\mathrm{train}}}
           \sum_{s=1}^{S_{\mathrm{train}}} E_{s,i}.
\]

The independent Bernoulli model assigns a bit-flip vector \(e\) the log
probability

\[
\log p_{\mathrm{ind}}(e)
=
\sum_{i=1}^{B}
\left[
e_i \log \hat p_i
+ (1-e_i)\log(1-\hat p_i)
\right].
\]

The implementation clips the probabilities to
\([10^{-6}, 1-10^{-6}]\) before evaluating the logarithms.

This model preserves the bit-wise marginal reliability but assumes that all
bits flip independently.

## Rare-error-aware marginal model

The model-free error-structure analysis in
`scripts/analysis_error_structure.py` uses a different marginal estimator.
This estimator is not yet used by the RBM authentication experiment.

Applying Jeffreys smoothing independently to every bit gives

\[
\hat p_{d,i}^{\mathrm{Jeffreys}}
=
\frac{x_{d,i}+1/2}{n_d+1}.
\]

For (B) bits, its implied mean error count is

\[
E_0[K]
=
\frac{n_d\overline K_d+B/2}{n_d+1}.
\]

When (B=256), (n_d\approx80), and the device is reliable, the (B/2=128)
pseudocount contribution can add approximately 1.5 errors to the predicted
mean count. The error-structure analysis therefore uses a rare-error-aware
leave-one-device-out family prior.

Let (X_{f,i}^{(-d)}) be the number of errors at bit (i) among all family
devices except target device (d), and let (N_f^{(-d)}) be their number of
readouts. The family BER is

\[
\overline p_f^{(-d)}
=
\frac{\sum_i X_{f,i}^{(-d)}}{B N_f^{(-d)}}.
\]

The bit-specific prior is

\[
\mu_{f,i}^{(-d)}
=
\frac{X_{f,i}^{(-d)}+\eta\overline p_f^{(-d)}}
{N_f^{(-d)}+\eta},
\]

with default prior strength \(\eta=1\). A small numerical floor keeps the
probabilities strictly between zero and one. Because the target device is
excluded, this prior measures transferable family information rather than
leaking the target readouts into their own null model.

For a target-device enrollment fold with (n) readouts and error count
(x_{d,i}), the shrunk marginal is

\[
\widetilde p_{d,i}(\lambda)
=
\frac{x_{d,i}+\lambda\mu_{f,i}^{(-d)}}{n+\lambda}.
\]

The shrinkage strength is selected from a configured grid using
leave-one-readout-out predictive likelihood within the enrollment fold. The
outer held-out fold is not used for selecting \(\lambda\). A small selected
value indicates device-specific behavior; a large value indicates that the
family bit profile predicts the device well.

The resulting per-device independent count model is

\[
p_{0,d}(K=k)
=
\operatorname{PB}
\left(k;\widetilde p_{d,1},\ldots,\widetilde p_{d,B}\right).
\]

The family null is the equal-device mixture

\[
p_{0,f}^{\mathrm{mixture}}(k)
=
\frac{1}{D_f}\sum_d p_{0,d}(K=k),
\]

not a single Poisson-binomial distribution fitted to globally averaged
marginals. The latter describes a hypothetical average device and erases
between-device variability.

## Count and conditional error structure

Let

\[
K(E)=\sum_i E_i.
\]

The error-vector distribution factorizes exactly as

\[
p(E)=p(K)\,p(E\mid K).
\]

The error-count and ECC-capacity results describe (p(K)). The conditional
residual heatmaps describe (p(E\mid K)). For device (d), count (k), and
bit (i), the residual is

\[
R_{d,k,i}
=
P_d(E_i=1\mid K=k)
-P_{0,d}(E_i=1\mid K=k).
\]

Every valid row satisfies

\[
\sum_iR_{d,k,i}=0.
\]

Consequently, conditional residuals redistribute error locations within a
fixed count but cannot change (p(K)). Their mean, inter-device IQR, and
support cannot be convolved into an error-count distribution. They instead
measure shared conditional structure, device heterogeneity plus sampling
noise, and the device coverage of each count stratum.

For a bounded-distance code that corrects every pattern with
(K\leq\tau\), only the tail

\[
P(K>\tau)
\]

determines failure. Conditional structure becomes relevant when decoding
performance depends on error location or higher-order patterns.

### Reference-polarity-conditioned error structure

The reference response determines the physical transition represented by a
flip. For device (d), error count (k), and reference polarity
(b\in\{0,1\}), define

\[
q^{\mathrm{emp}}_{d,k,b}
=
\frac{1}{|I_{d,b}|}
\sum_{i\in I_{d,b}}
P_d(E_i=1\mid K=k),
\qquad
I_{d,b}=\{i:r_{d,i}=b\}.
\]

Thus, (q_{d,k,0}) is the conditional (0\to1) flip probability and
(q_{d,k,1}) is the conditional (1\to0) flip probability. The same
average is computed from the independent conditional null to obtain
(q^{\mathrm{ind}}_{d,k,b}). The raw and residual polarity contrasts are

\[
\Delta^{\mathrm{raw}}_{d,k}
=q^{\mathrm{emp}}_{d,k,1}-q^{\mathrm{emp}}_{d,k,0},
\]

\[
\Delta^{\mathrm{res}}_{d,k}
=
\Delta^{\mathrm{raw}}_{d,k}
-
\left(q^{\mathrm{ind}}_{d,k,1}-q^{\mathrm{ind}}_{d,k,0}\right).
\]

A positive contrast indicates more (1\to0) than (0\to1) transitions;
a negative contrast indicates the reverse. The residual contrast asks
whether this directionality remains after accounting for the enrolled
per-bit error probabilities. It therefore measures more than a difference
between the marginal reliability of reference-zero and reference-one bits.

Family summaries weight each supported count stratum by its empirical
readout frequency and then give every device equal weight. Device bootstrap
intervals quantify uncertainty across the enrolled population. A
within-device permutation test randomly reallocates the reference labels
while preserving each device's number of zeros and ones; it tests whether
the observed residual landscape is unusually aligned with reference
polarity. Its resolution is limited by the configured number of
permutations.

Reference-polarity asymmetry does not change the hard bounded-distance
guarantee, which still depends only on (K). It can nevertheless inform
asymmetric-channel models, soft-decision decoding, reference-aware bit
selection, and a reference-conditioned baseline for a generative model.

## Random-device ECC metrics

Selecting a device uniformly and then selecting one of its readouts gives
the family survival function

\[
S_f(k)
=
P(K\geq k\mid f)
=
\frac{1}{D_f}\sum_dP(K\geq k\mid d).
\]

For correction capacity \(\tau\), the random-device/readout failure
probability is

\[
P_{\mathrm{fail},f}(\tau)=S_f(\tau+1).
\]

The mean population risk must be reported together with device-risk
quantiles. A low family mean can otherwise hide a minority of devices with
large failure probabilities.

For target failure probability \(\epsilon\), device coverage is

\[
C_f(\tau,\epsilon)
=
\frac{1}{D_f}\sum_d
\mathbf 1\left[P_d(K>\tau)\leq\epsilon\right].
\]

This answers what fraction of devices meets the risk target at a given
correction capacity. The analysis records empirical and rare-error
independent-model coverage.

The observed maximum

\[
K_{\max}^{\mathrm{observed}}=\max_{d,s}K_{d,s}
\]

is also recorded, but it is not an intrinsic family bound: it can only stay
constant or increase as devices or readouts are added. Predictive tail
probabilities, capacity quantiles, and device coverage provide more stable
design quantities.

Nested bootstrap intervals resample devices and then readouts within each
selected device. They describe uncertainty from the observed device
population and finite readout sets; they do not cover unmeasured operating
conditions. With approximately 100 readouts per device, empirical failure
probabilities much below (10^{-2}) are not resolvable.

## RBM model

The binary RBM contains visible units \(v \in \{0,1\}^B\), hidden units
\(h \in \{0,1\}^H\), visible biases \(b\), hidden biases \(c\), and weights
\(W\).

After marginalizing the hidden units, the free energy of a visible vector is

\[
F_\theta(v)
=
-b^\mathsf{T}v
-\sum_{j=1}^{H}
\log\left(1+\exp\left(c_j + v^\mathsf{T}W_{:,j}\right)\right).
\]

The unnormalized RBM log-probability score is

\[
s_{\mathrm{RBM}}(v) = -F_\theta(v).
\]

Higher scores indicate states preferred by the RBM.

### Training configurations

The current experiment compares:

- Contrastive divergence: CD-1, CD-5, and CD-10.
- Persistent contrastive divergence: PCD-1.

The visible biases are initialized from the empirical training marginals:

\[
b_i^{(0)}
=
\operatorname{logit}
\left(
\operatorname{clip}(\hat p_i,10^{-4},1-10^{-4})
\right).
\]

This initialization places the model near the correct marginal BER before it
learns interactions. The optimizer is still allowed to update the visible
biases during training.

## Residual dependency score

The residual score compares the RBM score with the independent Bernoulli
baseline:

\[
r_\theta(v)
=
s_{\mathrm{RBM}}(v)-\log p_{\mathrm{ind}}(v).
\]

The residual removes much of the contribution explained by marginal
reliability. It is intended to expose dependency structure learned by the
RBM.

The score should be interpreted as an unnormalized log-evidence ratio. The
unknown RBM partition-function constant does not affect sample ranking,
AUROC, or thresholds calibrated using genuine data from the same model.

The raw and residual scores answer different questions:

| Score | Primary question |
|---|---|
| \(\log p_{\mathrm{ind}}(v)\) | Are the bit-wise marginal reliabilities plausible? |
| \(s_{\mathrm{RBM}}(v)\) | Does the complete RBM assign a high score to the vector? |
| \(r_\theta(v)\) | Is the vector more RBM-like than its marginal behavior alone explains? |

The residual is especially meaningful for alternatives that preserve the
same marginals. It is not a universal replacement for the raw score.

## Test scenarios

Every alternative scenario is evaluated against genuine held-out bit-flip
vectors.

### Genuine empirical

`genuine`

Held-out bit-flip vectors from the enrolled device. These vectors are not
used for RBM training or threshold calibration.

### Independent, same marginals

`marginal`

Each bit is sampled independently using the enrolled device's training
marginal:

\[
\tilde E_i \sim \operatorname{Bernoulli}(\hat p_i).
\]

This is the most important null model for testing contemporaneous
dependencies. It preserves marginal reliability while destroying joint
bit-flip structure.

### Independent, same mean BER

`mean_ber`

A single device-level mean BER is calculated:

\[
\bar p = \frac{1}{B}\sum_{i=1}^{B}\hat p_i.
\]

Every bit is then sampled independently:

\[
\tilde E_i \sim \operatorname{Bernoulli}(\bar p).
\]

This preserves only the average BER. It destroys both bit-specific
reliability and dependencies.

### Independent, \(p=0.5\)

`uniform`

Every bit is sampled independently with probability 0.5:

\[
\tilde E_i \sim \operatorname{Bernoulli}(0.5).
\]

This represents an unstructured random vector and is a deliberately strong
marginal mismatch for reliable PUFs.

### Fixed bit permutation

`permuted`

A held-out genuine bit-flip vector is transformed by a single fixed random
permutation of its bit positions:

\[
\tilde E_{s,i} = E_{s,\pi(i)}.
\]

The same permutation is applied to every sample. This preserves the number
of bit flips in each readout but destroys alignment between learned bit
positions and their reliability/dependency structure.

### Cross-device impostor

`cross_device`

Responses from another device in the same family are XORed with the enrolled
device's reference response:

\[
\tilde E_{s}
=
X_{d',s} \oplus R_d,
\qquad d' \ne d.
\]

This models a response presented as if it came from the enrolled device. It
contains differences caused by both inter-device response variation and
readout noise.

### RBM-generated

`rbm`

Synthetic bit-flip vectors generated by alternating block Gibbs sampling:

\[
h \sim p_\theta(h\mid v),
\qquad
v' \sim p_\theta(v\mid h).
\]

The study records checkpoints after 1, 10, 50, and 100 Gibbs steps.

## Generation initializations

Each trained RBM is sampled using three initial visible-state distributions.

### Empirical initialization

Initial states are sampled with replacement from the device's training
bit-flip vectors.

### Marginal initialization

Initial states are sampled independently from the training marginals:

\[
v_i^{(0)} \sim \operatorname{Bernoulli}(\hat p_i).
\]

### Uniform initialization

Initial states are sampled independently with \(p=0.5\):

\[
v_i^{(0)} \sim \operatorname{Bernoulli}(0.5).
\]

The three chains use common random numbers. Differences between their
outputs therefore provide a coupling diagnostic: convergence to identical
results indicates that the chains have forgotten their initial states under
the shared stochastic updates.

In the current high-readout study, initialization has little effect after
one Gibbs step and no measurable effect on the recorded aggregate metrics
after ten steps.

## Sequence evidence

Authentication can combine \(N\) readouts by adding their scores:

\[
S_N = \sum_{n=1}^{N}s_{\mathrm{RBM}}(E_n),
\]

\[
L_N = \sum_{n=1}^{N}r_\theta(E_n).
\]

The analysis uses bootstrap sampling with replacement to construct repeated
groups of size \(N\). The current `run_all.sh` evaluates

\[
N \in \{1,2,4,6,8,10,12,16,20\}.
\]

The normalized score

\[
\frac{S_N}{N}
\]

is also recorded. It distinguishes additive accumulation from a change in
the average per-readout score.

## Threshold calibration

Thresholds are calculated exclusively from genuine calibration readouts.

For a configured false-rejection rate \(\alpha_{\mathrm{FR}}\), the threshold
is the lower calibration quantile:

\[
\tau_N
=
Q_{\alpha_{\mathrm{FR}}}
\left(S_N^{\mathrm{calibration}}\right).
\]

The same procedure is applied to residual evidence. A sample group is
accepted when its score is at least the corresponding threshold.

The default false-rejection rate is 0.05.

For an alternative scenario, the reported acceptance rate is

\[
\widehat{\operatorname{FAR}}
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathbf{1}\left[S_N^{(m)} \ge \tau_N\right].
\]

## AUROC

AUROC is calculated with genuine groups labeled as positive and alternative
groups labeled as negative.

| AUROC | Interpretation |
|---:|---|
| Near 1 | Genuine samples receive higher scores and are strongly separated. |
| Near 0.5 | The score does not distinguish the two populations. |
| Near 0 | The populations are strongly separated, but the alternative receives higher scores. |

An AUROC near zero is not inherently a lack of information. It indicates a
reversed score direction. Nevertheless, an authentication rule that accepts
high scores will fail unless that reversal is explicitly handled.

The orientation-independent diagnostic

\[
A_{\mathrm{sep}} = 2\left|\operatorname{AUROC}-0.5\right|
\]

can describe separation strength, but it must not replace the signed AUROC
or calibrated acceptance rate when evaluating an operational detector.

## Generation fidelity metrics

### Marginal MAE

\[
\operatorname{MAE}_{p}
=
\frac{1}{B}
\sum_{i=1}^{B}
\left|
\hat p_i^{\mathrm{real}}-\hat p_i^{\mathrm{generated}}
\right|.
\]

Lower values indicate better marginal reliability reproduction.

### Error-count distribution

For each vector,

\[
C(E_s)=\sum_{i=1}^{B}E_{s,i}
\]

is the number of bit flips in one readout.

The empirical real and generated histograms are compared with the exact
Poisson-binomial distribution implied by the independent bit marginals.

The recorded CDF distance is

\[
D_{\mathrm{CDF}}
=
\max_c
\left|
F_{\mathrm{real}}(c)-F_{\mathrm{generated}}(c)
\right|.
\]

Lower values indicate a better error-count match.

### Covariance and precision fidelity

Ledoit-Wolf shrinkage estimates are fitted to real and generated data. For a
matrix \(A\), the relative Frobenius error is

\[
D_F
=
\frac{\left\|A_{\mathrm{real}}-A_{\mathrm{generated}}\right\|_F}
       {\max\left(\left\|A_{\mathrm{real}}\right\|_F,\epsilon\right)}.
\]

The Pearson correlation between off-diagonal matrix entries is also
recorded.

!!! caution "High-dimensional uncertainty"

    The test split can contain approximately 20 vectors for 256 bits.
    Covariance and precision fidelity estimates are therefore underpowered,
    even with shrinkage. They should be interpreted together with
    independent-null controls and bootstrap uncertainty.

## Relaxation diagnostics

The analysis applies additional Gibbs transitions to genuine, alternative,
and generated inputs. It records:

- Initial and relaxed RBM scores.
- Mean and median score gain.
- Fraction of changed bits.
- Reconstruction binary cross-entropy.
- Fraction of vectors left unchanged.

These diagnostics test whether the RBM maps different initial populations
toward the same region and whether genuine vectors are already close to
states favored by the model.

## Output artifacts

The analysis writes files under `data/rbm_free_energy/`.

| File | Contents |
|---|---|
| `authentication.csv` | Raw and residual authentication results for genuine and non-RBM alternatives. |
| `generation_sweep.csv` | Authentication and fidelity results for RBM-generated vectors. |
| `fidelity.csv` | Marginal, error-count, covariance, and precision metrics. |
| `relaxation.csv` | Gibbs relaxation and reconstruction diagnostics. |
| `error_count_distributions.csv` | Empirical, Poisson-binomial, and RBM error-count PMFs. |
| `score_statistics.csv` | Per-readout moments and quantiles for the independent, raw RBM, and residual scores. |
| `score_tests.csv` | Empirical and Gaussian-predicted AUROC as a function of \(N\). |
| `required_readouts.csv` | Gaussian estimates of the \(N\) required for each target directional AUROC. |
| `joint_score_statistics.csv` | Paired marginal/residual means, variances, covariance, and correlation. |
| `joint_score_tests.csv` | Genuine-calibrated marginal, residual, and joint Mahalanobis tests. |
| `required_readouts_summary.csv` | Device-level directional coverage and conditional required-\(N\) quartiles. |
| `<family>/device_<id>_<method>_k<k>.pt` | Trained RBM state dictionaries. |

The model-free analysis writes files under `data/error_structure/`.

| File | Contents |
|---|---|
| `count_distributions.parquet` | Empirical and cross-fitted rare-error Poisson-binomial PMFs per device. |
| `conditional_probabilities.parquet` | Empirical and independent (P(E_i=1\mid K=k)), conditional residuals, and support. |
| `marginal_estimates.parquet` | Family prior, selected shrinkage, and device marginal probability per fold and bit. |
| `device_metrics.parquet` | Count, conditional, shrinkage, and family-prior summaries per device. |
| `device_tail_risks.parquet` | Empirical and independent (P_d(K>\tau)) for every device and capacity. |
| `population_survival.parquet` | Random-device mean tails, device quantiles, bootstrap intervals, and observed maximum. |
| `device_coverage.parquet` | Fraction of devices meeting each configured failure target, including its finite-sample failure-count threshold. |
| `predictive_capacities.csv` | Population and device-coverage capacities for configured risk targets. |
| `family_summary.csv` | Polars-generated family quartiles of device-level metrics. |
| `reference_polarity_conditionals.parquet` | Per-device and per-count empirical and independent transition rates, polarity contrasts, residual contrasts, and support. |
| `reference_polarity_device_summary.parquet` | Support-weighted raw and residual polarity contrasts for every device. |
| `reference_polarity_family_summary.csv` | Equal-device family contrasts, bootstrap intervals, and reference-label permutation tests. |

Each CSV identifies the relevant experimental configuration using fields
such as:

- `family`
- `device`
- `training_method`
- `cd_k`
- `gibbs_steps`
- `visible_bias_init`
- `initialization`
- `group_size`

## Main plots

Figures are written under `figures/rbm_free_energy/`.

### Authentication AUROC

`authentication_auc/<family>_<method>_k<k>.png`

The left panel shows raw RBM-score AUROC. The right panel shows residual
RBM-versus-marginal AUROC.

The most important comparison is `Independent, same marginals`:

- Raw AUROC tests whether the full energy score separates the populations.
- Residual AUROC tests whether the RBM learned dependencies beyond the
  enrolled marginal reliability.
- Increasing separation with \(N\) indicates accumulating per-readout
  evidence.

Gross marginal attacks can produce raw AUROC near one and residual AUROC
near zero. This is score reversal caused by the independent denominator and
demonstrates why raw and residual channels are complementary.

### Empirical versus Gaussian AUROC

`gaussian_auc/<family>_<method>_k<k>_<scenario>.png`

For one of the three score channels, let \(X_g=s(E_g)\) and
\(X_a=s(E_a)\) denote the score of one genuine and one alternative
readout. Write their means and variances as

\[
(\mu_g,\sigma_g^2)
\quad\text{and}\quad
(\mu_a,\sigma_a^2).
\]

A decision using \(N\) readouts compares the additive sequence scores

\[
S_{g,N}=\sum_{n=1}^{N}X_{g,n},
\qquad
S_{a,N}=\sum_{n=1}^{N}X_{a,n}.
\]

Under the IID, finite-variance approximation, the central limit theorem
gives

\[
S_{g,N}\approx\mathcal N(N\mu_g,N\sigma_g^2),
\qquad
S_{a,N}\approx\mathcal N(N\mu_a,N\sigma_a^2).
\]

Signed AUROC is the probability that an independently selected genuine
sequence receives a higher score than an alternative sequence:

\[
\operatorname{AUROC}_N
=
P(S_{g,N}>S_{a,N}).
\]

The difference \(S_{g,N}-S_{a,N}\) is approximately Gaussian with mean
\(N(\mu_g-\mu_a)\) and variance
\(N(\sigma_g^2+\sigma_a^2)\). Therefore,

\[
\operatorname{AUROC}_N
\approx
\Phi\left(
\frac{\sqrt{N}(\mu_g-\mu_a)}
{\sqrt{\sigma_g^2+\sigma_a^2}}
\right)
\]

where \(\Phi\) is the standard-normal CDF. The signed per-readout effect
size is

\[
d=
\frac{\mu_g-\mu_a}
{\sqrt{\sigma_g^2+\sigma_a^2}},
\]

so the sequence-level separation grows as \(\sqrt{N}d\). A positive
effect approaches AUROC one, a negative effect approaches AUROC zero,
and a zero effect remains at AUROC one half.

The figures compare this prediction with AUROC calculated from bootstrap
sums for the independent marginal, raw RBM, and residual channels.
Because the bootstrap samples per-readout scores independently with
replacement, close agreement validates the additive Gaussian
approximation for the observed empirical score distributions. It does
not by itself establish temporal independence of physical readouts or
generalization to unseen devices. Systematic disagreement can indicate
skewness, finite-sample effects, temporal dependence, nonstationarity, or
device heterogeneity.

### Required readouts

`required_readouts/<family>_<method>_k<k>_<scenario>.png`

These figures show the Gaussian estimate of the number of readouts required
to reach each target directional AUROC for the matched-marginal scenario.
Each score has a separate panel on a shared logarithmic vertical scale.
Faint lines represent individual directional devices, while the thick line
is the conditional median. The panel title reports how many evaluated
devices have the genuine-high direction.

A non-directional device has
\(\mu_{\mathrm{genuine}}\leq\mu_{\mathrm{alternative}}\), so increasing
\(N\) cannot reach the requested AUROC without reversing the decision rule.
Conditional required-\(N\) values must therefore always be interpreted
together with directional coverage.

### Joint tampering detection

`joint_detection/<family>_<method>_k<k>_auc.png`

`joint_detection/<family>_<method>_k<k>_far.png`

The three panels compare genuine-centered marginal-only, residual-only,
and joint Mahalanobis anomaly scores. Their centers and covariance are
estimated exclusively from genuine calibration readouts. The joint score
retains the marginal and residual channels separately:

\[
Y_N =
\begin{bmatrix}
M_N\\
R_N
\end{bmatrix}.
\]

It then measures distance from the genuine calibration distribution:

\[
D_N^2
=
(Y_N-\hat\mu_{g,N})^\mathsf{T}
\hat\Sigma_{g,N}^{-1}
(Y_N-\hat\mu_{g,N}).
\]

The plotted genuine score is \(-D_N^2\), so higher values indicate greater
consistency with enrollment. Unlike the arithmetic sum \(M_N+R_N\), which
is the raw RBM score, the joint test does not allow one channel to cancel
an anomaly in the other.

The AUROC figures measure threshold-independent separation. The FAR figures
show alternative acceptance at the threshold chosen for the configured
genuine calibration false-rejection rate.

The two-sided Mahalanobis test is retained as a low-enrollment baseline.
With only approximately 20 calibration readouts, its empirical calibration
region can become too narrow as \(N\) increases. In the present experiment,
matched-marginal vectors can consequently lie closer to the calibration
center than held-out genuine vectors.

The directional figures add a predictive low-enrollment test:

`joint_detection/<family>_<method>_k<k>_directional_auc.png`

`joint_detection/<family>_<method>_k<k>_directional_far.png`

For a channel \(X\in\{M,R\}\), the standardized sequence score is

\[
Z_{X,N}
=
\frac{X_N-N\hat\mu_X}
{\hat\sigma_X\sqrt{N+N^2/n_{\mathrm{cal}}}}.
\]

The \(N^2/n_{\mathrm{cal}}\) term accounts for uncertainty in the
calibration mean. The directional joint score is

\[
Z_{\mathrm{joint},N}
=
\min(Z_{M,N},Z_{R,N}).
\]

It preserves the genuine-high direction of both channels: gross marginal
changes are rejected by \(Z_M\), while matched-marginal dependency changes
are rejected by \(Z_R\). The joint lower-tail threshold uses a Bonferroni
allocation of the configured false-rejection rate. This is conservative
and relies on the approximate Gaussian predictive model; held-out genuine
false-rejection rates must still be reported.

### Generation AUROC

`generation_sweep/<family>_<initialization>_auc.png`

These plots compare genuine and RBM-generated vectors.

For faithful generation, both signed AUROCs should approach 0.5. An AUROC
near either zero or one means that a discriminator can still distinguish
generated vectors from genuine vectors.

Separate files for empirical, marginal, and uniform initialization show
whether generation quality depends on the initial state.

### Error-count distributions

`error_count_distributions/<family>_<method>_k<k>_g<gibbs>_<initialization>.png`

These plots overlay:

- The empirical error-count histogram.
- The RBM-generated histogram.
- The independent Poisson-binomial PMF.

A matching histogram demonstrates fidelity of the total number of bit
flips, but it does not demonstrate dependency fidelity. Residual evidence
or another joint-distribution test is still required.

### Centered evidence

`centered_evidence/<family>_<method>_k<k>_<initialization>.png`

Residual quantiles are centered using the matched-marginal median at each
\(N\):

\[
\widetilde L_N
=
L_N
-\operatorname{median}\left(L_N^{\mathrm{marginal}}\right).
\]

Values above zero indicate more RBM-specific structure than the independent
matched-marginal baseline. Genuine and generated curves should overlap if
the generator reproduces the learned dependencies.

### Normalized scores

`normalized_scores/<family>_<method>_k<k>_<initialization>.png`

These plots show \(S_N/N\). A stable curve indicates that the total evidence
grows additively with the number of readouts.

### Relaxation

`relaxation/<family>_<method>_k<k>_transition_hamming.png`

These plots show the mean Hamming fraction between consecutive Gibbs
states at each requested checkpoint:

\[
H_k
=
\frac{1}{B}d_H(v_k,v_{k-1}).
\]

The checkpoints \(1,5,10\) therefore compare \(v_1\) with \(v_0\),
\(v_5\) with \(v_4\), and \(v_{10}\) with \(v_9\). This measures local
chain activity rather than cumulative displacement from the initial
vector. A stable transition Hamming fraction does not by itself establish
mixing; autocorrelation and effective-sample-size diagnostics would still
be required.

## Current empirical findings

The following observations apply to the present high-readout experiment and
should not be treated as universal properties of the PUF families.

### Dependency discrimination

For `SRAM_XMC4500`, the residual score strongly distinguishes genuine
readouts from independent vectors with the same marginals. At \(N=20\), the
median residual AUROC across the evaluated devices is approximately 0.99
for CD-1, CD-5, CD-10, and PCD-1.

The corresponding raw-score AUROC is below 0.2. The independent vectors can
receive higher raw RBM scores than genuine vectors, while the residual
correctly exposes their missing dependencies.

The RO families show weaker and more device-dependent residual separation.
No training configuration dominates across all families:

- Higher CD performs well for `RO_Maiti_Spartan3E`.
- CD-1 performs better for `RO_Spartan7_roomT`.
- `SRAM_XMC4500` is comparatively insensitive to CD or PCD choice.

### Additive evidence

The genuine-minus-independent mean residual grows approximately linearly
with \(N\). Under CD-10, its approximate median value per readout is:

| Family | Residual advantage per readout |
|---|---:|
| `SRAM_XMC4500` | 0.92 |
| `RO_Maiti_Spartan3E` | 0.18 |
| `RO_Spartan7_roomT` | 0.01 |

These quantities provide an empirical measure of family-specific dependency
signal strength.

### Generation

The three generation initializations converge to effectively identical
aggregate metrics by ten Gibbs steps. Initialization is therefore not the
main generation bottleneck.

Generated vectors reproduce marginal BER and error-count distributions much
better than they reproduce residual dependency evidence. For CD-10 and
\(N=20\), the approximate median centered evidence is:

| Family | Genuine | Independent | RBM-generated |
|---|---:|---:|---:|
| `SRAM_XMC4500` | 17.8 | 0.0 | 0.8 |
| `RO_Maiti_Spartan3E` | 3.4 | 0.0 | 0.4 |
| `RO_Spartan7_roomT` | 0.24 | 0.0 | 0.7 |

Thus, matching marginals or error-count histograms does not imply that
synthetic vectors reproduce the genuine joint distribution.

## Reproducing the study

Install the locked environment:

``` console
uv sync
```

Verify that the artifact symlink resolves:

``` console
ls -l data/puf_artifact.h5
```

Run the configured experiment and generate all figures:

``` console
./run_all.sh
```

The equivalent analysis command is:

``` console
uv run python scripts/analysis_free_energy.py \
  --families SRAM_XMC4500 RO_Spartan7_roomT RO_Maiti_Spartan3E \
  --devices-per-family 20 \
  --group-sizes 1 2 4 6 8 10 12 16 20 \
  --training-configs cd:1 cd:5 cd:10 pcd:1 \
  --visible-bias-init marginal \
  --gibbs-steps 1 10 50 100 \
  --relaxation-steps 1 5 10
```

Generate figures from existing CSV artifacts:

``` console
uv run python scripts/plot_free_energy.py
```

Run the model-free count and conditional error-structure analysis:

``` console
uv run python scripts/analysis_error_structure.py \
  --families SRAM_XMC4500 RO_Spartan7_roomT RO_Maiti_Spartan3E
```

Generate its figures and family summary:

``` console
uv run python scripts/plot_error_structure.py
```

For RO Infineon, relate the corrected reliability results to raw-frequency
and manufacturing metadata, then generate the dedicated figures:

``` console
uv run python scripts/analysis_ro_infineon_manufacturing.py
uv run python scripts/plot_ro_infineon_manufacturing.py
uv run python scripts/analysis_ro_infineon_frequency_prediction.py
uv run python scripts/plot_ro_infineon_frequency_prediction.py
```

The additional figure directories are:

| Directory | Interpretation |
|---|---|
| `ecc_capacity/` | Median device correction probability and inter-device IQR. |
| `conditional_residuals/` | Shared conditional residual, inter-device IQR, and count-stratum support. |
| `reference_polarity/` | Reference-conditioned transition rates, raw and residual contrasts, and device support for each family. |
| `population_survival/` | Equal-device mean failure tail with nested-bootstrap interval and upper device-risk quantiles. |
| `device_coverage/` | Fraction of devices meeting each failure-probability target as capacity increases. |

`reference_polarity_family_summary.png` compares the support-weighted raw
and residual polarity contrasts across families. The per-family panels show
where in (K) the directionality occurs and whether it exceeds the
independent-marginal prediction. Count strata with inadequate device support
are omitted from the family curves rather than extrapolated.

In `population_survival`, the upper panel answers the random-device/random-
readout question. The lower panel shows whether that mean hides a poor-device
tail. In `device_coverage`, empirical curves are descriptive: observed zero
failures do not prove that the true device risk is zero. Targets that imply the
same integer number of allowed failures are therefore drawn as one empirical
curve. For example, with eight readouts, every target below \(1/8\) means
exactly zero observed failures. The continuous Poisson-binomial targets remain
separate because they are model probabilities rather than observed counts.

## RO Infineon manufacturing analysis

The RO Infineon side study uses the unique physical-device identifier
\((x,y,\text{wafer id},\text{lot id})\). A wafer is identified by the pair
`(lot_id, wafer_id)`, because the numeric wafer identifier is not globally
unique across lots. The raw tensor has shape

\[
D\times 6\times 255\times 8,
\]

where the second axis contains six copies of each of 255 RO designs and the
last axis contains eight operating conditions. Those eight values are not
temporal replicates. The analysis first reconstructs all 256 selected binary
responses from their signed frequency differences and refuses to continue if
they disagree with the corrected artifact.

The device features include mean frequency across conditions, global,
within-lot, and within-wafer frequency z-scores, frequency variation across
conditions, selected-pair frequency differences, and summaries of the
empirical error count. For artifact bit \(i\) in condition \(c\), define

\[
\bar f_{d,i,c}
=\frac{f_{d,g_a(i),r(i),c}+f_{d,g_b(i),r(i),c}}{2},
\]

\[
\Delta f_{d,i,c}
=f_{d,g_a(i),r(i),c}-f_{d,g_b(i),r(i),c},
\]

and

\[
\Delta f_{\mathrm{rel},d,i,c}
=\frac{\Delta f_{d,i,c}}{\bar f_{d,i,c}}.
\]

The analysis also stores \(\log(f_1/f_2)\). Small \(|\Delta f|\) means that
the selected RO pair is close to changing order. The fifth percentile
\(Q_{0.05}(|\Delta f|)\) summarizes the lower tail of absolute separation:
it describes the weakest 5% of comparisons, not departure from Gaussianity
or mass in the outer tails. It is a mechanistic positive control because the
same sign changes define the binary instability.

Raw and relative differences represent different physical hypotheses. A raw
difference is appropriate for approximately additive, frequency-independent
noise. A relative difference is appropriate when perturbation magnitude
scales with operating frequency. Retaining \(\bar f\), \(\Delta f\), and
\(\Delta f_{\mathrm{rel}}\) allows the study to ask whether the same absolute
separation behaves differently at different frequency scales. Because the
eight measurements are operating conditions rather than repeated measurements
under one condition, the observed flips quantify operating-condition
sensitivity rather than pure temporal noise.

For every predeclared feature/outcome pair, `correlations.csv` reports a
Spearman correlation and a two-sided stratified permutation p-value. Feature
ranks are permuted only among devices on the same physical wafer. This asks
whether an association remains beyond fixed lot/wafer shifts; it does not
establish causality, and the multiple exploratory pairs should not be read as
independent confirmatory tests. The eight rows marked `primary_analysis` are the
predeclared comparisons emphasized in the relationship figure.

The manufacturing analysis writes files under
`data/ro_infineon_manufacturing/`:

| File | Contents |
|---|---|
| `device_features.parquet` | Manufacturing identity, raw and relative frequency-difference features, and corrected reliability metrics for each device. |
| `selected_response_differences.parquet` | Pair mean frequency, raw/relative differences, log ratio, and aligned bit flip for every device, artifact bit, and condition. |
| `correlations.csv` | Spearman effects and within-wafer permutation p-values. |
| `coverage_by_feature_group.parquet` | ECC device coverage for low, middle, and high rank terciles of selected manufacturing features. |
| `device_exceedances.parquet` | Per-device numbers and fractions of observed readouts exceeding each correction capacity. |
| `difference_scale_surfaces.parquet` | Equal-count two-dimensional bins with device-balanced bit-flip probabilities. |
| `physical_bit_residual_summaries.parquet` | Support-weighted signed and absolute conditional residual attribution joined to device-bit physical features. |
| `physical_residual_associations.csv` | Mean within-device rank effects, wafer-cluster intervals, and bit-permutation p-values. |
| `residual_by_physical_decile.parquet` | Equal-device conditional residual summaries by within-device physical-feature decile and supported error count. |
| `residual_by_physical_tercile.parquet` | Low/middle/high physical-feature residual profiles. |
| `physical_variability_residual_iqr.parquet` | Across-device physical and conditional-residual IQR per response bit and supported error count. |
| `physical_iqr_residual_iqr_associations.csv` | Bit-label permutation tests relating physical IQR to residual IQR across supported error-count strata. |

The corresponding figures are under
`figures/ro_infineon_manufacturing/`:

- `frequency_vs_reliability.png` shows individual devices colored by physical
  wafer, with equal-count-bin medians to expose nonlinear trends.
- `coverage_by_feature/` compares the fraction of devices meeting correction
  criteria in low, middle, and high feature terciles. With eight readouts, the
  three panels mean at most 0, 1, or 2 observed failures; they are descriptive
  finite-enrollment criteria rather than precise estimates of rare-event risk.
- `wafer_distributions.png` shows whether physical-wafer shifts or
  heterogeneity accompany the reliability differences.
- `difference_scale_bitflip_probability.png` compares raw and relative
  separation across pair-frequency bins. Each cell first computes a bit-flip
  rate per represented device and then averages devices equally.
- `spatial_by_wafer/` maps frequency, weak relative frequency differences,
  error counts, and dependency residuals at die coordinates. Spatial patterns
  are evidence of association with manufacturing location, not proof of a
  manufacturing mechanism.
- `physical_residuals/` relates bit-level separation and condition variation
  to signed residual attribution, absolute residual magnitude, and
  across-device residual heterogeneity.

### Physical variation and conditional residuals

For a device, bit, and observed error-count stratum, the conditional residual
is

\[
r_{d,i,k}=P_{\mathrm{emp}}(E_i=1\mid K=k)
-P_{\mathrm{ind}}(E_i=1\mid K=k).
\]

Both conditional probability vectors sum to \(k\), so
\(\sum_i r_{d,i,k}=0\). The global signed mean is therefore fixed by
construction and is not tested. Instead, the analysis computes support-weighted
signed and absolute device-bit attribution:

\[
S_{d,i}=\frac{1}{S}\sum_k n_{d,k}r_{d,i,k},
\qquad
A_{d,i}=\frac{1}{S}\sum_k n_{d,k}|r_{d,i,k}|.
\]

Within-device rank tests relate raw/relative separation to \(S_{d,i}\) and
raw/relative condition variation to \(A_{d,i}\). Confidence intervals resample
whole physical wafers, while the null permutes bit labels within devices.
Heatmaps and tercile profiles retain exact \(k\), but display only strata
represented by at least 40 devices. Support is reported explicitly and is not
treated as an independent physical outcome.

The across-device analysis asks whether response positions with heterogeneous
physical condition sensitivity also have heterogeneous residual attribution.
It compares physical-feature IQR with residual IQR across 256 bits, retains the
same supported \(k\) strata, and uses one shared bit-label permutation across
all strata. Because only eight operating conditions are available, these are
descriptive physical associations rather than identified causal mechanisms.
Moreover, collapsing residuals over \(k\) can retain marginal-calibration
effects; the exact-\(k\) heatmaps are essential for interpreting the aggregate
rank associations.

### Held-out-condition frequency prediction

The cross-condition study avoids defining the reference from one arbitrarily
selected condition or from all eight observations. For each held-out condition
\(c\), the other seven conditions define an odd-sample majority reference:

\[
R_{d,i}^{(-c)}=
\mathbf{1}\!\left[\sum_{c'\ne c}R_{d,i,c'}\geq 4\right],
\qquad
E_{d,i,c}=R_{d,i,c}\oplus R_{d,i}^{(-c)}.
\]

Pair-frequency predictors are also computed only from the seven enrollment
conditions. Thus the held-out response and its frequency measurements do not
enter its predictors or reference. Every condition is held out once. Model
validation uses five group folds over physical `(lot_id, wafer_id)` groups, so
all devices from a physical wafer remain on the same side of a validation
split.

Four additive logistic spline models compare raw separation, relative
separation, and each separation together with pair mean frequency. All models
use the same held-out-condition and response-bit effects. Primary comparison
metrics are device-balanced log loss and Brier score; AUROC measures ranking,
calibration bins weight represented devices equally, and paired device
bootstraps quantify differences between candidate and baseline models.

The held-out bit probabilities also define a frequency-informed independent
Poisson-binomial model for the error count. Its predicted
\(P(K>\tau)\) is compared with the held-out empirical exceedance frequency.
Agreement in mean error count combined with disagreement in the upper tail
indicates remaining joint or heterogeneous error structure rather than a
failure to estimate average marginal reliability.

The cross-condition analysis writes files under
`data/ro_infineon_frequency_prediction/`:

| File | Contents |
|---|---|
| `cross_condition_predictions.parquet` | Wafer-blocked held-out probability for every model, device, bit, and condition. |
| `fold_metrics.csv` | Log loss, Brier score, and AUROC for every model and wafer-group fold. |
| `model_summary.csv` | Equal-device metrics with device-bootstrap intervals. |
| `paired_model_comparisons.csv` | Paired device-bootstrap changes between physical hypotheses. |
| `calibration.parquet` | Equal-count, equal-device calibration summaries. |
| `ecc_tail_predictions.parquet` | Frequency-informed Poisson-binomial tails and held-out exceedance observations. |
| `ecc_summary.csv` | Population tail calibration and tail-event Brier scores by correction capacity. |

Figures under `figures/ro_infineon_frequency_prediction/` show the model
comparison, paired changes, probability calibration, and ECC-tail prediction.

Run the checks:

``` console
uv run pytest
uv run ruff check scripts tests
uv run ty check src scripts tests
```

Build the documentation:

``` console
uv run zensical build
```

Random seeds control device selection, data splitting, model initialization,
data-loader shuffling, alternative generation, bootstrap sampling, and Gibbs
sampling. Exact numerical reproducibility additionally depends on the
PyTorch backend and hardware.

## Future work

### Two-channel tampering detector

Use an explicit marginal channel and dependency channel:

\[
M_N = \sum_{n=1}^{N}\log p_{\mathrm{ind}}(E_n),
\]

\[
R_N = \sum_{n=1}^{N}
\left[
s_{\mathrm{RBM}}(E_n)-\log p_{\mathrm{ind}}(E_n)
\right].
\]

A joint rule can reject a claimant when either its marginal reliability or
its dependency structure is inconsistent with enrollment. The thresholds
must be jointly calibrated to control the genuine false-rejection rate.

An arithmetic combination is not sufficient because

\[
M_N+R_N=S_{\mathrm{RBM},N}.
\]

The current raw RBM score already implements that sum and can hide an
anomalous channel through cancellation. The implemented joint Mahalanobis
test instead retains \((M_N,R_N)\) as a two-dimensional statistic and
calibrates its genuine distribution.

Publishing or transferring the marginal probabilities can leak device
reliability information. A deployment study should determine:

- Whether marginals remain verifier-side secrets.
- Whether quantized, hashed, or aggregate marginals are sufficient.
- Whether helper data leaks unstable-bit locations.
- How this leakage interacts with fuzzy extractors and error-correction
  helper data.
- Whether an attacker can synthesize matched-marginal vectors and then learn
  the residual structure through repeated queries.

### Statistical test and required number of readouts

Let \(r_g\) and \(r_0\) denote per-readout residual scores under genuine data
and a matched-marginal null. Under an approximate independent Gaussian
model,

\[
\operatorname{AUROC}_N
\approx
\Phi\left(
\frac{\sqrt{N}(\mu_g-\mu_0)}
{\sqrt{\sigma_g^2+\sigma_0^2}}
\right).
\]

Estimating the per-readout means and variances would allow the study to:

- Predict AUROC as a function of \(N\).
- Estimate the minimum \(N\) required for a target significance or
  confidence level.
- Estimate \(N\) for a target false-acceptance rate at a fixed
  false-rejection rate.
- Compare theoretical predictions with bootstrap results.
- Identify violations caused by skewness, temporal dependence, or device
  heterogeneity.

Repeated splits and device-level confidence intervals are needed before
interpreting this as a deployment guarantee.

### Residual RBM for generation

A residual energy model can use the independent device marginals as an
explicit base distribution:

\[
p_\theta(v)
\propto
p_{\mathrm{ind}}(v)\exp(g_\theta(v)).
\]

Equivalently,

\[
E_\theta(v)
=
E_{\mathrm{ind}}(v)+E_{\mathrm{residual},\theta}(v).
\]

The base logits enforce the target reliability, while the residual energy
learns only the dependencies. The present marginal visible-bias
initialization is related to this idea, but it does not enforce the
factorization because training can move the biases and hidden-unit
interactions can alter the visible marginals.

Possible experiments include:

- Freeze or strongly regularize visible biases at the empirical marginal
  logits.
- Use a centered RBM parameterization so weights model deviations from the
  marginal means.
- Alternate interaction learning with visible-bias moment matching.
- Apply stochastic marginal correction to the visible biases until generated
  marginals match their targets.
- Train separate discriminator and generator RBMs when their objectives
  conflict.
- Evaluate every generator using held-out residual evidence, not only BER or
  error-count fidelity.

These approaches are hypotheses and are not implemented in the current
study.

The first residual architecture will keep smoothed visible marginal logits
fixed and train hidden interactions plus a scalar density-ratio offset. This
avoids spending an optimization stage relearning the closed-form Bernoulli
marginals and reduces the opportunity for the residual score to exploit
finite-sample marginal differences.

After testing that architecture, a centered RBM is a relevant alternative.
Centered binary RBMs subtract visible and hidden offsets so their weights
describe deviations around average activity. This reparameterization can
improve optimization and generative likelihood and can produce update
directions closer to the natural gradient. Adaptive offsets require care:
using moving-average offsets is more stable than replacing them directly
with noisy batch or model means. See
[Melchior, Fischer, and Wiskott, *How to Center Deep Boltzmann Machines*](https://jmlr.csail.mit.edu/papers/v17/14-237.html).

This centered variant should be evaluated only after the fixed-marginal
density-ratio model establishes a baseline. It should be compared on
held-out residual discrimination, generated marginal drift, dependency
fidelity, and chain mixing.

### RBM as an evaluator for other generators

The residual discriminator can evaluate synthetic PUF data produced by
another model. A generator intended for reliability, error-correction, or
fuzzy-extractor studies should reproduce:

- Per-bit marginal BER.
- Error-count distributions.
- Contemporaneous dependency evidence.
- Device-to-device variability.
- Relevant environmental behavior.

Passing only marginal or total-error-count tests is insufficient.

Care must be taken to avoid evaluating a generator with the same data and
model family used to train it. Held-out devices or an independently trained
discriminator provide a stronger test.

### Independence rejection and transferability

The results connect to two axes previously studied with sparse nodewise
models:

1. Whether learned dependencies exceed those expected under an independent
   matched-marginal null.
2. Whether dependency structure transfers to held-out devices from the same
   PUF family.

Future experiments can train RBMs on a subset of devices and evaluate
residual evidence on held-out devices without refitting. This separates
device-specific memorization from family-level physical structure.

### Discrimination-versus-generation diagnosis

Compare residual evidence for:

- Training data.
- Calibration data.
- Held-out test data.
- Independent matched-marginal vectors.
- RBM-generated vectors.

If training evidence is substantially larger than calibration and test
evidence, the model is overfitting. If training and test agree while
generated evidence remains near the independent null, the likely problem is
in the learned equilibrium distribution or negative-phase training.

Additional useful diagnostics include:

- Pairwise Hamming distance between coupled chains.
- Chain autocorrelation and effective sample size.
- PCD with additional persistent-chain steps.
- Hidden-unit and regularization sweeps.
- Independent-null-corrected covariance or graph statistics.

<script id="MathJax-script" src="https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
  window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };

  document$.subscribe(() => {
    MathJax.startup.output.clearCache()
    MathJax.typesetClear()
    MathJax.texReset()
    MathJax.typesetPromise()
  })
</script>
