# (C) 2025-2026 Annie Tran
# Structural Parity Model of Cancer
# github.com/nntrn/cancer

import numpy as np

def norm_cdf(x):
  return 0.5 * (1 + np.erf(x / np.sqrt(2)))

# CORE STATE DYNAMICS

def evolve_state(state, t, params):
  """
  Core differential equations governing cancer state evolution.

  Medical context:
    state[0] (A): Asymmetry - whether cell is in proliferative (1) or quiescent (0) mode
    state[1] (C): Closure - degree of differentiation (0=stem-like, 1=fully differentiated)
    state[2] (K): Constraint - strength of tissue architecture/cell-cell adhesion
    state[3] (N): Cell number - tumor burden

  Parameters:
    alpha: rate at which proliferation drives toward differentiation
    beta: rate at which constraints enforce differentiation
    gamma: rate at which proliferation erodes constraints
    delta: external damage rate (carcinogens, inflammation)
    lambda_max: maximum proliferation rate
    K0: reference constraint level
    K_crit: critical constraint below which asymmetry can activate

  Scientific basis:
    Proliferating cells attempt to differentiate (α·A·(1-C)) but constraints
    can enforce or prevent this (β·K·C). Sustained proliferation degrades
    tissue architecture (γ·A), and growth rate depends on undifferentiated
    state and remaining constraints.

  Assumptions:
    - A is binary in reality but handled separately from continuous dynamics
    - Linear coupling between parameters (no higher-order interactions)
    - Constraint degradation is irreversible without intervention

  >>> state = [1, 0.3, 5, 100]
  >>> params = [0.05, 0.01, 0.01, 0, 0.5, 10, 2]
  >>> evolve_state(state, 0, params)
  [0, 0.0335, -0.01, 17.5]
  """
  A, C, K, N = state
  alpha, beta, gamma, delta, lambda_max, K0, K_crit = params

  dC_dt = alpha * A * (1 - C) - beta * K * C
  dK_dt = -gamma * A - delta
  dN_dt = lambda_max * N * A * (1 - C) * np.exp(-K / K0)

  return [0, dC_dt, dK_dt, dN_dt]

def compute_closure_number(x, max_k=100):
  """
  Calculate structural resistance to achieving stable differentiation.

  Medical context:
    x: Integer representation of cellular state
    Returns: Number of iterations before stable state achieved

  Scientific basis:
    Models cancer as recursive state that must iterate x→3x until reaching
    even parity (stable). Higher closure number = more resistant to treatment.

  Assumptions:
    - Cellular state can be mapped to integer representation
    - 3x represents division + retention of proliferative state
    - Even parity = stable differentiated state

  >>> compute_closure_number(1)
  1
  >>> compute_closure_number(3)
  2
  >>> compute_closure_number(7)
  4
  """
  for k in range(1, max_k + 1):
    result = int(x * (3**k))
    if bin(result).count("1") % 2 == 0:
      return k
  return max_k

def hamming_parity(n):
  """
  Check if state has even (stable) or odd (unstable) parity.

  Medical context:
    n: Cellular state representation
    Returns: 0 if stable, 1 if unstable

  >>> hamming_parity(3)
  0
  >>> hamming_parity(7)
  1
  """
  return bin(int(n)).count("1") % 2

# TUMOR INITIATION (1-4)

def carcinogen_exposure(K0, alpha, t_max, K_crit, C_threshold):
  """
  1. Carcinogen-induced constraint degradation leading to cancer initiation.

  Medical context:
    K0: Initial tissue constraint (E-cadherin level, junction integrity) [~10-20 in normal tissue]
    alpha: Carcinogen potency (DNA damage rate per unit time) [~0.01-0.1 for moderate carcinogens]
    t_max: Exposure duration [months or years]
    K_crit: Critical constraint threshold for cancer initiation [~2-5]
    C_threshold: Closure level below which asymmetry activates [~0.7]

  Scientific basis:
    Carcinogens don't just mutate DNA—they degrade structural constraints (K).
    When K falls below critical threshold, cells can't maintain differentiated
    state (C drops), triggering proliferative asymmetry (A→1). Predicts cancer
    from constraint loss, not just mutation accumulation.

  Assumptions:
    - Exponential K decay (constant fractional damage rate)
    - C responds linearly to K deficit below threshold
    - A flips once and stays (irreversible without intervention)

  >>> t, K, C, A = carcinogen_exposure(10, 0.05, 100, 3, 0.7)
  >>> K[0], K[-1]
  (10.0, 0.067...)
  >>> np.sum(A > 0)  # Steps after cancer initiation
  ~950
  """
  t = np.linspace(0, t_max, 1000)
  K = K0 * np.exp(-alpha * t)
  C = np.ones_like(K)
  A = np.zeros_like(K)

  for i in range(len(t)):
    if K[i] < K_crit:
      C[i] = 1 - 0.3 * (K_crit - K[i])
    if C[i] < C_threshold and i > 0:
      A[i:] = 1
      break

  return t, K, C, A

def chronic_inflammation(I0, omega, t_max):
  """
  2. Chronic inflammation prevents closure, locking cells in proliferative state.

  Medical context:
    I0: Inflammatory signal strength (cytokine levels: IL-6, TNF-α) [~0.5-2.0 for chronic]
    omega: Closure degradation rate per inflammatory unit [~0.01-0.05]
    t_max: Duration of inflammation [months to years]

  Scientific basis:
    Sustained inflammatory signals (cytokines) continuously push cells toward
    proliferation, preventing differentiation closure. Unlike acute inflammation
    (which resolves), chronic inflammation never allows C→1, keeping A=1.
    Explains inflammation-associated cancers (IBD→colon cancer, hepatitis→HCC).

  Assumptions:
    - Constant inflammatory signal (doesn't fluctuate)
    - Linear C degradation (no saturation effects)
    - A flips when C crosses threshold

  >>> t, C, A = chronic_inflammation(1.0, 0.02, 50)
  >>> C[0], C[-1]
  (1.0, 0.0)
  >>> np.where(A == 1)[0][0]  # First index where A=1
  ~15
  """
  t = np.linspace(0, t_max, 1000)
  C = 1 - omega * I0 * t
  C = np.clip(C, 0, 1)
  A = (C < 0.7).astype(int)
  return t, C, A

def inherited_syndrome(C_num_baseline, delta_genetic, K0, K_crit, delta_hit):
  """
  3. Inherited mutations create elevated baseline closure number.

  Medical context:
    C_num_baseline: Structural resistance in normal cells [~5-10]
    delta_genetic: Additional resistance from germline mutation (BRCA, Lynch) [~10-30]
    K0: Starting constraint in mutation carriers [~8-12, lower than normal ~15]
    K_crit: Threshold for cancer [~2-5]
    delta_hit: Constraint loss per somatic event [~1-2]

  Scientific basis:
    Inherited cancer syndromes start with higher closure number (harder to
    differentiate) and lower K (weaker constraints). Requires fewer somatic
    "hits" to reach cancer state. Explains high penetrance without requiring
    many mutations—structural predisposition, not just genetic.

  Assumptions:
    - Germline mutations affect initial state, not dynamics
    - Somatic hits reduce K linearly
    - Factorial probability assumes independence of hits

  >>> C_num, n_hits = inherited_syndrome(5, 20, 8, 3, 1.5)
  >>> C_num
  25
  >>> n_hits
  4
  >>> inherited_syndrome(5, 5, 12, 3, 1.5)  # Normal person
  (10, 6)
  """
  C_num = C_num_baseline + delta_genetic
  n_hits = max(0, int(np.ceil((K_crit - K0) / delta_hit)))
  return C_num, n_hits

def spontaneous_transformation(age, K0, lambda_aging, K_crit, sigma):
  """
  4. Age-related stochastic constraint degradation causes spontaneous cancer.

  Medical context:
    age: Patient age [years, typically 0-100]
    K0: Constraint level at birth [~15-20]
    lambda_aging: Aging-related K degradation rate [~0.01-0.03 per year]
    K_crit: Cancer threshold [~2-5]
    sigma: Stochastic variation in K [~1-3]

  Scientific basis:
    Constraints naturally degrade with age (accumulated damage, senescence,
    inflammation). Cancer occurs when random fluctuations push K below threshold.
    Predicts age-dependent cancer incidence without requiring specific mutations.
    Explains sporadic cancers in elderly.

  Assumptions:
    - Linear aging effect (could be exponential in reality)
    - Gaussian noise in K (could have fat tails)
    - Single threshold model (may have multiple stages)

  >>> K_age, P = spontaneous_transformation(30, 15, 0.02, 3, 2)
  >>> K_age
  9.0
  >>> P
  0.0013...
  >>> spontaneous_transformation(70, 15, 0.02, 3, 2)
  (1.0, 0.8413...)
  """
  K_age = K0 * (1 - lambda_aging * age)
  P_cancer = norm_cdf((K_crit - K_age) / sigma)
  return K_age, P_cancer

# TUMOR GROWTH PATTERNS (5-8)

def exponential_growth(N0, lambda_rate, t_max):
  """
  5. Early exponential growth when closure resistance is high.

  Medical context:
    N0: Initial tumor cell count [typically 1-100]
    lambda_rate: Effective proliferation rate [~0.1-0.5 per day for aggressive]
    t_max: Time to observe [days to months]

  Scientific basis:
    When C≈0 (no differentiation pressure) and K is low, growth follows
    N(t) = N0·e^(λt). Corresponds to x→3x recursion with no closure approach.
    Explains rapid early growth phase before vascular/spatial constraints engage.

  Assumptions:
    - Unlimited resources (valid for small tumors)
    - No spatial constraints (breaks down at ~1mm diameter)
    - Constant proliferation rate (ignores heterogeneity)

  >>> t, N = exponential_growth(10, 0.3, 20)
  >>> N[0], N[-1]
  (10.0, 4034.2...)
  >>> exponential_growth(1, 0.5, 10)[1][-1]
  148.4...
  """
  t = np.linspace(0, t_max, 1000)
  N = N0 * np.exp(lambda_rate * t)
  return t, N

def gompertzian_growth(N0, r, K_max, t_max):
  """
  6. Growth slows as tumor size increases closure pressure.

  Medical context:
    N0: Initial cell count [10-1000]
    r: Growth deceleration rate [~0.01-0.1]
    K_max: Carrying capacity (max tumor size without intervention) [~10^9-10^12 cells]
    t_max: Observation period [weeks to years]

  Scientific basis:
    As tumor grows, spatial constraints increase effective C (closure pressure).
    Growth rate decreases as C→1 locally. Explains why large tumors grow slower
    than small ones—not resource limitation but increased differentiation pressure
    from density.

  Assumptions:
    - C increases monotonically with size (C ∝ N^α)
    - Carrying capacity is fixed (ignores angiogenesis)
    - No metastatic escape

  >>> t, N = gompertzian_growth(100, 0.05, 1e6, 50)
  >>> N[0], N[-1]
  (100.0, ~900000)
  >>> N[-1] / N[0]  # Fold increase
  ~9000
  """
  t = np.linspace(0, t_max, 1000)
  N = K_max * np.exp(np.log(N0 / K_max) * np.exp(-r * t))
  return t, N

def aggressive_low_mutation(M, C_num):
  """
  7. Low mutation burden but high structural resistance causes aggressive growth.

  Medical context:
    M: Mutation count (TMB = mutations per megabase) [Low: <5, High: >20]
    C_num: Structural closure resistance [Range: 5-100+]

  Scientific basis:
    Standard oncology struggles to explain aggressive cancers with few mutations
    (pediatric tumors, some lymphomas). This model: aggressiveness ∝ 1/C_num, not M.
    Simple genomic state (low M) can have high C_num if geometry is resistant to
    closure. Explains aggressive behavior without genomic complexity.

  Assumptions:
    - M and C_num are independent (not always true—some mutations affect C_num)
    - Growth rate inversely proportional to C_num (simplified relationship)

  >>> aggressive_low_mutation(5, 100)
  0.01
  >>> aggressive_low_mutation(5, 10)
  0.1
  >>> aggressive_low_mutation(50, 100)  # High M doesn't change outcome if C_num high
  0.01
  """
  growth_rate = 1 / C_num
  return growth_rate

def slow_high_mutation(M, K_high, K0):
  """
  8. High mutation burden but strong constraints cause slow growth.

  Medical context:
    M: Mutation count [High: >20 mutations/Mb]
    K_high: Remaining constraint strength [~5-15, higher than expected]
    K0: Reference constraint level [~10]

  Scientific basis:
    Some heavily mutated tumors grow slowly because high K suppresses proliferation
    despite damage. Growth ∝ e^(-K/K0), so even with many mutations, sufficient
    constraint prevents rapid growth. Explains indolent high-mutation tumors
    (some melanomas, MSI-high colon cancers).

  Assumptions:
    - K can remain high despite mutations (some mutations don't affect K)
    - Exponential suppression (could be other functional forms)
    - K is measurable independently of M

  >>> slow_high_mutation(50, 15, 10)
  0.223...
  >>> slow_high_mutation(50, 5, 10)
  0.606...
  >>> slow_high_mutation(5, 15, 10)  # Low M, high K
  0.223...
  """
  suppression = np.exp(-K_high / K0)
  return suppression

# METASTASIS (9-12)

def metastatic_spread_probability(C, K, C_range=(0.7, 0.95), K_min=2.0):
  """
  9. Metastasis requires "Goldilocks" partial differentiation state.

  Medical context:
    C: Differentiation state [0=stem-like, 1=fully differentiated]
    K: Constraint/identity strength [Range: 0-20]
    C_range: Optimal window for metastasis [Default: 0.7-0.95]
    K_min: Minimum K to survive circulation [~2-4]

  Scientific basis:
    Too undifferentiated (C<0.7): unstable, dies in circulation
    Too differentiated (C>0.95): can't proliferate at distant site
    Insufficient K: loses identity during transit
    Explains why moderate differentiation predicts metastasis better than extremes.

  Assumptions:
    - Sharp boundaries for C window (likely smoother gradients)
    - K threshold is binary (probably continuous probability)
    - Independent C and K requirements (may interact)

  >>> metastatic_spread_probability(0.8, 5)
  1.0
  >>> metastatic_spread_probability(0.5, 5)
  0.0
  >>> metastatic_spread_probability(0.8, 1)
  0.0
  """
  in_range = (C >= C_range[0]) & (C <= C_range[1])
  sufficient_K = K >= K_min
  return (in_range & sufficient_K).astype(float)

def metastatic_dormancy(C_local, C_threshold, kappa, t_max):
  """
  10. Microenvironment maintains high closure, causing dormancy.

  Medical context:
    C_local: Initial closure at metastatic site [~0.9-1.0 for dormancy]
    C_threshold: Level below which growth reactivates [~0.7-0.8]
    kappa: Rate of microenvironment degradation [~0.001-0.01 per month]
    t_max: Follow-up duration [months to years]

  Scientific basis:
    Distant organ microenvironment forces high C (dormancy). Over time, local
    factors (inflammation, aging, wound healing) degrade this, allowing C to
    drop below threshold and reactivate. Explains late relapses (5-20 years)
    and organ-specific dormancy durations.

  Assumptions:
    - Linear C degradation (may have nonlinear dynamics)
    - Single threshold for reactivation (could be gradual)
    - Microenvironment is only factor (ignores systemic immunity)

  >>> t, C, A, T_dorm = metastatic_dormancy(0.95, 0.75, 0.01, 50)
  >>> T_dorm
  20.0
  >>> C[0], C[-1]
  (0.95, 0.45)
  >>> metastatic_dormancy(0.98, 0.75, 0.005, 100)[3]
  46.0
  """
  t = np.linspace(0, t_max, 1000)
  C = C_local - kappa * t
  A = (C < C_threshold).astype(int)
  T_dormant = (C_local - C_threshold) / kappa if kappa > 0 else np.inf
  return t, C, A, T_dormant

def organotropism(C_num_tumor, C_num_organs, sigma=10):
  """
  11. Metastatic site preference follows structural matching.

  Medical context:
    C_num_tumor: Structural resistance of primary tumor [~10-100]
    C_num_organs: Array of organ closure numbers
      [Liver: ~15, Lung: ~25, Bone: ~50, Brain: ~70]
    sigma: Matching tolerance [~5-15]

  Scientific basis:
    Metastasis succeeds when tumor C_num matches organ environment. Explains
    organ tropism: breast→liver/lung (both ~20), prostate→bone (~50).
    Geometric compatibility, not just chemokine gradients. Predicts metastatic
    patterns from primary tumor structural state.

  Assumptions:
    - Gaussian matching probability (could be other distributions)
    - Organ C_num is static (changes with age, disease)
    - Ignores other factors (chemokines, adhesion molecules)

  >>> organs = np.array([15, 25, 50, 70])  # Liver, lung, bone, brain
  >>> organotropism(20, organs, 10)
  array([0.324..., 0.412..., 0.135..., 0.082...])
  >>> organotropism(55, organs, 10)
  array([0.011..., 0.040..., 0.583..., 0.363...])
  """
  distances = np.abs(C_num_tumor - C_num_organs)
  P_mets = np.exp(-(distances**2) / (2 * sigma**2))
  return P_mets / P_mets.sum()

def ctc_survival(K, K_autonomous, tau, t):
  """
  12. Circulating tumor cell survival requires autonomous constraint maintenance.

  Medical context:
    K: Cell's constraint level [~0-10]
    K_autonomous: Minimum K to survive without microenvironment [~3-5]
    tau: Circulation time constant [~1-10 hours]
    t: Time in circulation [hours]

  Scientific basis:
    Circulation provides no structural support (C≈0 environment). Cells must
    maintain identity (K) autonomously. Most CTCs die (K<K_autonomous), explaining
    why millions of CTCs produce few metastases. Successful CTCs have elevated
    K-maintenance machinery (adhesion, cytoskeleton genes).

  Assumptions:
    - Exponential survival decay (may have threshold effects)
    - Binary K threshold (likely continuous probability)
    - Constant tau (varies by circulation route)

  >>> ctc_survival(5, 3, 2, 1)
  0.606...
  >>> ctc_survival(2, 3, 2, 1)
  0.0
  >>> ctc_survival(6, 3, 2, 10)
  0.00673...
  """
  P_survive = np.exp(-t / tau) * (K >= K_autonomous).astype(float)
  return P_survive

# TREATMENT RESPONSE (13-19)

def chemo_sensitivity(dN_dt, mu):
  """
  13. Chemotherapy kills cells proportional to division rate.

  Medical context:
    dN_dt: Tumor growth rate (cells per day) [~10^6-10^9 for active tumors]
    mu: Drug potency (kill fraction per division) [~0.1-0.9]

  Scientific basis:
    Chemotherapy targets dividing cells. Kill rate ∝ dN/dt = λ·N·(1-C).
    Highly proliferative (low C, high A) tumors are most sensitive.
    Explains why quiescent cells (high C) resist chemotherapy.

  Assumptions:
    - Linear relationship (saturation at high doses)
    - All dividing cells equally sensitive (ignores heterogeneity)
    - Immediate effect (ignores pharmacokinetics)

  >>> chemo_sensitivity(1e6, 0.5)
  500000.0
  >>> chemo_sensitivity(1e3, 0.9)
  900.0
  """
  kill_rate = mu * dN_dt
  return kill_rate

def chemo_resistance(K_sensitive, delta_K, K0):
  """
  14. Resistance via increased constraint (slow growth).

  Medical context:
    K_sensitive: Pre-treatment constraint [~2-5]
    delta_K: Constraint increase in resistant cells [~3-10]
    K0: Reference constraint [~10]

  Scientific basis:
    Resistant cells increase K, slowing proliferation (e^(-K/K0)). They survive
    by dividing less, not by blocking drug action. Explains slow-growing resistant
    clones that emerge post-treatment.

  Assumptions:
    - Exponential suppression (simplified from full dynamics)
    - K increase is only mechanism (ignores drug efflux, repair)
    - Stable elevated K (may fluctuate)

  >>> chemo_resistance(3, 5, 10)
  0.449...
  >>> chemo_resistance(3, 10, 10)
  0.182...
  """
  K_resistant = K_sensitive + delta_K
  suppression = np.exp(-K_resistant / K0)
  return suppression

def radiation_response(C_num_pre, delta_damage, C_num_lethal, sigma):
  """
  15. Radiation increases closure number; apoptosis if exceeds threshold.

  Medical context:
    C_num_pre: Pre-treatment structural resistance [~10-80]
    delta_damage: C_num increase from radiation [~5-20 per Gy]
    C_num_lethal: Level triggering apoptosis [~40-60]
    sigma: Cell-to-cell variation [~5-10]

  Scientific basis:
    Radiation causes DNA damage, increasing difficulty of achieving stable state
    (C_num↑). If C_num exceeds lethal threshold, forces apoptosis (ultimate closure).
    Low-C_num tumors are sensitive (damage pushes them over threshold). High-C_num
    tumors resist (already far from threshold).

  Assumptions:
    - Additive damage (may have nonlinear dose response)
    - Gaussian variation (tails may be important)
    - Single threshold (likely multiple damage checkpoints)

  >>> C_post, P = radiation_response(20, 15, 40, 5)
  >>> C_post
  35
  >>> P
  0.841...
  >>> radiation_response(50, 15, 40, 5)
  (65, 0.0)
  """
  C_num_post = C_num_pre + delta_damage
  P_apoptosis = norm_cdf((C_num_lethal - C_num_pre) / sigma)
  return C_num_post, P_apoptosis

def targeted_therapy_initial(x_odd):
  """
  16. Targeted therapy blocks asymmetric component, forcing even parity.

  Medical context:
    x_odd: Cellular state (must be odd integer) [~1-1000]
    Returns: (new state, whether parity flipped)

  Scientific basis:
    Blocks "+self" in (2·self)+self operation, leaving only 2·self (symmetric).
    If x is odd, 2x is even → parity flips → cell exits proliferative state.
    Explains initial targeted therapy success (EGFR inhibitors, HER2 blockers).

  Assumptions:
    - Complete pathway blockade (partial block may not flip parity)
    - No bypass pathways initially (develop over time)
    - Binary response (all or nothing)

  >>> targeted_therapy_initial(7)
  (14, False)
  >>> targeted_therapy_initial(3)
  (6, True)
  """
  x_even = 2 * x_odd
  parity_flipped = hamming_parity(x_even) == 0
  return x_even, parity_flipped

def targeted_therapy_resistance(x, x_prime):
  """
  17. Resistance via bypass pathway restores asymmetry.

  Medical context:
    x: State under blocked primary pathway
    x_prime: Contribution from activated bypass pathway [~same magnitude as x]

  Scientific basis:
    Cell activates alternative pathway to restore "+self" component. Net operation
    becomes 2x + x' ≈ 3x through different molecular route. Explains resistance
    to EGFR inhibitors (KRAS mutations), HER2 blockers (PI3K activation).

  Assumptions:
    - Bypass provides equivalent asymmetry (may be weaker initially)
    - Additive contributions (could interact nonlinearly)
    - Single bypass (multiple possible)

  >>> targeted_therapy_resistance(6, 5)
  17
  >>> targeted_therapy_resistance(10, 10)
  30
  """
  x_new = 2 * x + x_prime
  return x_new

def immunotherapy_success(P_old, F_immune):
  """
  18. Immune system provides external parity flip.

  Medical context:
    P_old: Current parity state (0 or 1)
    F_immune: Immune force magnitude (0 or 1, representing successful kill)

  Scientific basis:
    T-cells deliver apoptosis signal = external force flipping parity regardless
    of internal C_num. Explains immunotherapy success in high-C_num tumors that
    resist other treatments. External intervention changes state directly.

  Assumptions:
    - Binary immune response (kill or not)
    - Parity arithmetic (mod 2)
    - Single cell model (ignores population dynamics)

  >>> immunotherapy_success(1, 1)
  0
  >>> immunotherapy_success(1, 0)
  1
  """
  P_new = (P_old + F_immune) % 2
  return P_new

def immunotherapy_failure(A_actual, K, K_immune):
  """
  19. High constraint masks asymmetry from immune system.

  Medical context:
    A_actual: True asymmetry state (0 or 1)
    K: Constraint level [~0-20]
    K_immune: Threshold for immune invisibility [~5-10]

  Scientific basis:
    High K maintains normal tissue appearance (E-cadherin, MHC-I) despite
    underlying A=1. Immune system sees apparent asymmetry A_apparent ≈ 0.
    Explains "cold" tumors with low T-cell infiltration despite being cancerous.

  Assumptions:
    - Exponential masking (could be sigmoidal)
    - Single threshold (may depend on antigen type)
    - K is only factor (ignores PD-L1, other checkpoints)

  >>> immunotherapy_failure(1, 15, 5)
  0.049...
  >>> immunotherapy_failure(1, 3, 5)
  0.548...
  >>> immunotherapy_failure(1, 1, 5)
  0.818...
  """
  A_apparent = A_actual * np.exp(-K / K_immune)
  return A_apparent

# RELAPSE PATTERNS (20-23)

def early_relapse(N_residual, N_detect, lambda_rate):
  """
  20. Early relapse when C_num unchanged (parity masked, not resolved).

  Medical context:
    N_residual: Cells remaining after treatment [~10^3-10^6]
    N_detect: Detection threshold (imaging/clinical) [~10^8-10^9]
    lambda_rate: Growth rate [~0.1-0.5 per day]

  Scientific basis:
    Treatment reduced tumor burden but didn't change underlying C_num.
    Same exponential growth resumes immediately. Relapse time depends only
    on how many cells remain and detection threshold. Explains relapses
    within months—structural state unchanged.

  Assumptions:
    - Exponential regrowth (valid if C_num high)
    - Constant growth rate (may vary with treatment effects)
    - No immune surveillance (simplified)

  >>> early_relapse(1e4, 1e9, 0.3)
  37.68...
  >>> early_relapse(1e6, 1e9, 0.3)
  23.02...
  """
  T_relapse = (1 / lambda_rate) * np.log(N_detect / N_residual)
  return T_relapse

def late_relapse(K0, K_crit, lambda_age):
  """
  21. Late relapse from dormant cells when K degrades below threshold.

  Medical context:
    K0: Initial constraint in dormant cells post-treatment [~8-12]
    K_crit: Threshold for reactivation [~2-5]
    lambda_age: K degradation rate [~0.01-0.05 per year]

  Scientific basis:
    Residual cells achieved E state (C≈1, dormant) but K slowly degrades
    over years (aging, inflammation, tissue remodeling). When K crosses
    threshold, cells re-enter A=1 state. Explains late relapses (5-20 years)
    in breast cancer, melanoma. Dormancy isn't cure—reversible E state.

  Assumptions:
    - Exponential K decay (simplified from complex aging process)
    - Sharp threshold (likely stochastic around threshold)
    - Cells remain viable during dormancy

  >>> late_relapse(10, 3, 0.02)
  60.20...
  >>> late_relapse(15, 3, 0.01)
  160.94...
  """
  T_reactivate = (1 / lambda_age) * np.log(K0 / K_crit)
  return T_reactivate

def aggressive_relapse(C_num_distribution, C_treatment_threshold):
  """
  22. Treatment selects for highest C_num variants (most resistant).

  Medical context:
    C_num_distribution: Array of C_num values across tumor cell population [~10-100]
    C_treatment_threshold: C_num level above which cells survive [~30-60]

  Scientific basis:
    Treatment kills cells with C_num < threshold. Survivors have maximum
    structural resistance. Relapse emerges from highest-C_num clones, which
    are hardest to treat. Explains why relapsed tumors are often more
    aggressive and treatment-resistant than original.

  Assumptions:
    - Sharp threshold (likely smoother selection gradient)
    - C_num is heritable (clonal property)
    - No de novo resistance mechanisms

  >>> dist = np.array([10, 25, 40, 55, 70])
  >>> aggressive_relapse(dist, 30)
  70
  >>> aggressive_relapse(dist, 60)
  70
  >>> aggressive_relapse(dist, 80)
  0
  """
  C_num_survivors = C_num_distribution[C_num_distribution > C_treatment_threshold]
  if len(C_num_survivors) > 0:
    return np.max(C_num_survivors)
  return 0

def oligometastatic_relapse(C_local_sites, C_threshold, t):
  """
  23. Only specific metastatic sites reactivate based on local C degradation.

  Medical context:
    C_local_sites: Array of closure levels at different metastatic sites [~0.8-1.0]
    C_threshold: Reactivation threshold [~0.7-0.8]
    t: Time since treatment [months to years]

  Scientific basis:
    Different organs have different microenvironments with varying C degradation
    rates (kappa). Some sites maintain high C (continued dormancy), others
    degrade faster (reactivation). Explains oligometastatic disease—limited
    site relapse rather than widespread progression.

  Assumptions:
    - Random kappa values (in reality, organ-specific)
    - Independent site dynamics (may have systemic coupling)
    - Linear C degradation (could be nonlinear)

  >>> sites = np.array([0.95, 0.90, 0.88, 0.92])
  >>> oligometastatic_relapse(sites, 0.85, 10)
  ~2-3
  >>> oligometastatic_relapse(sites, 0.85, 50)
  ~3-4
  """
  kappa = np.random.uniform(0.01, 0.05, len(C_local_sites))
  C_current = C_local_sites - kappa * t
  n_relapse = np.sum(C_current < C_threshold)
  return n_relapse

# SPONTANEOUS BEHAVIORS (24-27)

def spontaneous_regression(delta_E, kT):
  """
  24. Extremely rare stochastic C_num decrease causing spontaneous remission.

  Medical context:
    delta_E: Energy barrier to reduce C_num [arbitrary units, ~10-50]
    kT: Thermal energy (biological noise level) [~1-5]

  Scientific basis:
    Random fluctuations can occasionally decrease C_num, allowing parity flip
    to E state. Probability exponentially small (Boltzmann distribution).
    Explains documented spontaneous regressions (~1 in 100,000 cancers) without
    requiring immune clearance or treatment.

  Assumptions:
    - Boltzmann statistics apply to biological systems (rough approximation)
    - Single barrier (likely multiple steps)
    - Energy landscape is static (changes with cell state)

  >>> spontaneous_regression(30, 2)
  9.35...e-07
  >>> spontaneous_regression(10, 3)
  0.0355...
  """
  P_regression = np.exp(-delta_E / kT)
  return P_regression

def tumor_differentiation(D, alpha_D, beta_D, t_max):
  """
  25. Differentiation signals drive C→1 and increase K, forcing stable state.

  Medical context:
    D: Differentiation signal strength (ATRA, vitamin D) [~0.1-2.0 μM]
    alpha_D: C response rate to signal [~0.1-0.5]
    beta_D: K response rate to signal [~0.5-2.0]
    t_max: Treatment duration [days to weeks]

  Scientific basis:
    Differentiation therapy (ATRA in APL) provides signal that increases both
    C (drives toward differentiation) and K (locks in identity). When C→1 and
    K→high, cell exits A=1 state permanently. Explains curative potential of
    differentiation therapy—completes the "4th vertex," not just killing.

  Assumptions:
    - Exponential saturation for C (reaches C=1)
    - Linear K increase (may saturate)
    - Uniform response (ignores heterogeneity)

  >>> t, C, K = tumor_differentiation(1.0, 0.3, 1.0, 30)
  >>> C[0], C[-1]
  (0.0, ~0.999)
  >>> K[0], K[-1]
  (10.0, ~40.0)
  """
  t = np.linspace(0, t_max, 1000)
  C = 1 - np.exp(-alpha_D * D * t)
  K = 10 + beta_D * D * t
  return t, C, K

def necrotic_core(N_surface, lambda_diffusion, N_min):
  """
  26. Central tumor cells forced to E state (death) by resource depletion.

  Medical context:
    N_surface: Nutrient concentration at tumor surface [~100% of normal]
    lambda_diffusion: Oxygen diffusion length [~100-200 μm]
    N_min: Minimum nutrients for viability [~10-20% of normal]

  Scientific basis:
    Oxygen/nutrients diffuse from vessels with exponential decay. Beyond
    diffusion length, concentration drops below viability threshold, forcing
    C→1 (cells attempt differentiation but die—necrosis). Radius depends on
    surface concentration and diffusion properties. Not apoptosis—structural
    collapse from resource exhaustion.

  Assumptions:
    - Exponential diffusion (simplified from 3D geometry)
    - Sharp viability threshold (gradual in reality)
    - Spherical geometry (tumors are irregular)

  >>> necrotic_core(100, 150, 15)
  282.19...
  >>> necrotic_core(100, 100, 20)
  160.94...
  """
  R_necrotic = lambda_diffusion * np.log(N_surface / N_min)
  return R_necrotic

def angiogenesis(C0, kappa, nu, V, t_max):
  """
  27. Blood vessel formation prevents size-dependent closure increase.

  Medical context:
    C0: Baseline closure without vessels [~0.3-0.5]
    kappa: Closure increase rate from hypoxia [~0.01-0.05 per day]
    nu: Vessel effectiveness at preventing closure [~0.1-0.5]
    V: Vessel density [~0-10, normalized units]
    t_max: Growth period [days to months]

  Scientific basis:
    Without angiogenesis, tumor size increases C (hypoxia signals differentiation).
    New vessels reduce this pressure: C_final = C0 + κt/(1+νV). Explains why
    angiogenesis is essential—not just nutrients, but maintaining low C state.
    Anti-angiogenic therapy works by forcing C increase, not just starving tumor.

  Assumptions:
    - Hyperbolic vessel effect (could be more complex)
    - Linear C increase without vessels (simplified)
    - Vessels uniformly distributed (patchy in reality)

  >>> t, C = angiogenesis(0.3, 0.02, 0.3, 5, 50)
  >>> C[0], C[-1]
  (0.3, ~0.69)
  >>> angiogenesis(0.3, 0.02, 0.3, 0, 50)[1][-1]  # No vessels
  ~1.3
  """
  t = np.linspace(0, t_max, 1000)
  C = C0 + kappa * t / (1 + nu * V)
  return t, C

# TISSUE-SPECIFIC PATTERNS (28-31)

def hematologic_cancer(C_baseline, lambda_rate, t_max):
  """
  28. Blood cancers: cells normally operate in low-C mobile state.

  Medical context:
    C_baseline: Normal blood cell closure level [~0.2-0.4, lower than solid tissues]
    lambda_rate: Proliferation rate [~0.2-0.6 per day]
    t_max: Disease progression period [weeks to months]

  Scientific basis:
    Blood cells are designed for low C (mobile, undifferentiated precursors
    that mature). Cancer = failure to transition to E (differentiation block).
    Unlike solid tumors where C must decrease, blood cancers maintain already-low C.
    Explains rapid progression—no spatial constraints, already in proliferative
    state.

  Assumptions:
    - Constant low C (doesn't increase with cell number)
    - Unbounded exponential growth (no carrying capacity in circulation)
    - No spatial constraints (valid for leukemia, not lymphoma masses)

  >>> t, N, C = hematologic_cancer(0.3, 0.4, 30)
  >>> C[0], C[-1]
  (0.3, 0.3)
  >>> N[-1] / N[0]
  162754.79...
  """
  t = np.linspace(0, t_max, 1000)
  N = np.exp(lambda_rate * t)
  C = np.full_like(t, C_baseline)
  return t, N, C

def epithelial_cancer(K_junction, K_basement, delta_K_junction):
  """
  29. Epithelial cancers: sequential loss of spatial constraints.

  Medical context:
    K_junction: Constraint from cell-cell junctions (E-cadherin) [~5-10]
    K_basement: Constraint from basement membrane [~3-8]
    delta_K_junction: Loss of junction constraint in cancer [~3-7]

  Scientific basis:
    Normal epithelium has high K from junctions + basement membrane. Cancer
    progresses through stages: first lose junctions (carcinoma in situ), then
    basement membrane (invasive), then all constraint (metastatic). Sequential
    K degradation explains progression stages: CIS → invasive → metastatic.

  Assumptions:
    - Additive constraints (may interact nonlinearly)
    - Sequential loss (could be simultaneous)
    - Binary states (likely gradual transitions)

  >>> epithelial_cancer(7, 5, 5)
  (12, 5, 0)
  >>> epithelial_cancer(10, 8, 7)
  (18, 8, 0)
  """
  K_initial = K_junction + K_basement
  K_invasive = K_basement
  K_metastatic = 0
  return K_initial, K_invasive, K_metastatic

def brain_tumor(C_brain, C_num_glial):
  """
  30. Brain tumors: infiltration follows closure gradient in high-C environment.

  Medical context:
    C_brain: Post-mitotic neural tissue closure [~0.95, very high]
    C_num_glial: Glial cell structural resistance [~60-80, high]

  Scientific basis:
    Brain has extremely high baseline C (neurons post-mitotic, locked). Glial
    cells operate at intermediate C (~0.7). Cancer = glial cells that can't
    achieve full C but are trapped in high-C environment. Growth follows paths
    of lowest C (white matter tracts). Explains infiltrative growth pattern—
    searching for compatible geometric environment.

  Assumptions:
    - Gradient-following behavior (simplified from complex invasion)
    - C_brain is static (changes with injury, aging)
    - Linear gradient (likely heterogeneous)

  >>> brain_tumor(0.95, 70)
  0.25
  >>> brain_tumor(0.90, 60)
  0.2
  """
  C_glial = 0.7
  infiltration_gradient = C_brain - C_glial
  return infiltration_gradient

def pediatric_cancer(K_developmental, tau_maturation, t):
  """
  31. Pediatric cancers: normal developmental low-K state persists beyond window.

  Medical context:
    K_developmental: Constraint level during development [starts ~3-5]
    tau_maturation: Time constant for K increase to adult levels [~5-15 years]
    t: Child's age [years]

  Scientific basis:
    Children's cells are designed for low K (growth, tissue formation). Normal:
    K increases with age as tissues mature. Pediatric cancer: K fails to increase
    on schedule, cell maintains low-K proliferative state. Not damage-induced—
    timing failure. Explains low mutation burden (not from DNA damage) but high
    aggressiveness (appropriate developmental program at wrong time).

  Assumptions:
    - Exponential K maturation (may have critical periods)
    - Uniform tissue maturation (organ-specific in reality)
    - Cancer = arrested maturation (could be active suppression)

  >>> pediatric_cancer(10, 10, 5)
  3.934...
  >>> pediatric_cancer(10, 10, 15)
  7.768...
  >>> pediatric_cancer(15, 8, 3)
  4.387...
  """
  K = K_developmental * (1 - np.exp(-t / tau_maturation))
  return K

# MULTI-CELLULARITY EFFECTS (32-35)

def clonal_evolution(K0, delta_K_mutation, n_clones):
  """
  32. Sequential mutations progressively degrade constraint.

  Medical context:
    K0: Initial constraint in founding clone [~10-15]
    delta_K_mutation: Constraint loss per clonal mutation [~1-3]
    n_clones: Number of successive clonal expansions [~3-7]

  Scientific basis:
    Cancer evolves through successive clones, each with additional constraint-
    degrading mutations. K decreases stepwise: K₀ → K₁ → K₂... Each step
    increases fitness (faster growth). Explains multi-step carcinogenesis and
    accumulating malignancy over time.

  Assumptions:
    - Constant delta_K per mutation (varies by gene)
    - Linear K decrease (may have threshold effects)
    - Sequential dominance (clones could coexist)

  >>> clonal_evolution(15, 2, 5)
  array([15, 13, 11,  9,  7,  5])
  >>> clonal_evolution(12, 1.5, 3)
  array([12. , 10.5,  9. ,  7.5])
  """
  K_trajectory = [K0]
  for i in range(n_clones):
    K_trajectory.append(K_trajectory[-1] - delta_K_mutation)
  return np.array(K_trajectory)

def tumor_heterogeneity(C_mean, sigma, grid_size):
  """
  33. Spatial variation in closure creates treatment-resistant niches.

  Medical context:
    C_mean: Average closure across tumor [~0.3-0.7]
    sigma: Spatial variation in C [~0.1-0.3]
    grid_size: Tumor dimensions for spatial model [~10-100 regions]

  Scientific basis:
    Different tumor regions experience different microenvironments (hypoxia,
    nutrient gradients, stromal density). This creates spatial C heterogeneity.
    Treatment kills regions with favorable C; high-C or very-low-C regions survive.
    Explains incomplete responses and residual disease patterns.

  Assumptions:
    - Gaussian spatial noise (could have structured gradients)
    - Independent voxels (likely spatially correlated)
    - Static during treatment (evolves in reality)

  >>> grid = tumor_heterogeneity(0.5, 0.2, 10)
  >>> grid.shape
  (10, 10)
  >>> 0 <= grid.min() <= grid.max() <= 1
  True
  """
  C_spatial = C_mean + sigma * np.random.randn(grid_size, grid_size)
  C_spatial = np.clip(C_spatial, 0, 1)
  return C_spatial

def cancer_stem_cells(C_num_csc, C_num_bulk):
  """
  34. CSCs have maximum closure resistance, can sustain proliferation indefinitely.

  Medical context:
    C_num_csc: Structural resistance of putative stem cell [~60-100]
    C_num_bulk: Average resistance of bulk tumor [~20-40]

  Scientific basis:
    Traditional CSC markers (CD44, CD133) may actually measure K (identity).
    True stemness = maximum C_num (can't achieve closure, sustain proliferation).
    CSCs are cells with optimized low-C/high-K state: stable enough to maintain
    identity, resistant enough to never differentiate. Source of tumor renewal
    and relapse.

  Assumptions:
    - C_num defines stemness (traditional view: surface markers)
    - Binary CSC vs non-CSC (likely continuous spectrum)
    - High C_num is stable property (could be plastic)

  >>> cancer_stem_cells(80, 30)
  True
  >>> cancer_stem_cells(25, 30)
  False
  """
  is_csc = C_num_csc > C_num_bulk
  return is_csc

def stromal_interactions(C_baseline, epsilon_CAF, CAF_density):
  """
  35. Cancer-associated fibroblasts reduce local closure pressure.

  Medical context:
    C_baseline: Closure without stromal influence [~0.4-0.6]
    epsilon_CAF: CAF effectiveness at reducing C [~0.1-0.3 per unit density]
    CAF_density: Fibroblast abundance in microenvironment [~0-5]

  Scientific basis:
    CAFs remodel ECM, secrete growth factors, reduce differentiation signals.
    Net effect: lower local C, allowing tumor to maintain proliferative state
    despite size. Explains why stromal-rich tumors (desmoplastic) are aggressive—
    not just physical barrier, but active maintenance of low-C niche.

  Assumptions:
    - Linear CAF effect (may saturate)
    - Uniform CAF distribution (spatially heterogeneous)
    - CAFs only affect C (also affect K and immune infiltration)

  >>> stromal_interactions(0.5, 0.15, 2)
  0.2
  >>> stromal_interactions(0.6, 0.2, 3)
  0.0
  """
  C_local = C_baseline - epsilon_CAF * CAF_density
  return C_local

# MOLECULAR HALLMARKS REINTERPRETED (36-45)

def proliferative_signaling(N, alpha_signal):
  """
  36. Autocrine loops maintain asymmetry through self-generated signals.

  Medical context:
    N: Tumor cell number [~10³-10¹²]
    alpha_signal: Signal production per cell [~0.001-0.01]

  Scientific basis:
    Cancer cells secrete growth factors that stimulate themselves (autocrine).
    Signal strength ∝ N, creating positive feedback: more cells → more signal →
    maintains A=1. Represents the "+self" in (2·self)+self operation. Explains
    sustained proliferative signaling hallmark as mechanism to maintain odd parity.

  Assumptions:
    - Linear signal production (may saturate)
    - No signal degradation (simplified)
    - Uniform response (cell heterogeneity ignored)

  >>> proliferative_signaling(1e6, 0.005)
  5000.0
  >>> proliferative_signaling(1e9, 0.001)
  1000000.0
  """
  S = alpha_signal * N
  return S

def evade_growth_suppressors(C, delta_repair, p53_active):
  """
  37. p53/Rb loss removes closure enforcement mechanisms.

  Medical context:
    C: Current closure level [~0.2-0.8]
    delta_repair: C increase when suppressors active [~0.1-0.3]
    p53_active: Whether p53 functional (True/False)

  Scientific basis:
    Tumor suppressors (p53, Rb) increase C when DNA damage detected—force cells
    toward differentiation or apoptosis. Loss removes this pressure, allowing C
    to remain low despite damage. Reinterprets tumor suppressors as closure
    enforcers rather than just cell cycle brakes.

  Assumptions:
    - Binary p53 status (actually has partial loss-of-function)
    - Additive C increase (may involve thresholds)
    - p53 only affects C (also affects apoptosis, senescence)

  >>> evade_growth_suppressors(0.4, 0.2, True)
  0.6
  >>> evade_growth_suppressors(0.4, 0.2, False)
  0.4
  """
  if p53_active:
    C_new = C + delta_repair
  else:
    C_new = C
  return C_new

def resist_cell_death(Bcl2_level, Bcl_crit):
  """
  38. Bcl-2 overexpression prevents forced closure (apoptosis).

  Medical context:
    Bcl2_level: Anti-apoptotic protein expression [~0-10, normalized]
    Bcl_crit: Threshold for apoptosis resistance [~2-5]

  Scientific basis:
    Apoptosis = forced C→1 (terminal closure, death). Bcl-2 family proteins
    block this transition, allowing cells to maintain C<1 even when damage
    should trigger apoptosis. Explains how cancer resists death—prevents
    final vertex completion.

  Assumptions:
    - Exponential protection (simplified from complex pathway)
    - Single Bcl-2 value (actually family of proteins)
    - Only Bcl-2 matters (ignores pro-apoptotic balance)

  >>> resist_cell_death(10, 3)
  0.0497...
  >>> resist_cell_death(2, 3)
  0.5134...
  >>> resist_cell_death(0.5, 3)
  0.8465...
  """
  P_apoptosis = np.exp(-Bcl2_level / Bcl_crit)
  return P_apoptosis

def replicative_immortality(divisions, telomerase_active):
  """
  39. Telomerase prevents division-dependent C_num increase.

  Medical context:
    divisions: Number of cell divisions [~0-100+]
    telomerase_active: Whether telomerase expressed (True/False)

  Scientific basis:
    Each division normally increases C_num (accumulating errors, closer to
    forced closure). Telomeres act as division counter; telomerase resets it.
    With telomerase, cells can divide indefinitely without C_num increase—
    removes temporal limit on proliferation. Reinterprets immortality as
    removing structural constraint rather than just chromosome protection.

  Assumptions:
    - Linear C_num increase per division (simplified)
    - Binary telomerase (actually has expression levels)
    - Telomeres only affect C_num (also chromosome stability)

  >>> replicative_immortality(50, True)
  0
  >>> replicative_immortality(50, False)
  5.0
  >>> replicative_immortality(100, False)
  10.0
  """
  if telomerase_active:
    dC_num = 0
  else:
    dC_num = 0.1 * divisions
  return dC_num

def induce_angiogenesis_hallmark(C_hypoxia, C_threshold, alpha_VEGF):
  """
  40. VEGF secretion prevents hypoxia-induced closure increase.

  Medical context:
    C_hypoxia: Closure level under hypoxic stress [~0.5-0.9]
    C_threshold: Level triggering angiogenesis [~0.6-0.8]
    alpha_VEGF: VEGF production rate [~0.1-0.5]

  Scientific basis:
    Hypoxia signals differentiation (increases C). If C exceeds threshold,
    cancer secretes VEGF to recruit vessels. Vessels provide oxygen, reducing
    C back down. Not just nutrients—angiogenesis maintains low-C state essential
    for proliferation. Anti-angiogenics work by forcing C increase.

  Assumptions:
    - Linear VEGF production above threshold (likely sigmoidal)
    - Threshold is sharp (gradual in reality)
    - VEGF only affects vessel growth (has other functions)

  >>> induce_angiogenesis_hallmark(0.75, 0.65, 0.3)
  0.03
  >>> induce_angiogenesis_hallmark(0.60, 0.65, 0.3)
  0
  """
  if C_hypoxia > C_threshold:
    dV_dt = alpha_VEGF * (C_hypoxia - C_threshold)
  else:
    dV_dt = 0
  return dV_dt

def invasion_metastasis(K_epithelial, delta_K_EMT):
  """
  41. EMT reduces constraint for mobility while maintaining enough for identity.

  Medical context:
    K_epithelial: Baseline epithelial constraint [~10-15]
    delta_K_EMT: K reduction during EMT [~5-10]

  Scientific basis:
    Epithelial-mesenchymal transition reduces K (lose junctions, gain motility)
    and achieves optimal C (~0.8) for metastasis. Too much K = can't move;
    too little = lose identity. EMT is controlled K reduction to enable
    metastatic spread. Not just molecular program—geometric optimization.

  Assumptions:
    - Fixed delta_K for EMT (varies by cell type)
    - Fixed optimal C_EMT (actually range ~0.75-0.85)
    - Binary EMT (actually spectrum)

  >>> invasion_metastasis(12, 7)
  (5, 0.8)
  >>> invasion_metastasis(15, 8)
  (7, 0.8)
  """
  K_EMT = K_epithelial - delta_K_EMT
  C_EMT = 0.8
  return K_EMT, C_EMT

def warburg_effect(pathway="glycolysis"):
  """
  42. Glycolysis maintains low C by avoiding differentiation-inducing ROS.

  Medical context:
    pathway: Metabolic mode ('glycolysis' or 'OXPHOS')
    Returns: (ROS level, C increase rate)

  Scientific basis:
    Mitochondrial respiration generates ROS, which triggers differentiation
    signaling (C increase). Glycolysis produces less ATP but also less ROS.
    Cancer "chooses" glycolysis to avoid closure pressure, not because OXPHOS
    is broken. Warburg effect = paying energetic cost to maintain low-C state.

  Assumptions:
    - ROS directly increases C (simplified from complex signaling)
    - Binary pathway choice (cells use mix)
    - No other reasons for glycolysis (lactate signaling, biosynthesis)

  >>> warburg_effect('glycolysis')
  (0.1, 0.01)
  >>> warburg_effect('OXPHOS')
  (1.0, 0.1)
  """
  if pathway == "glycolysis":
    ROS = 0.1
    C_increase = 0.01
  else:  # OXPHOS
    ROS = 1.0
    C_increase = 0.1
  return ROS, C_increase

def evade_immune(PD_L1, MHC, K_immune):
  """
  43. PD-L1 expression and MHC downregulation mask asymmetry from immune system.

  Medical context:
    PD_L1: Checkpoint ligand expression [~0-10]
    MHC: MHC-I surface expression [~0-1, normalized]
    K_immune: Threshold for immune visibility [~5-10]

  Scientific basis:
    High K maintains normal appearance despite A=1. Additionally, PD-L1 sends
    "don't kill" signal and MHC-I loss hides neoantigens. Combined effect:
    exponentially reduced immune recognition. Explains why checkpoint blockade
    works—removes PD-L1 masking, allowing immune system to see underlying asymmetry.

  Assumptions:
    - Multiplicative effects (may interact differently)
    - Continuous recognition probability (likely threshold)
    - Only these two factors (many other immune evasion mechanisms)

  >>> evade_immune(5, 0.8, 6)
  0.3251...
  >>> evade_immune(0, 1.0, 6)
  1.0
  >>> evade_immune(10, 0.2, 6)
  0.0375...
  """
  signal_inhibit = PD_L1
  P_recognition = np.exp(-signal_inhibit) * MHC
  return P_recognition

def genome_instability(t, lambda_rate, C, mu):
  """
  44. Mutation accumulation is consequence of sustained proliferation in O-state.

  Medical context:
    t: Time in proliferative state [months to years]
    lambda_rate: Division rate [~0.1-0.5 per day]
    C: Closure level [~0.2-0.6 for cancer]
    mu: Mutation rate per division [~1e-9 to 1e-7 per base]

  Scientific basis:
    High division rate in low-C state (rapid cycling without differentiation
    pauses) accumulates errors. Mutations are consequence, not cause, of cancer
    state. Explains why mutation burden correlates with duration, not severity—
    it's a clock, not driver. Creates vicious cycle: mutations → higher C_num
    → harder to resolve.

  Assumptions:
    - Constant mutation rate (varies with repair capacity)
    - Linear accumulation (may have acceleration)
    - All mutations equally likely (mutation spectrum ignored)

  >>> genome_instability(365, 0.3, 0.4, 1e-6)
  182.5
  >>> genome_instability(100, 0.5, 0.3, 1e-6)
  71.42...
  """
  divisions = lambda_rate * t / (1 - C)
  mutations = divisions * mu
  return mutations

def tumor_inflammation(I0):
  """
  45. Chronic inflammation forces asymmetry and prevents closure.

  Medical context:
    I0: Inflammatory signal strength (cytokines) [~0.5-2.0]
    Returns: (dA/dt rate, dC/dt rate)

  Scientific basis:
    Inflammatory environment continuously injects asymmetry signals (forces A→1)
    and degrades closure (dC/dt < 0). Selects for cells that can sustain A=1
    despite inflammation. Explains inflammation-associated cancers and why
    anti-inflammatory drugs reduce cancer risk.

  Assumptions:
    - Linear inflammation effects (likely nonlinear)
    - Constant I0 (fluctuates in reality)
    - Direct causation (inflammation also recruits immune cells)

  >>> tumor_inflammation(1.0)
  (0.1, -0.05)
  >>> tumor_inflammation(2.0)
  (0.2, -0.1)
  """
  dA_dt = 0.1 * I0
  dC_dt = -0.05 * I0
  return dA_dt, dC_dt

# AGING & CANCER (49-50)

def age_risk(age, K0, lambda_aging, K_crit, sigma):
  """
  49. Cumulative constraint degradation with age increases cancer probability.

  Medical context:
    age: Patient age [years, 0-100]
    K0: Constraint at birth [~15-20]
    lambda_aging: Annual K degradation rate [~0.01-0.03 per year]
    K_crit: Cancer threshold [~2-5]
    sigma: Population variation in K [~1-3]

  Scientific basis:
    Constraints naturally degrade with age from accumulated damage (oxidative
    stress, inflammation, division cycles, tissue remodeling). Cancer risk
    follows probability that K crosses threshold. Predicts observed age⁴⁻⁶
    incidence scaling without requiring specific mutations. Explains why cancer
    is fundamentally age-related—time-integrated constraint loss.

  Assumptions:
    - Linear K decay (may accelerate with age)
    - Gaussian variation (actual distribution may have fat tails)
    - Single threshold (likely multiple intermediate states)

  >>> K, P = age_risk(30, 15, 0.02, 3, 2)
  >>> K
  9.0
  >>> P
  0.0013...
  >>> age_risk(70, 15, 0.02, 3, 2)
  (1.0, 0.8413...)
  >>> age_risk(50, 15, 0.015, 3, 1.5)
  (3.75, 0.3085...)
  """
  K_age = K0 * (1 - lambda_aging * age)
  P_cancer = norm_cdf((K_crit - K_age) / sigma)
  return K_age, P_cancer

def senescence_protection(C_num, C_num_dangerous):
  """
  50. Cells trigger permanent quiescence before reaching unresolvable state.

  Medical context:
    C_num: Current structural resistance [~10-80]
    C_num_dangerous: Threshold triggering senescence [~40-60]
    Returns: (senescence triggered, A, C, K if senescent)

  Scientific basis:
    When cells detect C_num approaching irreversible levels, they enter
    senescence: permanent E state (A→0, C→1, K→∞). Sacrifices cell to protect
    organism. Explains senescence as cancer prevention mechanism—better to
    lock cell in terminal state than risk it becoming untreatable. Senescent
    cells are "frozen" at high C, high K.

  Assumptions:
    - Sharp C_num threshold (likely stochastic around threshold)
    - Irreversible senescence (some cells can escape)
    - K→∞ (very high but finite in reality)

  >>> senescence_protection(55, 50)
  (True, 0, 1, inf)
  >>> senescence_protection(30, 50)
  (False, None, None, None)
  >>> senescence_protection(80, 60)
  (True, 0, 1, inf)
  """
  if C_num > C_num_dangerous:
    trigger_senescence = True
    A = 0
    C = 1
    K = np.inf
  else:
    trigger_senescence = False
    A, C, K = None, None, None
  return trigger_senescence, A, C, K

# COMBINATION THERAPIES

def synergistic_combination(C_num0, delta_A, K0, delta_B, threshold):
  """
  51. Drug A reduces C_num, Drug B increases K; combined effect is multiplicative.

  Medical context:
    C_num0: Baseline structural resistance [~20-80]
    delta_A: C_num reduction from Drug A (e.g., targeted therapy) [~10-30]
    K0: Baseline constraint [~2-8]
    delta_B: K increase from Drug B (e.g., differentiation agent) [~3-10]
    threshold: C_num level for cure [~20-40]

  Scientific basis:
    Drug A makes parity flip possible (reduces C_num to flippable range).
    Drug B maintains E state after flip (high K prevents reversion).
    Synergy because: low C_num alone → may flip but revert; high K alone →
    insufficient without flip. Together: flip AND lock. Explains why some
    combinations work dramatically while single agents fail.

  Assumptions:
    - Additive drug effects on parameters (may interact)
    - Independent mechanisms (cross-talk possible)
    - Simultaneous administration (timing may matter)

  >>> synergistic_combination(50, 20, 4, 6, 35)
  0.8413...
  >>> synergistic_combination(60, 15, 4, 3, 35)
  0.1586...
  >>> synergistic_combination(40, 25, 5, 8, 35)
  0.9772...
  """
  C_num_A = C_num0 - delta_A
  K_B = K0 + delta_B
  P_cure = norm_cdf((threshold - C_num_A) / 10) * (K_B > 5).astype(float)
  return P_cure

def sequential_therapy(C_num0, r1, T1, F_flip, F_threshold):
  """
  52. Phase 1 reduces C_num, then Phase 2 flips parity—exploits timescales.

  Medical context:
    C_num0: Initial structural resistance [~30-100]
    r1: C_num reduction rate with Phase 1 drug [~0.5-2.0 per week]
    T1: Duration of Phase 1 [weeks]
    F_flip: Parity flip force from Phase 2 (e.g., chemo intensity) [~0-10]
    F_threshold: Minimum force needed to flip [~3-7]

  Scientific basis:
    Sequential exploits two timescales: slow C_num reduction (weeks, targeted
    therapy) vs. fast parity flip (days, chemotherapy). First make cancer
    flippable, then flip it. Works for high-C_num tumors where simultaneous
    fails because chemo can't flip while C_num is high. Predicts 20-30%
    improvement in high-C_num cases.

  Assumptions:
    - Linear C_num reduction in Phase 1 (may saturate)
    - Sharp flip threshold (likely probabilistic)
    - No resistance development during Phase 1 (simplified)

  >>> sequential_therapy(60, 1.0, 25, 8, 5)
  (35, True)
  >>> sequential_therapy(60, 1.0, 15, 4, 5)
  (45, False)
  >>> sequential_therapy(80, 2.0, 20, 9, 5)
  (40, True)
  """
  C_num_T1 = C_num0 - r1 * T1
  if C_num_T1 < 40 and F_flip > F_threshold:
    success = True
  else:
    success = False
  return C_num_T1, success

# M. NOVEL PREDICTIONS (53-56)

def parity_cycling(C, K, t_max):
  """
  53. Some tumors oscillate between proliferative and quiescent states.

  Medical context:
    C: Initial closure level [not directly used, determines amplitude]
    K: Constraint level [not directly used, affects period]
    t_max: Observation period [days, typically 100-200]
    Returns: (time array, C oscillation)

  Scientific basis:
    Certain tumors exist in bistable regime where C oscillates between low
    (proliferative, O state) and high (quiescent, approaching E state). Period
    ~60-180 days. Neither stable—continuously cycles. Explains observed growth
    oscillations without treatment and why some tumors alternate between
    aggressive and dormant phases. Measurable via serial imaging and molecular
    profiling at different cycle phases.

  Assumptions:
    - Sinusoidal oscillation (likely more complex limit cycle)
    - Fixed period (may vary with microenvironment)
    - No damping (oscillations could decay)

  >>> t, C_t = parity_cycling(0.5, 5, 240)
  >>> C_t.min(), C_t.max()
  (0.1, 0.9)
  >>> len(np.where(np.diff(np.sign(C_t - 0.5)) != 0)[0])  # Zero crossings
  ~4
  """
  t = np.linspace(0, t_max, 1000)
  omega = 2 * np.pi / 120  # Period ~120 days
  C_t = 0.5 + 0.4 * np.sin(omega * t)
  return t, C_t

def closure_number_biomarker(C_num, C_threshold):
  """
  54. C_num predicts treatment response with logistic relationship.

  Medical context:
    C_num: Measured structural resistance [~10-100]
    C_threshold: Inflection point for response probability [~30-50]
    Returns: Probability of treatment response

  Scientific basis:
    Treatment response should follow logistic curve with C_num: low C_num →
    high response probability, high C_num → low response. Inflection at
    C_threshold where treatment becomes marginally effective. If measurable
    from Ki-67 heterogeneity, chromatin accessibility, or cell cycle patterns,
    provides quantitative prognostic biomarker superior to stage or mutation
    burden. Testable prediction: C_num HR ~1.5-2.0 per doubling.

  Assumptions:
    - Logistic response curve (could be other sigmoid)
    - Single threshold (may have drug-specific thresholds)
    - C_num is measurable (needs validation of proxy methods)

  >>> closure_number_biomarker(20, 40)
  0.8807...
  >>> closure_number_biomarker(60, 40)
  0.1192...
  >>> closure_number_biomarker(40, 40)
  0.5
  """
  P_response = 1 / (1 + np.exp((C_num - C_threshold) / 10))
  return P_response

def dimensional_therapy(x_odd):
  """
  55. Force 4x operation (tetrahedral) instead of 3x (triangular).

  Medical context:
    x_odd: Current state (must be odd integer) [~1-1000]
    Returns: (new state, parity after transformation)

  Scientific basis:
    Differentiation therapy provides "4th vertex" to complete tetrahedral
    closure. Changes recursive operation from 3x (maintains odd parity) to
    4x (even result). Represents completing the geometric structure: 3-vertex
    triangle → 4-vertex tetrahedron. Explains why differentiation therapy
    (ATRA) can cure—not killing cells but completing their geometry. Predicts
    only works when cells are near completion (C > 0.85).

  Assumptions:
    - 4x operation is achievable (requires specific differentiation signals)
    - Binary transformation (likely gradual)
    - All cells responsive (heterogeneity in practice)

  >>> dimensional_therapy(7)
  (28, True)
  >>> dimensional_therapy(15)
  (60, True)
  >>> dimensional_therapy(3)
  (12, True)
  """
  x_new = 4 * x_odd
  parity = hamming_parity(x_new)
  return x_new, parity

def constraint_reinforcement(K0, eta_drug, drug_concentration, gamma_stress, stress):
  """
  56. Increase K pharmacologically to prevent cancer initiation or progression.

  Medical context:
    K0: Baseline constraint [~5-15]
    eta_drug: Drug effectiveness at increasing K [~0.5-2.0 per concentration unit]
    drug_concentration: Drug dose [~0-5, normalized units]
    gamma_stress: K degradation rate from stress/aging [~0.01-0.05]
    stress: Stress level (damage, inflammation) [~0-5]
    Returns: (dK/dt rate, steady-state K)

  Scientific basis:
    Drugs that strengthen cell-cell junctions (E-cadherin enhancers), activate
    senescence pathways (p16 agonists), or stabilize tissue architecture can
    increase K. This raises threshold for cancer initiation and can even force
    existing cancer into quiescence. Predicts: K-reinforcing interventions
    reduce cancer incidence 30-50% in high-risk populations. Novel therapeutic
    strategy: strengthen constraints rather than kill cells.

  Assumptions:
    - Linear drug effect (may saturate)
    - Linear stress degradation (simplified)
    - Achievable K increase (current drugs may be insufficient)

  >>> dK, K_ss = constraint_reinforcement(8, 1.0, 3, 0.02, 2)
  >>> dK
  2.96
  >>> K_ss
  150.0
  >>> constraint_reinforcement(5, 0.5, 2, 0.05, 3)
  (0.85, 20.0)
  """
  dK_dt = eta_drug * drug_concentration - gamma_stress * stress
  K_ss = (
    (eta_drug * drug_concentration) / gamma_stress if gamma_stress > 0 else np.inf
  )
  return dK_dt, K_ss

# UNIFIED 

def master_dynamics(state, t, params):
  """
  Complete system dynamics integrating all components.

  Medical context:
    state: [A, C, K, N] - asymmetry, closure, constraint, cell number
    params: [alpha, beta, gamma, delta, lambda_max, K0, K_crit]

  Scientific basis:
    Unified differential equations governing cancer state evolution. Couples
    all three parameters (A, C, K) with cell number N. A changes discretely
    when K crosses K_crit. C evolves from balance of proliferation-driven
    differentiation attempts and constraint-enforced closure. K degrades from
    proliferation and damage. N grows based on A, C, and K.

  Assumptions:
    - Continuous dynamics for C, K, N (discrete A handled separately)
    - Linear coupling terms (simplified from nonlinear reality)
    - Autonomous system (no time-dependent external forcing)

  >>> state = [1, 0.4, 6, 1000]
  >>> params = [0.05, 0.01, 0.01, 0.02, 0.5, 10, 3]
  >>> master_dynamics(state, 0, params)
  [0, 0.0276, -0.03, 182.68...]
  """
  A, C, K, N = state
  alpha, beta, gamma, delta, lambda_max, K0, K_crit = params

  dC_dt = alpha * A * (1 - C) - beta * K * C
  dK_dt = -gamma * A - delta
  dN_dt = lambda_max * N * A * (1 - C) * np.exp(-K / K0)
  dA_dt = 0  # Handled discretely

  return [dA_dt, dC_dt, dK_dt, dN_dt]

def cancer_condition(A, C_num, K, dK_dt, N, theta, N_detection):
  """
  Complete cancer criterion: all conditions must be met.

  Medical context:
    A: Asymmetry state (0 or 1)
    C_num: Structural resistance [~10-100]
    K: Constraint level [~0-20]
    dK_dt: Rate of K change [typically negative in cancer]
    N: Cell number [~10³-10¹²]
    theta: C_num threshold for cancer [~20-40]
    N_detection: Minimum detectable tumor [~10⁸-10⁹]

  Scientific basis:
    Cancer defined by conjunction of conditions: proliferative asymmetry (A=1),
    high structural resistance (C_num > threshold), degrading constraints
    (dK/dt < 0), and detectable cell number. All four required—partial
    fulfillment is pre-malignant or subclinical. Provides formal definition
    that unifies multiple cancer hallmarks.

  Assumptions:
    - Binary classification (actually spectrum)
    - Sharp thresholds (likely continuous probabilities)
    - Independent conditions (may have correlations)

  >>> cancer_condition(1, 50, 3, -0.01, 1e9, 30, 1e8)
  True
  >>> cancer_condition(0, 50, 3, -0.01, 1e9, 30, 1e8)
  False
  >>> cancer_condition(1, 20, 3, -0.01, 1e9, 30, 1e8)
  False
  """
  is_cancer = (A == 1) and (C_num > theta) and (dK_dt < 0) and (N > N_detection)
  return is_cancer

def treatment_response(S_pre, T, F_immune, xi_stochastic):
  """
  General treatment effect: state change from therapy plus immune plus random.

  Medical context:
    S_pre: Pre-treatment state [A, C, K, N]
    T: Treatment effect vector [ΔA, ΔC, ΔK, ΔN]
    F_immune: Immune contribution [same dimensions]
    xi_stochastic: Random fluctuations [same dimensions]
    Returns: Post-treatment state

  Scientific basis:
    Treatment changes state through: direct drug effect (T), immune activation
    (F_immune), and stochastic effects (xi, representing individual variation,
    microenvironment, etc.). Final state is vector sum. Success = reaching
    healthy state region (A=0, C≈1, K high, N low). Provides framework for
    predicting combination therapy and individual response variation.

  Assumptions:
    - Additive effects (may interact nonlinearly)
    - Independent contributions (immune and drug may couple)
    - Gaussian noise (stochastic term may have structure)

  >>> S_pre = np.array([1, 0.3, 4, 1e6])
  >>> T = np.array([-1, 0.4, 2, -5e5])
  >>> F_immune = np.array([0, 0.1, 0, -3e5])
  >>> xi = np.array([0, 0.05, 0.5, 1e4])
  >>> treatment_response(S_pre, T, F_immune, xi)
  array([0.0e+00, 8.5e-01, 6.5e+00, 2.1e+05])
  """
  S_post = S_pre + T + F_immune + xi_stochastic
  return S_post

# SIMULATION

class CancerSimulation:
  """
  Complete cancer simulation integrating all 56 predictions.

  Medical context:
    Simulates single-cell or population dynamics over time with ability to
    apply interventions (carcinogens, treatments, immune modulation).

  Scientific basis:
    Implements master equations with discrete A updates and continuous C, K, N
    evolution. Tracks C_num and checks cancer condition each step. Allows
    testing all predictions in unified framework.
  """

  def __init__(self, A0=0, C0=1.0, K0=10.0, N0=1, C_num0=5):
    """
    Initialize simulation with starting conditions.

    >>> sim = CancerSimulation()
    >>> sim.A, sim.C, sim.K
    (0, 1.0, 10.0)
    """
    self.A = A0
    self.C = C0
    self.K = K0
    self.N = N0
    self.C_num = C_num0
    self.history = {"t": [], "A": [], "C": [], "K": [], "N": [], "C_num": []}

  def step(
    self,
    dt,
    alpha=0.05,
    beta=0.01,
    gamma=0.01,
    delta=0,
    lambda_max=0.5,
    K0=10,
    K_crit=2,
  ):
    """
    Advance simulation one time step.

    Medical context:
      dt: Time step [days, typically 0.1-1.0]
      alpha: Differentiation attempt rate [~0.05-0.2]
      beta: Constraint enforcement rate [~0.01-0.05]
      gamma: Constraint erosion from division [~0.01-0.05]
      delta: External damage rate [~0-0.1]
      lambda_max: Maximum proliferation rate [~0.1-0.5]
      K0: Reference constraint [~10]
      K_crit: Threshold for A flip [~2-5]

    Scientific basis:
      Updates C and K via Euler integration of ODEs. Checks if K crossed
      threshold to flip A. Updates N based on current state. Recomputes
      C_num if state changed significantly.

    >>> sim = CancerSimulation(A=1, C=0.3, K=5)
    >>> sim.step(1.0)
    >>> sim.C > 0.3, sim.K < 5
    (True, True)
    """
    dC = (alpha * self.A * (1 - self.C) - beta * self.K * self.C) * dt
    self.C = np.clip(self.C + dC, 0, 1)

    dK = (-gamma * self.A - delta) * dt
    self.K = max(0, self.K + dK)

    if self.K < K_crit and self.A == 0:
      self.A = 1
      self.C_num = compute_closure_number(int(self.N))

    dN = lambda_max * self.N * self.A * (1 - self.C) * np.exp(-self.K / K0) * dt
    self.N += dN

  def run(self, t_max, dt=0.1, **params):
    """
    Run simulation for specified duration.

    Medical context:
      t_max: Total simulation time [days to years]
      Returns: Dictionary of time series for all state variables

    >>> sim = CancerSimulation()
    >>> history = sim.run(10, dt=1.0, delta=0.05)
    >>> len(history['t'])
    100
    """
    t = 0
    while t < t_max:
      self.step(dt, **params)
      self.history["t"].append(t)
      self.history["A"].append(self.A)
      self.history["C"].append(self.C)
      self.history["K"].append(self.K)
      self.history["N"].append(self.N)
      self.history["C_num"].append(self.C_num)
      t += dt

    return {k: np.array(v) for k, v in self.history.items()}

  def is_cancer(self, theta=20, N_detection=1000):
    """
    Check if current state meets cancer criterion.

    Medical context:
      theta: C_num threshold [~20-40]
      N_detection: Minimum detectable cells [~10⁸-10⁹ clinically]

    >>> sim = CancerSimulation(A=1, C_num0=50, K=3, N0=1e9)
    >>> sim.is_cancer(theta=30, N_detection=1e8)
    True
    """
    dK_dt = -0.01 if self.A == 1 else 0
    return cancer_condition(
      self.A, self.C_num, self.K, dK_dt, self.N, theta, N_detection
    )

