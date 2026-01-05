# Structural Parity Model of Cancer

A symbolic framework for explaining cancer behavior using structural primitives and parity logic.

By [Annie Tran]

## About

This project introduces a symbolic, structural model of cancer that explains cancer behavior using a small set of system-level primitives rather than molecular detail.

The model treats cancer as a persistent failure of growth to resolve into stable structure, formalized using parity logic and three core variables: Asymmetry (A), Closure (C), and Constraint (K).

Cancer is modeled as a generative state that repeatedly attempts but fails to achieve structural closure, resulting in persistent growth.

    * Growth attempts resolution (closure),
    * But each attempt reproduces asymmetry,
    * Leading to expansion without completion.

The key primitives are **closure** (degree of tissue organization), **asymmetry** (parity of state: symmetric vs. asymmetric), and **constraint** (maintenance of cell identity and boundaries), which together define whether a tissue is stable (benign) or unstable (malignant).

## Overview

Cancer is a structural phenomenon driven by generative, open, and unresolved cycles. It results from failure to achieve closure, leading to persistent, fractal growth and instability. All core behaviors—heterogeneity, resistance, relapse, metastasis—stem from fundamental properties of generativity, openness, and parity asymmetry. Stability (cure) is only achieved when all generative cycles close.

Instead of asking which genes are broken, ask:

    * Is growth asymmetric or symmetric?
    * Is the system capable of reaching closure?
    * Are constraints sufficient to preserve identity once closure is reached?

The insight came from reading about tunicates and then asking ChatGPT about an unusually odd-numbered chemical bond (C39H43N3O11S) known for being "anti-cancer".

## Core Idea

Cancer is modeled as a generative state that repeatedly attempts but fails to achieve structural closure, resulting in persistent growth.

Cancer growth resembles a recursive generative process:

    * Growth attempts resolution (closure),
    * But each attempt reproduces asymmetry,
    * Leading to expansion without completion.

The operation `(2 * self) + self (3x)` captures how asymmetric growth can amplify itself while repeatedly failing to resolve, and the closure number measures how structurally difficult it is for a system to reach stability.

### Parity

Cancer's behavior maps structurally well onto even/odd parity logic, though it is a conceptual rather than a mechanistic or chemical framework. This logic can capture and unify several persistent features of cancer and offers a structural tool for explanation that is both simple and extensible.

Closure resistance can be expressed as the number of recursive steps required to reach even parity (if convergence is possible at all).

    EVENT          BITWISE     RESULT     INTERPRETATION
    =====          =======     ======     ==============
    self           1           1 (odd)    cancer potential
    self+self      1+1=10      2 (even)   regulated
    even+self      10+1=11     3 (odd)    unclosed/generative
    odd+odd        1+1=10      2 (even)   can close (in principle)
    even+even      10+10=100   4 (even)   remains regulated
    2(self)+self   10+1=11     3 (odd)    cancer logic

### Primitives

    1. Odd parity (O) = generative/unstable
       Even parity (E) = stable/closed

    2. Operation:  x -> 3x = (2*self) + self
         - Maintains odd parity
         - Attempts closure through recursion

    3. Closure Number (C_num):
       Minimum steps k where parity(3^k * x) = even
         - Low C_num: quick resolution possible
         - High C_num: structural resistance to closure

    4. Three parameters:
         A = Asymmetry (0 or 1)
         C = Closure completion (0 to 1)
         K = Constraint strength (positive number)

    5. Cancer condition:
         A = 1 AND C_num > threshold AND K degrading

#### Generativity / Asymmetry (A)
Asymmetry is the breaking of equivalence within a system. In cancer, asymmetry represents the initiation of abnormal growth behavior: uneven cell division, irregular tissue architecture, loss of balanced proliferation signals.

    * A=1: the system has ongoing generative potential
    * A=0: the system tends toward self-symmetry and convergence

    Examples:
    * Fractal dimension > 0.075 ⇒ A(S) = 1
    * Texture mean > threshold ⇒ A(S) = 1

#### Closure (C)
Closure is the capacity of a system to resolve its internal processes and return to a stable configuration. A closed system terminates growth and reaches equilibrium. A system that lacks closure assumulates unfinished processes and continues expanding or branching. 

    C(S) ∈ [0, 1] where:
    * C ≈ 1 → high closure: the system stabilizes quickly
    * C ≈ 0 → low closure: the system stays open indefinitely

    HIGH C: system is structurally stable and terminates growth.
    LOW C: system resists collapse and persists in open generative loops.

#### Constraint (K)
Constraint measures structural containment: the system's ability to resist deformation, maintain boundaries, enforce symmetry, and absorb pressure

    K(S) ∈ [0, 1]
    * K ≈ 1 → strong constraint: system resists divergence
    * K ≈ 0 → weak constraint: system is deformable, disordered

    STRONG K: system has high internal regulation and structural order
    WEAK K: system lacks order 


### Interpretation

Label | A | C | K | C_num | Example | Note
--- | --- | --- | --- | --- | --- | ---
**Normal** | 0 | 1.0 | High | 0 | Quiescent cell | Stable, healthy tissue
**Benign Tumor** | 1 | 0.7 | High | 2 | Adenoma | Low risk, slow growth
**Indolent Cancer** | 1 | 0.7 | Med | 4 | Low-grade carcinoma | Early stage, monitor closely
**Aggressive Cancer** | 1 | 0.3 | Low | 10 | High-grade carcinoma | High risk, rapid spread
**Dormant Cancer** | 0 | 0.9 | Low | 3 | Residual cell | Risk of recurrence
**Metastatic Cancer** | 1 | 0.2 | Very Low | 15 | Metastatic lesion | Urgent intervention required
**Remission** | 0 | 0.95 | Med | 1 | Post-treatment | Monitor for recurrence

## Usage

The framework is instantiated as a deterministic, rule-based model applied to the Breast Cancer Wisconsin dataset.

Characteristics:
* No machine learning
* No parameter training
* Explicit feature-to-primitive mappings
* Threshold-based logic
* Fully interpretable rules

The implementation maps morphological features to:
* Asymmetry (binary trigger)
* Closure (geometric ratio)
* Constraint (structural regularity composite)

Classification is performed by combining these structural states rather than optimizing a loss function.

## License

&copy; 2025-2026 [Annie Tran][Author]

This repository is licensed under the MIT license. See [LICENSE] for
details.

[Author]: nntrn.github.io
[Annie Tran]: github.com/nntrn/cancer
[license]: ./LICENSE