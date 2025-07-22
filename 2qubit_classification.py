"""
Self-contained enumeration of 2-qubit tetrahedral bases
with  Clifford-hierarchy tests.

Hierarchy tests use `is_pauli` (equality to a Pauli up to a global phase)
Checks are reported for levels 3, 4, 5 for both the basis matrix M
and the diagonal phase gate Df.

User input:
level = 3    # e.g. level-3 of the Clifford hierarchy

Output:
- Patterns of tetrahedral bases with autocorrelation
- Phase polynomial coefficients (a1, a2, a12)
- Hierarchy tests for M and Df
"""

level = 3 # Level of the Clifford hierarchy to enumerate 

import numpy as np
import itertools
import sys


# ---------------- basic gates ----------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array(
    [[1, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0],
     [0, 1, 0, 0]],
    dtype=complex
)

########################
## Clifford hierarchy ##
########################


# ---------------- Pauli group ----------------
#   16 two-qubit Paulis: I⊗I , … , Z⊗Z
paulis = [
    np.kron(p1, p2)
    for p1 in (I, X, Y, Z)
    for p2 in (I, X, Y, Z)
]

# ---------- helper functions ----------
def is_pauli(U: np.ndarray, tol: float = 1e-6) -> bool:
    """Return True iff U is proportional to a two-qubit Pauli."""
    for P in paulis:
        # find any non-zero entry to estimate the phase
        idx = np.flatnonzero(np.abs(P) > tol)[0]
        r, c = divmod(idx, 4)
        phase = U[r, c] / P[r, c]
        if np.allclose(U, phase * P, atol=tol):
            return True
    return False

def level2_via_M(M: np.ndarray) -> bool:
    """
    Level-2 (Clifford): for every Pauli P,
      M P M†  must itself be proportional to a Pauli.
    """
    for P in paulis:
        V = M @ P @ M.conj().T
        if not is_pauli(V):
            return False
    return True

def level3_via_M(M: np.ndarray) -> bool:
    """
    Test:  for every Pauli P and Q,
           N = (M P M†) Q (M P M†)†  must itself be Pauli.
    """
    for P in paulis:
        V = M @ P @ M.conj().T
        for Q in paulis:
            N = V @ Q @ V.conj().T
            if not is_pauli(N):
                return False
    return True


def level4_via_M(M: np.ndarray) -> bool:
    """Level-4:  M P M† must satisfy level-3 for every Pauli P."""
    return all(level3_via_M(M @ P @ M.conj().T) for P in paulis)


def level5_via_M(M: np.ndarray) -> bool:
    """Level-5:  M P M† must satisfy level-4 for every Pauli P."""
    return all(level4_via_M(M @ P @ M.conj().T) for P in paulis)

##########################################
#### Bloch vector and autocorrelation ####
#########################################

def bloch(v: np.ndarray, sys: int) -> np.ndarray:
    """Single-qubit Bloch vector on subsystem 0 or 1."""
    return np.array([
        np.vdot(v, (np.kron(P, I) if sys == 0 else np.kron(I, P)) @ v).real
        for P in (X, Y, Z)
    ])


def autocorr(vecs):
    prods = [np.dot(u, w) for u in vecs for w in vecs]
    return tuple(sorted(np.round(prods, 6))) ### Sort and round to 6 decimal places

##############################
##### Gottesman criterion ####
###############################

def weighted_degree_2q(a1, a2, a12, m):
    """
    Implements Gottesman's criterion:
    k = max over nonzero coefficients of [(m - v2(a_i) - 1) + wt(a_i)],
    where v2(a) is the 2-adic valuation (exponent of 2 dividing a),
    wt(a1)=wt(a2)=1, wt(a12)=2.
    """
    coeffs = [a1, a2, a12]
    weights = [1, 1, 2]
    degs = []
    for a, w in zip(coeffs, weights):
        if a != 0:
            degs.append((m - v2(a) - 1) + w)
    return max(degs) if degs else 0

def v2(a: int) -> int:
    """2-adic valuation: exponent of 2 dividing a (a > 0)."""
    if a == 0:
        return 0
    return (a & -a).bit_length() - 1



## Tetrahedral basis construction ##
plus = np.ones(4, dtype=complex) / 2
G_EJM = [np.kron(P, P) for P in (I, X, Y, Z)]

# Initialize a dictionary to store patterns
patterns = {}

###### Main enumeration loop ######

for m in range(1, level + 1):
    mod = 2 ** m
    denom = mod
    for a1, a2, a12 in itertools.product(range(mod), repeat=3):
        if weighted_degree_2q(a1, a2, a12, m) != level:
            continue

        # diagonal polynomial phase function
        phases = [
            np.exp(1j * 2 * np.pi * ((a1 * x1 + a2 * x2 + a12 * x1 * x2) % mod) / denom)
            for x1, x2 in itertools.product((0, 1), repeat=2)
        ]
        Df = np.diag(phases)

        # prepare tetrahedral basis
        psi = CNOT @ np.kron(I, H) @ Df @ plus
        basis = [g @ psi for g in G_EJM]

        # classify via autocorrelation of Bloch vectors on qubit A
        auto0= autocorr([bloch(b, 0) for b in basis])
        auto1= autocorr([bloch(b, 1) for b in basis])
        sorted_key = tuple(sorted((auto0, auto1)))
        patterns.setdefault(sorted_key, []).append((m, a1, a2, a12, basis, Df))

        # # Print when the specific example appears
        # if (a1, a2, a12, m) == (1, 0, 2, 4):
        #     print(f"Found example: a1={a1}, a2={a2}, a12={a12}, m={m}")
        #     for j, b in enumerate(basis, 1):
        #         print(f"    v{j} =", np.round(bloch(b, 0), 6))
        #     for j, b in enumerate(basis, 1):
        #         print(f"    v{j} =", np.round(bloch(b, 1), 6))
        #     sys.exit(0)  # Exit after finding the specific example
        


# --------- report patterns and hierarchy tests ----------
for idx, (key, reps) in enumerate(patterns.items(), start=1):
    m, a1, a2, a12, basis, Df = reps[0]          # one representative
    mod = 2 ** (m)
    M = np.column_stack(basis)

    # hierarchy tests
    tests_M = {
        "C2": level2_via_M(M),
        "C3": level3_via_M(M),
        "C4": level4_via_M(M),
       # "C5": level5_via_M(M)
    }
    tests_Df = {
        "C2": level2_via_M(Df),
        "C3": level3_via_M(Df),
        "C4": level4_via_M(Df),
      # "C5": level5_via_M(Df)
    }

    print(f"\nPattern {idx}")
    print("  Autocorr 1:", autocorr([bloch(b, 0) for b in basis]))
    print("  Autocorr 2:", autocorr([bloch(b, 1) for b in basis]))
    print(f"  Phase poly: m={m}, mod={mod}, (a1,a2,a12)=({a1},{a2},{a12})")
    print("  M =\n", np.round(M, 6))
    for j, b in enumerate(basis, 1):
        print(f"    v{j} =", np.round(bloch(b, 0), 6))
    for j, b in enumerate(basis, 1):
        print(f"    v{j} =", np.round(bloch(b, 1), 6))
    print("  Tests for M :", tests_M)
    print("  Tests for Df:", tests_Df)
    # # Save the output to a text file
    output_file = f"2_qubit_level_{level}.txt"
    with open(output_file, "a") as f:
        f.write(f"\nPattern {idx}\n")
        f.write(f"  Autocorr 1: {autocorr([bloch(b, 0) for b in basis])}\n")
        f.write(f"  Autocorr 2: {autocorr([bloch(b, 1) for b in basis])}\n")
        f.write(f"  Phase poly: m={m}, mod={mod}, (a1,a2,a12)=({a1},{a2},{a12})\n")
        f.write(f"  M =\n{np.round(M, 6)}\n")
        for j, b in enumerate(basis, 1):
            f.write(f"    v{j} = {np.round(bloch(b, 0), 6)}\n")
        for j, b in enumerate(basis, 1):
            f.write(f"    v{j} = {np.round(bloch(b, 1), 6)}\n")
        f.write(f"  Tests for M : {tests_M}\n")
        f.write(f"  Tests for Df: {tests_Df}\n")