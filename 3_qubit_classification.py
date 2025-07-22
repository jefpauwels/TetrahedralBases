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

k = 3    # e.g. level-3 of the Clifford hierarchy

import numpy as np
import itertools

# ---------------- basic gates ----------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# two-qubit CNOT (control on MSB, target LSB)
CNOTLSB = np.array([
    [1,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,1,0,0]
], dtype=complex)



# lift to three qubits: CNOT₁→₂ and CNOT₂→₃
CNOT12 = np.kron(CNOTLSB, I)
CNOT23 = np.kron(I, CNOTLSB)

# ---------------- Pauli group on 3 qubits ----------------
paulis3 = [
    np.kron(np.kron(p1, p2), p3)
    for p1 in (I, X, Y, Z)
    for p2 in (I, X, Y, Z)
    for p3 in (I, X, Y, Z)
]

# ---------- Clifford-hierarchy tests ----------
def is_pauli(U: np.ndarray, tol: float = 1e-6) -> bool:
    """True iff U ≈ e^{iφ}·P for some 3-qubit Pauli P."""
    for P in paulis3:
        idx = np.flatnonzero(np.abs(P) > tol)[0]
        r, c = divmod(idx, 8)
        phase = U[r, c] / P[r, c]
        if np.allclose(U, phase * P, atol=tol):
            return True
    return False

def level2_via_M(M: np.ndarray) -> bool:
    """
    Level-2 test (Clifford):  for every 3-qubit Pauli P,
    M P M† must itself be a Pauli (up to global phase).
    """
    for P in paulis3:
        if not is_pauli(M @ P @ M.conj().T):
            return False
    return True

def level3_via_M(M: np.ndarray) -> bool:
    """Level-3 test: ∀ P,Q ∈ Pauli₃,  (M P M†) Q (M P M†)† must be Pauli."""
    for P in paulis3:
        V = M @ P @ M.conj().T
        for Q in paulis3:
            N = V @ Q @ V.conj().T
            if not is_pauli(N):
                return False
    return True

# (optional)
def level4_via_M(M): return all(level3_via_M(M @ P @ M.conj().T) for P in paulis3)
def level5_via_M(M): return all(level4_via_M(M @ P @ M.conj().T) for P in paulis3)

# ---------- Bloch vectors & autocorrelation ------------
def bloch(v: np.ndarray, sys: int) -> np.ndarray:
    """Single-qubit Bloch vector of state `v` on qubit `sys` (0,1,2)."""
    coords = []
    for P in (X, Y, Z):
        ops = [I, I, I]
        ops[sys] = P
        O = np.kron(np.kron(ops[0], ops[1]), ops[2])
        coords.append(np.vdot(v, O @ v).real)
    return np.array(coords)

def autocorr(vecs):
    """Return sorted tuple of all pairwise dot products among Bloch vectors."""
    prods = [np.dot(u, w) for u in vecs for w in vecs]
    return tuple(sorted(np.round(prods, 6)))

# ---------- build tetrahedral group G_tet on 3 qubits ----------
Z12 = np.kron(np.kron(Z, Z), I)
Z23 = np.kron(np.kron(I, Z), Z)
XXX = np.kron(np.kron(X, X), X)
G_tet_3 = []
for b1,b2,b3 in itertools.product([0,1], repeat=3):
    g = np.eye(8, dtype=complex)
    if b1: g = Z12 @ g
    if b2: g = Z23 @ g
    if b3: g = XXX @ g
    G_tet_3.append(g)


# ---------- enumeration of phase-polynomial diagonals ----------
plus = np.ones(8, dtype=complex)/np.sqrt(8)
patterns3 = {}

import time
import itertools
start = time.time()

import numpy as np
import itertools


# weights of the 7 monomials (x1,x2,x3, x1x2, x1x3, x2x3, x1x2x3)
WEIGHTS = np.array([1, 1, 1, 2, 2, 2, 3])

def v2(a: int) -> int:
    """2-adic valuation: exponent of 2 dividing a (a > 0)."""
    return (a & -a).bit_length() - 1

def weighted_degree(coeffs, m):
    """
    Compute max over nonzero coeffs of [(m - nu2(a_I) - 1) + weight_I].
    That is Gottesman’s level test.
    """
    degs = []
    for a, w in zip(coeffs, WEIGHTS):
        if a != 0:
            degs.append((m - v2(a) - 1) + w)
    return max(degs) if degs else 0

def tuples_of_weight(mod, m, k):
    """
    Yield all 7‐tuples in {0,…,mod-1}^7 whose weighted_degree(...) == k.
    We only allow nonzero entries on those indices I where
      (m - 1) + WEIGHTS[I] >= k,
    and at least one I exactly saturates (m - 1)+WEIGHTS[I] - nu2(a_I) == k.
    For simplicity we just brute‐force over the “allowed” support patterns.
    """
    # which positions can possibly contribute?
    possible = [i for i,w in enumerate(WEIGHTS) if (m - 1 + w) >= k]
    # now brute‐force all assignments on that smaller set,
    # zero elsewhere.
    for subset in itertools.chain.from_iterable(
            itertools.combinations(possible, r) for r in range(1, len(possible)+1)
    ):
        # for each choice of nonzero-support subset:
        for vals in itertools.product(range(1, mod), repeat=len(subset)):
            coeffs = [0]*7
            for idx, v in zip(subset, vals):
                coeffs[idx] = v
            if weighted_degree(coeffs, m) == k:
                yield tuple(coeffs)

# --- now the enumeration using tuples_of_weight ---
start = time.time()
patterns3 = {}
plus = np.ones(8, dtype=complex)/np.sqrt(8)

total = 0
for m in range(1, k+1):
    total += sum(1 for _ in tuples_of_weight(2**m, m, k))
print(f"Total diagonals at level {k}: {total:,}")

count = 0
for m in range(1, k+1):
    mod = 2**m
    denom = mod
    for coeffs in tuples_of_weight(mod, m, k):
        count += 1
        # build f, Df, psi, basis, etc exactly as before:
        def f(x1,x2,x3, c=coeffs):
            a1,a2,a3,a12,a13,a23,a123 = c
            return (a1*x1 + a2*x2 + a3*x3
                  + a12*x1*x2 + a13*x1*x3 + a23*x2*x3
                  + a123*x1*x2*x3) % mod

        phases = [
            np.exp(1j * 2*np.pi * f(x1,x2,x3) / denom)
            for x1,x2,x3 in itertools.product((0,1), repeat=3)
        ]
        Df   = np.diag(phases)
        psi  = CNOT12 @ CNOT23 @ np.kron(np.kron(I,I), H) @ Df @ plus
        basis = [g @ psi for g in G_tet_3]

        auto0 = autocorr([bloch(b,0) for b in basis])
        auto1 = autocorr([bloch(b,1) for b in basis])
        auto2 = autocorr([bloch(b,2) for b in basis])
        sorted_key = tuple(sorted((auto0, auto1, auto2)))
        patterns3.setdefault(sorted_key, []).append((m, coeffs, basis, Df))

        # progress every 1000
        if count % 1000 == 0:
            elapsed = time.time() - start
            eta = elapsed/count * (total - count)
            print(f"→ {count:,}/{total:,} done—elapsed {elapsed:.1f}s, ETA {eta/60:.1f} min")

print(f"\nFinished enumeration in {time.time()-start:.1f}s, found {len(patterns3)} patterns.")

# report how many patterns we found
num_patterns = len(patterns3)
print(f"\nFound {num_patterns} unique patterns. Now reporting:\n")


import numpy as np


# — now go on to heavy reporting … 

# ---------- reporting ----------
with open(f"3_qubit_level{k}.txt", "w") as report_file:
    for idx, (key, reps) in enumerate(patterns3.items(), start=1):
        auto0, auto1, auto2 = key
        m, coeffs, basis, Df = reps[0]
        mod = 2**(m)
        M   = np.column_stack(basis)

        report_file.write(f"\nPattern {idx}\n")
        report_file.write(f"  Autocorr q0: {auto0}\n")
        report_file.write(f"  Autocorr q1: {auto1}\n")
        report_file.write(f"  Autocorr q2: {auto2}\n")
        report_file.write(f"  Phase poly (m={m}, mod={mod}): coeffs={coeffs}\n")
        report_file.write(f"  M =\n{np.round(M, 6)}\n")
        for j, b in enumerate(basis, 1):
            v0 = np.round(bloch(b, 0), 6)
            v1 = np.round(bloch(b, 1), 6)
            v2 = np.round(bloch(b, 2), 6)
            report_file.write(f"    psi{j}:  v0={v0},  v1={v1},  v2={v2}\n")
        #report_file.write(f"  Level-2 test for M : {level2_via_M(M)}\n")
        #report_file.write(f"  Level-2 test for Df: {level2_via_M(Df)}\n")
        #report_file.write(f"  Level-3 test for M : {level3_via_M(M)}\n")
        #report_file.write(f"  Level-3 test for Df: {level3_via_M(Df)}\n")
        # Uncomment the following lines if Level-4 tests are needed
        # report_file.write(f"  Level-4 test for M : {level4_via_M(M)}\n")
        # report_file.write(f"  Level-4 test for Df: {level4_via_M(Df)}\n")