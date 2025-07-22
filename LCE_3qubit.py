import numpy as np, itertools as it
from functools import reduce

# ==========  helper matrices  ======================================
I = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], complex)
Y = np.array([[0,-1j],[1j,0]], complex)
Z = np.diag([1,-1])

H = np.array([[1,1],[1,-1]], complex) / np.sqrt(2)
S = np.diag([1,1j])
one_q_Cliffords = [P @ R
                   for P in (I, X, Y, Z)
                   for R in (I, S, H, H @ S)]

def kron(*ops):
    return reduce(np.kron, ops)

def make_G_tet():
    pair_ZZ = [kron(Z, Z, I), kron(I, Z, Z)]
    XXX = kron(X, X, X)
    gens = pair_ZZ + [XXX]
    G = []
    for bits in it.product((0,1), repeat=3):
        g = np.eye(8, dtype=complex)
        for bit, Gen in zip(bits, gens):
            if bit:
                g = Gen @ g
        G.append(g)
    return G

## Entanglement measures and LU invariants for 3-qubit pure states
def three_tangle(psi):
    """
    Coffma-Kundu-Wootters three-tangle for a length-8 complex vector.
    """
    idx = lambda a,b,c: (a<<2) + (b<<1) + c
    a = psi
    a000, a001, a010, a011 = a[idx(0,0,0)], a[idx(0,0,1)], a[idx(0,1,0)], a[idx(0,1,1)]
    a100, a101, a110, a111 = a[idx(1,0,0)], a[idx(1,0,1)], a[idx(1,1,0)], a[idx(1,1,1)]

    d1 = (a000**2 * a111**2 + a001**2 * a110**2 +
          a010**2 * a101**2 + a100**2 * a011**2)
    d2 = (a000*a111*(a011*a100 + a101*a010 + a110*a001) +
          a011*a100*a101*a010 + a011*a100*a110*a001 +
          a101*a010*a110*a001)
    d3 = a000*a110*a101*a011 + a111*a001*a010*a100

    return 4*abs(d1 - 2*d2 + 4*d3)

def pairwise_concurrence(psi, pair=(0,1)):
    """
    Compute the concurrence for the reduced 2-qubit density matrix
    for the specified qubit pair in a 3-qubit pure state.
    """
    psi = psi.flatten()
    # Indices for the pair and the traced qubit
    all_indices = [0,1,2]
    i,j = pair
    k = list(set(all_indices) - set(pair))[0]  # the qubit to trace out

    # Build the reduced 2-qubit density matrix
    rho = np.outer(psi, psi.conj()).reshape((2,2,2,2,2,2))
    # Axes: i, j, k, i', j', k'
    # We trace over k and k'
    rho_2q = np.zeros((2,2,2,2), dtype=complex)
    for a in (0,1):
        for b in (0,1):
            for ap in (0,1):
                for bp in (0,1):
                    s = 0.0
                    for c in (0,1):
                        idx = [0,0,0]
                        idxp = [0,0,0]
                        idx[i] = a
                        idx[j] = b
                        idx[k] = c
                        idxp[i] = ap
                        idxp[j] = bp
                        idxp[k] = c
                        s += rho[idx[0], idx[1], idx[2], idxp[0], idxp[1], idxp[2]]
                    rho_2q[a,b,ap,bp] = s
    rho_2q = rho_2q.reshape((4,4))

    # Wootters spin-flip
    Y = np.array([[0,-1j],[1j,0]])
    YY = np.kron(Y,Y)
    R = rho_2q @ (YY @ rho_2q.conj() @ YY)
    evals = np.sort(np.real(np.sqrt(np.maximum(np.linalg.eigvals(R), 0))))[::-1]
    C = max(0, evals[0] - evals[1] - evals[2] - evals[3])
    return C

# not a proper LU invariant, but useful
def hyperdeterminant(psi):
    """Return Δ = d1 - 2 d2 + 4 d3 (complex)."""
    idx = lambda a,b,c: (a<<2) + (b<<1) + c
    a = psi
    a000, a001, a010, a011 = a[idx(0,0,0)], a[idx(0,0,1)], a[idx(0,1,0)], a[idx(0,1,1)]
    a100, a101, a110, a111 = a[idx(1,0,0)], a[idx(1,0,1)], a[idx(1,1,0)], a[idx(1,1,1)]
    d1 = (a000**2 * a111**2 + a001**2 * a110**2 +
          a010**2 * a101**2 + a100**2 * a011**2)
    d2 = (a000*a111*(a011*a100 + a101*a010 + a110*a001) +
          a011*a100*a101*a010 + a011*a100*a110*a001 +
          a101*a010*a110*a001)
    d3 = a000*a110*a101*a011 + a111*a001*a010*a100
    return d1 - 2*d2 + 4*d3


def kron_n(ops):
    """n-fold Kronecker: reduce(np.kron, ops, 1×1 identity)."""
    return reduce(np.kron, ops, np.array([1], complex))

def bloch(v, sys, n):
    """⟨X⟩,⟨Y⟩,⟨Z⟩ on qubit `sys` of an n-qubit state `v`."""
    out = []
    for P in (X, Y, Z):
        ops = [I]*n
        ops[sys] = P
        O = kron_n(ops)
        out.append(np.vdot(v, O @ v).real)
    return np.array(out)


G_tet = make_G_tet()

fid_data = np.load('fiducials_120.npy', allow_pickle=True)
fids = [np.round(t[0], 6) for t in fid_data[:40]] # the rest are trivial duplicates
fid_poly_str = [t[3] for t in fid_data[:40]]



# --- quick local-Clifford equivalence test for single kets -----------

# local Clifford group on three qubits (24^3 elements)
LC3 = [kron(a, kron(b, c))
       for a in one_q_Cliffords
       for b in one_q_Cliffords
       for c in one_q_Cliffords]
def _permute_index(idx, perm):
    """
    Convert |abc⟩ basis index to bit list, apply qubit permutation `perm`,
    and return new integer index.
    `perm` is a tuple/list of length‑3 giving where qubit 0,1,2
    are sent to (elements are 0,1,2).
    """
    a = (idx >> 2) & 1
    b = (idx >> 1) & 1
    c = idx & 1
    bits = [a, b, c]
    new_bits = [bits[perm[i]] for i in range(3)]
    return (new_bits[0] << 2) + (new_bits[1] << 1) + new_bits[2]

def permute_state(psi, perm):
    """Return the state with qubits permuted according to `perm`."""
    out = np.zeros_like(psi)
    for idx in range(8):
        out[_permute_index(idx, perm)] = psi[idx]
    return out

import itertools
def equiv_vector_LC3(psi1, psi2, tol=1e-8):
    """
    Return True if psi2 can be mapped to psi1 by some
    (UA⊗UB⊗UC) drawn from pre-computed LC3 and possibly a qubit permutation.
    """
    for perm in itertools.permutations(range(3)):   # 6 permutations
        psi2p = permute_state(psi2, perm)
        for U in LC3:                               # 24^3 ≈ 13k
            #v = U @ psi2p # permutations are not needed!
            v = U @ psi2 
            phase = np.vdot(psi1, v)
            if np.abs(phase) < tol:
                continue
            v *= phase.conj() / np.abs(phase)       # match global phase
            if np.allclose(psi1, v, atol=tol):
                return True
    return False


# ==========  classify fiducials by three-tangle and LU equivalence  ==========
def classify_fiducials_by_LU(fids):
    """
    Group fiducials by three-tangle, then within each tangle group
    identify LU-inequivalent representatives using check_fiducial_LU_equivalence.
    Prints progress for each tangle group.
    """
    # Step 1: compute three-tangles and bucket by (rounded) value
    tangle_buckets = {}
    for idx, psi in enumerate(fids):
        tau = three_tangle(psi)
        tau_key = round(tau, 8)   # group by 8-decimal precision
        tangle_buckets.setdefault(tau_key, []).append(idx)

    # Step 2: within each bucket, identify LU-inequivalent classes
    lu_classes = {}
    for tau_key, indices in tangle_buckets.items(): #Delta_buckets.items():
        reps = []  # representative indices for this tangle
        print(f"Processing tangle = {tau_key} with {len(indices)} fiducials...")
        for i, idx in enumerate(indices):
            psi_i = fids[idx]
            new_class = True
            for r in reps:
                psi_r = fids[r]
                if equiv_vector_LC3(psi_r,psi_i):# or equiv_vector_LC3(psi_r,psi_i.conj()):#check_fiducial_LU_equivalence(psi_r, psi_i):
                    new_class = False
                    break
            if new_class:
                reps.append(idx)
            print(f"  [{i+1}/{len(indices)}] Fiducial {idx}: {'new class' if new_class else 'equivalent to existing'}")
        lu_classes[tau_key] = reps

    # Step 3: print results
    print("LU classification of fiducials by three-tangle:")
    for tau_key, reps in sorted(lu_classes.items()):
        print(f" Tangle = {tau_key}:")
        for idx in reps:
            print(f"   Representative index = {idx}, poly = {fid_poly_str[idx]}")
    return lu_classes




# ==========  symbolic computation of three-tangle and concurrence  ==========
# This section computes the three-tangle and concurrence for fiducials
# using symbolic expressions, which can be useful for exact analysis.

import sympy as sp
import pandas as pd

def symbolic_three_tangle_and_concurrence(fids, indices=None, pair=(0,1)):
    """
    For given fiducials and indices, compute and print the exact symbolic
    three-tangle and concurrence for the specified qubit pair.
    Modified to return results rather than just print.
    """
    # Symbolic variables for the 3-qubit amplitudes
    a_syms = sp.symbols('a000 a001 a010 a011 a100 a101 a110 a111')
    psi_symbols = list(a_syms)

    # Sub-expressions for three-tangle
    d1 = (a_syms[0]**2 * a_syms[7]**2 +
        a_syms[1]**2 * a_syms[6]**2 +
        a_syms[2]**2 * a_syms[5]**2 +
        a_syms[4]**2 * a_syms[3]**2)
    d2 = (a_syms[0]*a_syms[7]*(a_syms[3]*a_syms[4] + a_syms[5]*a_syms[2] + a_syms[6]*a_syms[1]) +
        a_syms[3]*a_syms[4]*a_syms[5]*a_syms[2] +
        a_syms[3]*a_syms[4]*a_syms[6]*a_syms[1] +
        a_syms[5]*a_syms[2]*a_syms[6]*a_syms[1])
    d3 = a_syms[0]*a_syms[6]*a_syms[5]*a_syms[3] + a_syms[7]*a_syms[1]*a_syms[2]*a_syms[4]
    tau_symbolic = sp.simplify(4 * sp.Abs(d1 - 2*d2 + 4*d3))

    # Y⊗Y spin-flip matrix
    I = sp.I
    Y = sp.Matrix([[0, -I], [I, 0]])
    YY = sp.kronecker_product(Y, Y)

    def reduced_rho_pair(psi_list, pair=(0,1)):
        """Return the 4×4 reduced density matrix for qubit pair `pair`."""
        i, j = pair
        k = [q for q in [0,1,2] if q not in pair][0]
        rho2q = sp.zeros(4, 4)
        def idx3(a, b, c): return (a << 2) + (b << 1) + c
        for a in (0,1):
            for b in (0,1):
                row = 2*a + b
                for ap in (0,1):
                    for bp in (0,1):
                        col = 2*ap + bp
                        s = 0
                        for c in (0,1):
                            idx1 = [None]*3
                            idx1[i], idx1[j], idx1[k] = a, b, c
                            idx2 = [None]*3
                            idx2[i], idx2[j], idx2[k] = ap, bp, c
                            s += psi_list[idx3(*idx1)] * sp.conjugate(psi_list[idx3(*idx2)])
                        rho2q[row, col] = sp.simplify(s)
        return rho2q

    if indices is None:
        indices = [0, 9, 12, 1]

    results = {}
    for idx in indices:
        psi_num = fids[idx]
        psi_rational = [sp.nsimplify(val) for val in psi_num]
        subs_dict = dict(zip(a_syms, psi_rational))

        # Three-tangle
        tau_exact = sp.simplify(tau_symbolic.subs(subs_dict))

        # Concurrence for specified pair
        psi_sub = [sp.simplify(sym.subs(subs_dict)) for sym in psi_symbols]
        rho2q = reduced_rho_pair(psi_sub, pair=pair)
        tilde_rho = YY * rho2q.conjugate() * YY
        R = sp.simplify(rho2q * tilde_rho)
        eigs = []
        for eigval, mult, _ in R.eigenvects():
            eigs.extend([eigval]*mult)
        roots = [sp.sqrt(sp.simplify(ev)) for ev in eigs]
        roots_sorted = sorted(roots, key=lambda r: sp.N(r), reverse=True)
        # Pad with zeros if needed
        while len(roots_sorted) < 4:
            roots_sorted.append(0)
        C_sym = sp.simplify(sp.Max(0, (roots_sorted[0] - roots_sorted[1] - roots_sorted[2] - roots_sorted[3])**2))

        results[idx] = {
            "tau_symbolic": tau_exact,
            "concurrence_symbolic": sp.simplify(C_sym)
        }
    return results


def symmetry_group(psi, tol=1e-8):
    """
    Stabiliser subgroup of S3 that leaves |psi⟩ invariant
    up to an overall phase.
    Returns the list of permutations (each as a 3‑tuple).
    """
    syms = []
    for perm in itertools.permutations(range(3)):          # all 6 elements of S3
        psi_p = permute_state(psi, perm)
        phase = np.vdot(psi, psi_p)
        if np.abs(phase) < tol:
            continue
        phase /= np.abs(phase)                              # normalise to unit modulus
        if np.allclose(psi_p, phase * psi, atol=tol):
            syms.append(perm)
    return syms

def fiducial_properties(fids):
    # Now, for each fiducial psi in 'fids', print (τ, C_AB, Delta, S3 symmetry group):
    for idx, psi in enumerate(fids):
        tau  = three_tangle(psi)
        c_ab = pairwise_concurrence(psi, (0,1))
        c_ac = pairwise_concurrence(psi, (0,2))
        c_bc = pairwise_concurrence(psi, (1,2))
        Delta = hyperdeterminant(psi)
        sym_group = symmetry_group(psi)
        print(f"Fiducial {idx}: τ = {tau:.8f}, C_AB = {c_ab:.6f}, C_BC = {c_bc:.6f}, C_AC = {c_ac:.6f}, Δ = {Delta:.8f}, S3 symmetries: {sym_group}")

def possible_s3_sizes_under_LC(psi):
    """
    For the 3‑qubit fiducial |psi⟩, try all local Cliffords U ∈ LC3,
    compute |S₃| = len(symmetry_group(U @ psi)), and return the sorted list of unique sizes.
    """
    sizes = set()
    for U in LC3:
        sz = len(symmetry_group(U @ psi))
        sizes.add(sz)
    return sorted(sizes)

def bloch(v, sys, n):
    """⟨X⟩,⟨Y⟩,⟨Z⟩ on qubit `sys` of an n-qubit state `v`."""
    out = []
    for P in (X, Y, Z):
        ops = [I]*n
        ops[sys] = P
        O = kron_n(ops)
        out.append(np.vdot(v, O @ v).real)
    return np.array(out)
        
##########################
### Main execution########
##########################

#fiducial_properties(fids)

# print("Classifying fiducials by three-tangle and LU equivalence...")
lu_classes = classify_fiducials_by_LU(fids)

# # Prepare table data
table_rows = []

for tau_key, reps in sorted(lu_classes.items()):
    # Find all fiducials in this tangle class
    tangle_indices = []
    for idx, psi in enumerate(fids):
        tau = three_tangle(psi)
        if round(tau, 8) == tau_key:
            tangle_indices.append(idx)
    # For each LU class representative, collect all equivalent fiducials in this tangle class
    for rep_idx in reps:
        class_members = []
        psi_rep = fids[rep_idx]
        for idx in tangle_indices:
            psi = fids[idx]
            if equiv_vector_LC3(psi_rep, psi):
                class_members.append(idx)
        # Compute symbolic three-tangle and concurrence for representative
        sym_results = symbolic_three_tangle_and_concurrence(fids, indices=[rep_idx])
        tau_sym = sym_results[rep_idx]["tau_symbolic"]
        c_ab_sym = sym_results[rep_idx]["concurrence_symbolic"]

        # Compute numeric concurrence for rep
        c_ab_num = pairwise_concurrence(psi_rep, (0,1))

        # Compute unique sizes of S3 stabilizer groups for class members
        s3_sizes = sorted(set(len(symmetry_group(fids[idx])) for idx in class_members))

        # Prepare row with reduced columns
        table_rows.append({
            "Tangle_symbolic": tau_sym,
            "Concurrence_AB_symbolic": c_ab_sym,
            "LU Rep. Index": rep_idx,
            "Fiducial": fids[rep_idx],
            "Phase Polynomial": fid_poly_str[rep_idx],
            "S3_sizes": s3_sizes,
            "Class Members": class_members
        })

# Create DataFrame and print as table
df = pd.DataFrame(table_rows, columns=[
    "Tangle_symbolic", "Concurrence_AB_symbolic",
    "LU Rep. Index","Fiducial", "Phase Polynomial", "S3_sizes", "Class Members",
])
print("\nSummary Table of LU Classes:")
print(df.to_string(index=False))
# Save the summary table to a text file
with open("LU_class_summary.txt", "w") as f:
    f.write("Summary Table of LU Classes:\n")
    f.write(df.to_string(index=False))
    f.write("\n")

print("\nDone.")







# idx = [1,7,9,10,12,22,0,2]
# for i in idx:
#     sizes = possible_s3_sizes_under_LC(fids[i])
#     print(f"Fiducial {i} can achieve S₃ symmetry sizes: {sizes} under local Cliffords.")
#     # Bloch vectors for each qubit
#     bloch_vectors = [bloch(fids[i], sys, 3) for sys in range(3)]
#     print(f"Bloch vectors for fiducial {i}: {bloch_vectors}")
    


