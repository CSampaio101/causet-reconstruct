"""
causet_reconstruct.py
=====================

Reconstruction of a finite causal set from nonlinear spectral data, following the thesis
"Hearing the Shape of Spacetime Using Nonlinearities in Causal Set Theory" (C. Sampaio, MSc thesis,
University of Waterloo, 2026). Section and table numbers below refer to the thesis; Appendix B of the
thesis describes this program.

The program has a FORWARD stage and a blind INVERSE stage.

FORWARD (Section 6.1; generates the input data from a causal set):
    causal set  ->  B (Benincasa-Dowker matrix)  ->  B_tilde = (B+B^T)/2 + i(B-B^T)/2
                ->  spectrum omega, unitary U (position basis -> eigenbasis of B_tilde)
                ->  vertex components in the eigenbasis of B_tilde, in any chosen representation (p,q):
                        cubic   W^{(p,q)} = g      sum_m nu_m |e_m>^{(x)p} <e_m|^{(x)q},  p+q = 3
                        quartic V^{(p,q)} = lambda sum_m nu_m |e_m>^{(x)p} <e_m|^{(x)q},  p+q = 4
    Here p is the number of ket legs and q the number of bra legs, so (p,q) is the q -> p
    representation of the thesis (--cubic 1,2 is the cubic 2 -> 1 representation).

INVERSE (Section 6.2; receives only omega, the vertex components, the set of BD coefficients and
the sign convention g, lambda > 0):
    1. Probe contraction of one vertex until two legs remain (Chapter 4, Table 4.1):
           free legs (1,1): normal matrix U C U^dagger          -> eigendecompose directly
           free legs (0,2): complex symmetric U^* C U^dagger    -> eigendecompose M^dagger M
           free legs (2,0): complex symmetric U C U^T           -> eigendecompose M M^dagger
       This recovers the columns of U up to a permutation and a phase per column.
    2. Phase fixing from the vertex data (Sections 5.6-5.9):
           self-contraction  s^{(p,q)}_m = e^{i(q-p) theta_m} kappa nu_m
           (checked against the eigenpair ratio for the representation used in step 1).
       Positivity reduces each column phase to the d-th roots of unity, d = |q-p|
       (U(1) for d = 0). With two vertices the residual set is Z_gcd(d1,d2).
    3. BD legality (Sections 5.4 and 5.10): a backtracking search over the residual set, pruning with
       the pairwise rule that exactly one of (B_jk, B_kj) is a legal coefficient and the other
       vanishes, followed by a global check that the decoded matrix is the BD matrix of the causal
       set defined by its own links. When the residual set is trivial this step only decodes.
    4. Decode with D_+ (and D_-), read off the links, and take the transitive closure.

Nothing from the forward stage other than omega and the vertex components is visible to the inverse
stage. The original B, B_tilde, U and causal set are used only afterwards, to compare the result
with the input.

Dependencies: numpy, networkx (matplotlib is needed only for figures).

Examples:
    python causet_reconstruct.py --sprinkle 20
    python causet_reconstruct.py --sprinkle 20 --scheme 1+1
    python causet_reconstruct.py --sprinkle 20 --cubic none --quartic 0,4 --enumerate 20
    python causet_reconstruct.py --links "a<b<e<f,a<c<e,a<d" --draw-only --figure bd_example.png
    python causet_reconstruct.py --blindspot --cubic none --quartic 0,4 --enumerate 20 \
        --only-blind-spots --figure blind_spot.pdf

The causal set of Chapter 7 of the thesis (LINKS is given in README.md and in Appendix B):
    python causet_reconstruct.py --links "$LINKS" --cubic none --quartic 0,4 --enumerate 20 \
        --match-labels --highlight --figure example23_blindspot.png
    python causet_reconstruct.py --links "$LINKS" --cubic 0,3 --quartic 0,4 \
        --figure example23_with_cubic.png

Elements are labelled by letters (a, b, c, ...) in the printed output and in figures; pass
--labels numbers for the integer labels 0, 1, 2, ...
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from math import gcd
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=160)

LETTERS = "abcdefghijklmnop"  # einsum subscripts (not element labels)


def element_names(n: int, style: str = "letters") -> List[str]:
    """Display names for causal-set elements: a, b, ..., z, aa, ab, ... (or 0, 1, 2, ... for 'numbers')."""
    if style == "numbers":
        return [str(i) for i in range(n)]
    names = []
    for i in range(n):
        s, k = "", i
        while k >= 0:
            s = chr(ord("a") + k % 26) + s
            k = k // 26 - 1
        names.append(s)
    return names

# =============================================================================
# 0. Benincasa-Dowker coefficient schemes (normalized)
# =============================================================================


@dataclass(frozen=True)
class BDScheme:
    """Normalized BD coefficients by interval cardinality n(x,y); cardinalities beyond the table give 0."""

    name: str
    diagonal: float
    coeff_by_cardinality: Dict[int, float]

    @property
    def legal_offdiag(self) -> np.ndarray:
        return np.array(list(self.coeff_by_cardinality.values()), dtype=float)

    @property
    def link_coefficient(self) -> float:
        return float(self.coeff_by_cardinality[0])

    def check_no_negation(self) -> bool:
        s = set(self.legal_offdiag.tolist())
        return not any((-c) in s for c in s)


BD_3P1 = BDScheme("3+1", -1.0, {0: 1.0, 1: -9.0, 2: 16.0, 3: -8.0})
BD_1P1 = BDScheme("1+1", -0.5, {0: 1.0, 1: -2.0, 2: 1.0})
SCHEMES = {"3+1": BD_3P1, "1+1": BD_1P1}

# =============================================================================
# 1. Causal sets
# =============================================================================


@dataclass
class CausalSet:
    """prec[j,k] is True iff k precedes j (k in the causal past of j)."""

    prec: np.ndarray

    @property
    def n(self) -> int:
        return self.prec.shape[0]

    def interval_cardinalities(self) -> np.ndarray:
        """n(j,k) = #{z : k < z < j} = (prec @ prec)[j,k]."""
        P = self.prec.astype(int)
        return P @ P

    def link_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n))
        for j in range(self.n):
            for k in range(self.n):
                if self.prec[j, k]:
                    G.add_edge(k, j)
        return nx.transitive_reduction(G)

    @staticmethod
    def from_links(G: nx.DiGraph, n: Optional[int] = None) -> "CausalSet":
        if n is None:
            n = G.number_of_nodes()
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("link graph must be a DAG")
        TC = nx.transitive_closure_dag(G)
        prec = np.zeros((n, n), dtype=bool)
        for k, j in TC.edges():
            prec[j, k] = True
        return CausalSet(prec)

    def is_isomorphic_to(self, other: "CausalSet") -> Tuple[bool, Optional[Dict[int, int]]]:
        gm = nx.algorithms.isomorphism.DiGraphMatcher(self.link_graph(), other.link_graph())
        if gm.is_isomorphic():
            return True, dict(gm.mapping)
        return False, None

    def reversed(self) -> "CausalSet":
        return CausalSet(self.prec.T.copy())


def sprinkle_minkowski(N: int, dim: int = 4, seed: int = 0, sort_by_time: bool = True) -> CausalSet:
    """Place N points uniformly at random in the unit box [0,1]^dim of Minkowski spacetime (a sprinkling with N fixed)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(N, dim))
    if sort_by_time:
        X = X[np.argsort(X[:, 0])]
    t = X[:, 0]
    x = X[:, 1:]
    prec = np.zeros((N, N), dtype=bool)
    for j in range(N):
        for k in range(N):
            if j == k:
                continue
            dt = t[j] - t[k]
            dx = np.linalg.norm(x[j] - x[k])
            if dt > dx:
                prec[j, k] = True
    return CausalSet(prec)


def causet_from_link_string(spec: str) -> Tuple[CausalSet, List[str]]:
    """
    Build a causal set from relations written as 'a<b<e,a<c<e,a<d' (x<y means x precedes y).
    Chains are allowed, the transitive closure is taken, and an unrelated element can be listed by
    itself. Elements are ordered alphabetically; returns (causal set, element names in that order).
    """
    pairs: List[Tuple[str, str]] = []
    names: set = set()
    for item in spec.split(","):
        chain = [x.strip() for x in item.split("<")]
        if any(not x for x in chain):
            if len(chain) == 1:
                continue
            raise ValueError(f"cannot parse relation '{item}'")
        names.update(chain)
        pairs.extend(zip(chain[:-1], chain[1:]))
    ordered = sorted(names)
    if not ordered:
        raise ValueError("no elements given")
    idx = {name: i for i, name in enumerate(ordered)}
    G = nx.DiGraph()
    G.add_nodes_from(range(len(ordered)))
    G.add_edges_from((idx[u], idx[v]) for u, v in pairs)
    return CausalSet.from_links(G, len(ordered)), ordered


def bd_matrix(causet: CausalSet, scheme: BDScheme = BD_3P1) -> np.ndarray:
    """Normalized BD matrix: B[j,k] = coeff(n(j,k)) if k < j, 0 otherwise; diagonal = scheme.diagonal."""
    n = causet.n
    card = causet.interval_cardinalities()
    B = np.zeros((n, n), dtype=float)
    np.fill_diagonal(B, scheme.diagonal)
    for j in range(n):
        for k in range(n):
            if j != k and causet.prec[j, k]:
                B[j, k] = scheme.coeff_by_cardinality.get(int(card[j, k]), 0.0)
    return B


def causet_from_bd(B: np.ndarray, scheme: BDScheme = BD_3P1, tol: float = 1e-6) -> CausalSet:
    """Links are the entries equal to the L_0 coefficient; the order is their transitive closure."""
    n = B.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for j in range(n):
        for k in range(n):
            if j != k and abs(B[j, k] - scheme.link_coefficient) < tol:
                G.add_edge(k, j)
    return CausalSet.from_links(G, n)


def is_self_consistent_bd(B: np.ndarray, scheme: BDScheme, tol: float = 1e-3) -> bool:
    """Blind global check: legal entries, acyclic links, and B equals the BD matrix of its own links."""
    n = B.shape[0]
    if not np.allclose(np.diag(B), scheme.diagonal, atol=tol):
        return False
    legal = scheme.legal_offdiag
    off = B[~np.eye(n, dtype=bool)]
    nonzero = off[np.abs(off) >= tol]
    if nonzero.size and np.any(np.min(np.abs(nonzero[:, None] - legal[None, :]), axis=1) >= tol):
        return False
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for j in range(n):
        for k in range(n):
            if j != k and abs(B[j, k] - scheme.link_coefficient) < tol:
                G.add_edge(k, j)
    if not nx.is_directed_acyclic_graph(G):
        return False
    B_re = bd_matrix(CausalSet.from_links(G, n), scheme)
    return bool(np.allclose(B, B_re, atol=tol))


# =============================================================================
# 2. Forward stage
# =============================================================================


def hermitize(B: np.ndarray) -> np.ndarray:
    B = np.asarray(B, dtype=float)
    return 0.5 * (B + B.T) + 0.5j * (B - B.T)


def canonical_eigenvectors(omega: np.ndarray, Q: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """
    Replace the eigenvectors returned by the eigensolver (columns of Q) by ones that depend only on the
    matrix, so that the same seed gives the same data on every machine.

    An eigensolver returns each eigenvector with an arbitrary phase, and an arbitrary basis of every
    degenerate eigenspace; which one depends on the linear-algebra library and on the CPU. The projector
    onto each eigenspace does not. From it we take, in turn, the projection of the first basis vector
    that has a substantial component in what is left of the eigenspace, normalise it, and remove it.
    """
    n = len(omega)
    out = np.zeros_like(Q)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(omega[j + 1] - omega[i]) < tol:
            j += 1
        P = Q[:, i : j + 1] @ Q[:, i : j + 1].conj().T
        for k in range(i, j + 1):
            d = np.real(np.diag(P))
            r = int(np.argmax(d > 0.3 * d.max()))
            v = P[:, r] / math.sqrt(d[r])
            out[:, k] = v
            P = P - np.outer(v, v.conj())
        i = j + 1
    return out


def diagonalize(B_tilde: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (omega, U) with B_tilde = U^dagger diag(omega) U and U[i,m] = <i|e_m>.
    If seed is given, the spectral basis is randomly relabeled (and rotated inside degenerate
    eigenspaces) so that the inverse stage cannot exploit any ordering.
    """
    omega, Q = np.linalg.eigh(B_tilde)
    Q = canonical_eigenvectors(omega, Q)
    U = Q.conj().T
    if seed is not None:
        rng = np.random.default_rng(seed)
        # random rotation within degenerate blocks
        i = 0
        while i < len(omega):
            j = i
            while j + 1 < len(omega) and abs(omega[j + 1] - omega[i]) < 1e-9:
                j += 1
            if j > i:
                k = j - i + 1
                Z = rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k))
                Y, _ = np.linalg.qr(Z)
                U[i : j + 1, :] = Y @ U[i : j + 1, :]
            i = j + 1
        perm = rng.permutation(len(omega))
        omega = omega[perm]
        U = U[perm, :]
        # random per-row phases (a different choice of eigenvector phases in the spectral basis)
        phases = np.exp(2j * np.pi * rng.uniform(size=len(omega)))
        U = np.diag(phases) @ U
    return omega.real, U


@dataclass
class VertexData:
    """A representation (p,q) of a local vertex, given as a rank-(p+q) tensor in the B_tilde eigenbasis.

    Index order: the first p indices are contravariant (ket legs, built from U), the last q are
    covariant (bra legs, built from U^*). kappa_known is optional and only used to report volumes.
    """

    name: str
    p: int
    q: int
    T: np.ndarray
    kappa_known: Optional[float] = None

    @property
    def order(self) -> int:
        return self.p + self.q

    @property
    def d(self) -> int:
        return abs(self.q - self.p)

    @property
    def label(self) -> str:
        return f"{self.name} {self.p}->{self.q}"


def representation_components(U: np.ndarray, p: int, q: int, kappa: float, volumes: np.ndarray) -> np.ndarray:
    """T[i_1..i_p, j_1..j_q] = kappa sum_m nu_m U[i_1,m]..U[i_p,m] U*[j_1,m]..U*[j_q,m]."""
    n = p + q
    ops, subs = [], []
    for t in range(p):
        ops.append(U)
        subs.append(LETTERS[t] + "z")
    for t in range(q):
        ops.append(U.conj())
        subs.append(LETTERS[p + t] + "z")
    ops.append(np.asarray(volumes, dtype=float))
    subs.append("z")
    return kappa * np.einsum(",".join(subs) + "->" + LETTERS[:n], *ops, optimize=True)


def forward_data(
    causet: CausalSet,
    scheme: BDScheme,
    cubic_representation: Optional[Tuple[int, int]],
    quartic_representation: Optional[Tuple[int, int]],
    g: float = 1.0,
    lam: float = 1.0,
    volumes: Optional[np.ndarray] = None,
    basis_seed: Optional[int] = 1,
) -> Tuple[np.ndarray, List[VertexData], np.ndarray, np.ndarray]:
    """Returns (omega, [vertex data...], B, U). Only omega and the vertex data are handed to the inverse."""
    if g <= 0 or lam <= 0:
        raise ValueError("convention: g > 0 (by phi -> -phi) and lambda > 0 (stability)")
    B = bd_matrix(causet, scheme)
    omega, U = diagonalize(hermitize(B), seed=basis_seed)
    n = causet.n
    nu = np.ones(n) if volumes is None else np.asarray(volumes, dtype=float)
    vertices: List[VertexData] = []
    if cubic_representation is not None:
        p, q = cubic_representation
        if p + q != 3:
            raise ValueError("cubic representation must have p+q=3")
        vertices.append(VertexData("cubic", p, q, representation_components(U, p, q, g, nu), g))
    if quartic_representation is not None:
        p, q = quartic_representation
        if p + q != 4:
            raise ValueError("quartic representation must have p+q=4")
        vertices.append(VertexData("quartic", p, q, representation_components(U, p, q, lam, nu), lam))
    if not vertices:
        raise ValueError("at least one vertex representation is required")
    return omega, vertices, B, U


# =============================================================================
# 3. Tensor contractions (thesis, Chapter 4)
# =============================================================================


def choose_split(p: int, q: int) -> Tuple[int, int]:
    """(p',q') such that two legs remain: (p-1,q-1) when possible, else the complex-symmetric route."""
    if p >= 1 and q >= 1:
        return p - 1, q - 1
    if p == 0:
        return 0, q - 2
    return p - 2, 0


def probe_contract(vd: VertexData, a: np.ndarray, pp: int, qq: int) -> np.ndarray:
    """Contract pp ket legs with <a| (i.e. a^*) and qq bra legs with |a>; return the 2-leg tensor."""
    p, q = vd.p, vd.q
    ops = [vd.T]
    subs = [LETTERS[: p + q]]
    for t in range(pp):
        ops.append(a.conj())
        subs.append(LETTERS[t])
    for t in range(qq):
        ops.append(a)
        subs.append(LETTERS[p + t])
    out = LETTERS[pp:p] + LETTERS[p + qq : p + q]
    return np.einsum(",".join(subs) + "->" + out, *ops, optimize=True)


def multilinear(vd: VertexData, kets_for_bra_legs: Sequence[np.ndarray], bras_for_ket_legs: Sequence[np.ndarray]) -> complex:
    """T(<u_1|,...,<u_p|, |w_1>,...,|w_q>) = sum T_{i..j..} u*_{i}.. w_{j}.."""
    p, q = vd.p, vd.q
    assert len(bras_for_ket_legs) == p and len(kets_for_bra_legs) == q
    ops = [vd.T]
    subs = [LETTERS[: p + q]]
    for t, u in enumerate(bras_for_ket_legs):
        ops.append(np.conj(u))
        subs.append(LETTERS[t])
    for t, w in enumerate(kets_for_bra_legs):
        ops.append(w)
        subs.append(LETTERS[p + t])
    return complex(np.einsum(",".join(subs) + "->", *ops, optimize=True))


def self_contraction(vd: VertexData, v: np.ndarray) -> complex:
    """s^{(p,q)} = T(<v|^{(x)p}, |v>^{(x)q}) = e^{i(q-p)theta} kappa nu for v = e^{i theta} e_m."""
    return multilinear(vd, [v] * vd.q, [v] * vd.p)


def lowdin_orthonormalize(W: np.ndarray) -> np.ndarray:
    """Symmetric (Lowdin) orthonormalization: closest unitary to W."""
    S = W.conj().T @ W
    evals, evecs = np.linalg.eigh(S)
    S_inv_half = evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-300))) @ evecs.conj().T
    return W @ S_inv_half


@dataclass
class ContractionResult:
    free_legs: Tuple[int, int]
    split: Tuple[int, int]
    matrix: np.ndarray
    columns: np.ndarray  # recovered |e_m^{(rec)}> as columns, in the spectral basis
    weights: np.ndarray  # c_m (free legs (1,1)) or |c_m| (complex-symmetric route)
    min_rel_gap: float
    nondegeneracy_on: str


def recover_columns(vd: VertexData, a: np.ndarray) -> ContractionResult:
    """One probe contraction and the matrix eigenproblem that returns the columns of U."""
    pp, qq = choose_split(vd.p, vd.q)
    free = (vd.p - pp, vd.q - qq)
    M = probe_contract(vd, a, pp, qq)

    if free == (1, 1):
        # M = U C U^dagger is normal: eigendecompose directly; eigenvalues are the complex weights c_m.
        w, W = np.linalg.eig(M)
        W = lowdin_orthonormalize(W)
        weights = np.array([W[:, k].conj() @ M @ W[:, k] for k in range(len(w))])
        cond = "c_m(a) pairwise distinct"
    elif free == (0, 2):
        # M = U^* C U^dagger complex symmetric: M^dagger M = U |C|^2 U^dagger.
        G = M.conj().T @ M
        ev, W = np.linalg.eigh(G)
        weights = np.sqrt(np.maximum(ev, 0.0)).astype(complex)
        cond = "|c_m(a)| pairwise distinct"
    elif free == (2, 0):
        # M = U C U^T complex symmetric: M M^dagger = U |C|^2 U^dagger.
        G = M @ M.conj().T
        ev, W = np.linalg.eigh(G)
        weights = np.sqrt(np.maximum(ev, 0.0)).astype(complex)
        cond = "|c_m(a)| pairwise distinct"
    else:
        raise RuntimeError(f"unexpected free legs {free}")

    diffs = np.abs(weights[:, None] - weights[None, :])
    np.fill_diagonal(diffs, np.inf)
    scale = max(float(np.max(np.abs(weights))), 1e-300)
    min_rel_gap = float(np.min(diffs)) / scale
    return ContractionResult(free, (pp, qq), M, W, weights, min_rel_gap, cond)


def eigenpair_ratio(vd: VertexData, cr: ContractionResult, a: np.ndarray, v: np.ndarray) -> complex:
    """
    rho^{(p,q)} = T^{(free)}( <v|^{(x)(p-p')}, |v>^{(x)(q-q')} ) / ( <a|v>^{p'} <v|a>^{q'} )
                = kappa nu e^{i(q-p)theta}.
    Equals the self-contraction (thesis, Section 5.7).
    """
    pp, qq = cr.split
    M = cr.matrix
    fp, fq = cr.free_legs
    if (fp, fq) == (1, 1):
        num = v.conj() @ M @ v
    elif (fp, fq) == (0, 2):
        num = v @ M @ v
    else:  # (2,0)
        num = v.conj() @ M @ v.conj()
    den = (a.conj() @ v) ** pp * (v.conj() @ a) ** qq
    return complex(num / den)


# =============================================================================
# 4. Phase fixing from the vertex data
# =============================================================================


def roots_of_unity(d: int) -> np.ndarray:
    return np.exp(2j * np.pi * np.arange(d) / d)


def residual_group_name(d: Optional[int]) -> str:
    if d is None or d == 0:
        return "U(1)"
    if d == 1:
        return "Z_1 (trivial)"
    return f"Z_{d}"


@dataclass
class PhaseFixResult:
    columns: np.ndarray  # rephased columns
    candidates: Optional[List[np.ndarray]]  # per column: admissible residual phase factors; None = U(1)
    residual_d: int  # 0 means U(1)
    log: List[str] = field(default_factory=list)


def fix_phases(columns: np.ndarray, vertices: List[VertexData], tol: float = 1e-6) -> PhaseFixResult:
    """
    For each recovered column, impose positivity of the self-contraction of every available vertex.
    Vertex 1 (smallest d>0) rephases to its principal root, leaving Z_{d1}; each further vertex
    keeps only the candidates in Z_{d1} that make its own self-contraction positive, leaving
    Z_gcd(d1,d2,...). Balanced representations (d=0) impose nothing.
    """
    n = columns.shape[1]
    cols = columns.copy()
    log: List[str] = []
    active = sorted([vd for vd in vertices if vd.d > 0], key=lambda vd: vd.d)
    if not active:
        log.append("All supplied representations are balanced (d=0): no vertex phase information, residual U(1).")
        return PhaseFixResult(cols, None, 0, log)

    first = active[0]
    d_res = first.d
    cand_sets: List[np.ndarray] = []
    for m in range(n):
        s = self_contraction(first, cols[:, m])
        if abs(s) < tol:
            raise ValueError(f"vanishing self-contraction for column {m}")
        # s = e^{i(q-p)theta} kappa nu; multiply by the principal root of (s/|s|)^{-1/(q-p)} = e^{-i theta}
        cols[:, m] *= np.exp(-1j * np.angle(s) / (first.q - first.p))
        cand_sets.append(roots_of_unity(first.d))
    log.append(f"{first.label}: d={first.d}, positivity of s_m leaves theta_m in {residual_group_name(first.d)}.")

    for vd in active[1:]:
        new_sets = []
        for m in range(n):
            keep = []
            for z in cand_sets[m]:
                s = self_contraction(vd, z * cols[:, m])
                if abs(s.imag) <= tol * abs(s) and s.real > 0:
                    keep.append(z)
            if not keep:
                raise ValueError(f"no residual phase of column {m} makes {vd.label} positive: vertices inconsistent")
            new_sets.append(np.array(keep))
        cand_sets = new_sets
        d_res = gcd(d_res, vd.d)
        log.append(f"{vd.label}: d={vd.d}; intersection leaves Z_gcd = {residual_group_name(d_res)} "
                   f"({len(cand_sets[0])} candidate(s) per column).")

    # Rephase each column by its first surviving candidate so candidates become exactly Z_{d_res}.
    for m in range(n):
        z0 = cand_sets[m][0]
        cols[:, m] *= z0
        cand_sets[m] = cand_sets[m] / z0
    return PhaseFixResult(cols, cand_sets, d_res, log)


# =============================================================================
# 5. BD legality: pairwise lemma + global consistency, searched over the residual group
# =============================================================================


def reconstruct_b_tilde(omega: np.ndarray, U_cols: np.ndarray) -> np.ndarray:
    """U_cols has the recovered |e_m> as columns (spectral components), i.e. U_cols = U^{(rec)}."""
    return U_cols.conj().T @ np.diag(omega) @ U_cols


def decode(B_tilde: np.ndarray, sign: int) -> np.ndarray:
    """D_+ (sign=+1): Re+Im, retarded;  D_- (sign=-1): Re-Im, advanced."""
    return np.real(B_tilde) + sign * np.imag(B_tilde)


def snap(B_raw: np.ndarray, scheme: BDScheme, tol: float) -> np.ndarray:
    legal = scheme.legal_offdiag
    B = B_raw.copy()
    n = B.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                B[i, j] = scheme.diagonal
                continue
            x = B_raw[i, j]
            if abs(x) < tol:
                B[i, j] = 0.0
            else:
                k = int(np.argmin(np.abs(legal - x)))
                B[i, j] = legal[k] if abs(legal[k] - x) < tol else x
    return B


def pair_is_legal(zi: complex, zj: complex, Bt_ij: complex, Bt_ji: complex, sign: int, legal: np.ndarray, tol: float) -> bool:
    """Lemma: for a comparable pair exactly one of the decoded entries is a legal coefficient, the other zero."""
    bij = decode(np.conj(zi) * zj * Bt_ij, sign)
    bji = decode(np.conj(zj) * zi * Bt_ji, sign)
    ij_zero, ji_zero = abs(bij) < tol, abs(bji) < tol
    ij_legal = np.min(np.abs(legal - bij)) < tol
    ji_legal = np.min(np.abs(legal - bji)) < tol
    return bool((ij_legal and ji_zero) or (ji_legal and ij_zero))


@dataclass
class LegalitySolution:
    z: np.ndarray
    columns: np.ndarray
    B_decoded: np.ndarray


def bd_legality_search(
    omega: np.ndarray,
    columns: np.ndarray,
    candidates: Optional[List[np.ndarray]],
    scheme: BDScheme,
    sign: int,
    pair_tol: float = 1e-3,
    magnitude_tol: float = 0.2,
    max_solutions: int = 1,
) -> List[LegalitySolution]:
    """
    Search the residual column phases for assignments that make every constrained pair legal and the
    decoded matrix globally self-consistent. Constrained pairs are read off the phase-invariant
    magnitudes |B_tilde_jk| of the reconstruction. The root of each link-connected component is fixed
    (a common phase on a component cancels). For a U(1) residual the relative phases of constrained
    pairs must lie in {0, 3pi/2} by the lemma, so Z_4 relative to the root is exhaustive.
    """
    n = columns.shape[1]
    Bt = reconstruct_b_tilde(omega, columns)
    legal = scheme.legal_offdiag
    adjacency: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Bt[i, j]) > magnitude_tol:
                adjacency[i].append(j)
                adjacency[j].append(i)

    # BFS order, components rooted at highest degree node
    order: List[int] = []
    roots: set = set()
    seen: set = set()
    for start in sorted(range(n), key=lambda v: -len(adjacency[v])):
        if start in seen:
            continue
        roots.add(start)
        seen.add(start)
        queue = [start]
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nb in sorted(adjacency[node], key=lambda v: -len(adjacency[v])):
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)

    if candidates is None:
        # U(1) residual: the Lemma quantizes the continuous relative phase of each constrained pair to
        # {0, 3pi/2}. Walk a spanning forest and rephase each column so that, relative to its parent,
        # the pair decodes with the retarded orientation (B_tilde_jk -> (1+i)/2 b). The other orientation
        # differs by the factor -i, so the remaining freedom is Z_4 relative to the root, which the
        # discrete search below explores; non-tree pairs are checked by the pairwise rule there.
        zq = np.ones(n, dtype=complex)
        seen2: set = set()
        for start in order:
            if start in seen2:
                continue
            seen2.add(start)
            queue = [start]
            while queue:
                i = queue.pop(0)
                for j in adjacency[i]:
                    if j in seen2:
                        continue
                    seen2.add(j)
                    queue.append(j)
                    # B_tilde'_ij = conj(z_i) z_j B_tilde_ij. Pick the legal b with |b| = sqrt(2)|B_tilde_ij|
                    # and choose z_j so that B_tilde'_ij = (1+i)/2 * b (retarded orientation for this pair).
                    mag = math.sqrt(2.0) * abs(Bt[i, j])
                    b = legal[int(np.argmin(np.abs(np.abs(legal) - mag)))]
                    zj = 0.5 * (1 + 1j) * b / (np.conj(zq[i]) * Bt[i, j])
                    zq[j] = zj / abs(zj)
        columns = columns @ np.diag(zq)
        Bt = reconstruct_b_tilde(omega, columns)
        cand = [roots_of_unity(4) for _ in range(n)]
    else:
        cand = [np.asarray(c) for c in candidates]

    z: List[Optional[complex]] = [None] * n
    solutions: List[LegalitySolution] = []

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            zarr = np.array(z, dtype=complex)
            cols = columns @ np.diag(zarr)
            Bd = snap(decode(reconstruct_b_tilde(omega, cols), sign), scheme, pair_tol)
            if is_self_consistent_bd(Bd, scheme, pair_tol):
                solutions.append(LegalitySolution(zarr, cols, Bd))
                return len(solutions) >= max_solutions
            return False
        node = order[pos]
        assigned = [nb for nb in adjacency[node] if z[nb] is not None]
        choices = [cand[node][0]] if (node in roots and not assigned) else cand[node]
        for zc in choices:
            z[node] = zc
            if all(pair_is_legal(z[min(node, nb)], z[max(node, nb)], Bt[min(node, nb), max(node, nb)],
                                 Bt[max(node, nb), min(node, nb)], sign, legal, pair_tol) for nb in assigned):
                if backtrack(pos + 1):
                    return True
            z[node] = None
        return False

    backtrack(0)
    return solutions


# =============================================================================
# 6. The blind inverse pipeline
# =============================================================================


@dataclass
class ReconstructionReport:
    recover_from: str
    contraction: ContractionResult
    probe: np.ndarray
    attempt: int
    residual_before_bd: int
    bd_needed: bool
    four_divides_d: bool
    solutions_plus: List[LegalitySolution]
    solutions_minus: List[LegalitySolution]
    columns_final: np.ndarray
    B_tilde_rec: np.ndarray
    B_plus: np.ndarray
    B_minus: np.ndarray
    causet_plus: CausalSet
    checks: Dict[str, object]
    log: List[str]


def reconstruct(
    omega: np.ndarray,
    vertices: List[VertexData],
    scheme: BDScheme = BD_3P1,
    recover_from: Optional[str] = None,
    seed: int = 0,
    max_probe_attempts: int = 100,
    min_rel_gap: float = 1e-6,
    pair_tol: float = 1e-3,
    enumerate_solutions: int = 1,
) -> ReconstructionReport:
    """Blind inverse: inputs are omega and the vertex representation tensors only."""
    omega = np.asarray(omega, dtype=float)
    n = len(omega)
    log: List[str] = []
    rng = np.random.default_rng(seed)
    if not scheme.check_no_negation():
        raise ValueError("BD scheme shares a coefficient with its negation; phase quantization fails")

    # --- choose the vertex to contract: prefer a (1,1) route, and the cubic among those
    def route_rank(vd: VertexData) -> Tuple[int, int]:
        pp, qq = choose_split(vd.p, vd.q)
        free = (vd.p - pp, vd.q - qq)
        return (0 if free == (1, 1) else 1, 0 if vd.name == "cubic" else 1)

    if recover_from is None:
        vd_rec = sorted(vertices, key=route_rank)[0]
    else:
        vd_rec = next(vd for vd in vertices if vd.name == recover_from)
    log.append(f"Recovering U from {vd_rec.label}; split {choose_split(vd_rec.p, vd_rec.q)}.")

    # --- 1. probe contraction and eigendecomposition
    cr = None
    for attempt in range(1, max_probe_attempts + 1):
        a = rng.normal(size=n) + 1j * rng.normal(size=n)
        cr_try = recover_columns(vd_rec, a)
        if cr_try.min_rel_gap < min_rel_gap or np.min(np.abs(cr_try.weights)) < 1e-10:
            continue
        cr = cr_try
        probe = a
        break
    if cr is None:
        raise RuntimeError("no probe gave nondegenerate weights")
    log.append(f"Probe attempt {attempt}: free legs {cr.free_legs}, nondegeneracy on {cr.nondegeneracy_on}, "
               f"min relative gap {cr.min_rel_gap:.2e}.")

    # --- 2. phase fixing from the vertices; check ratio == self-contraction for the recovery representation
    pf = fix_phases(cr.columns, vertices)
    log.extend(pf.log)
    ratio_err = 0.0
    for m in range(n):
        rho = eigenpair_ratio(vd_rec, cr, probe, cr.columns[:, m])
        s = self_contraction(vd_rec, cr.columns[:, m])
        ratio_err = max(ratio_err, abs(rho - s) / max(abs(s), 1e-300))
    log.append(f"Eigenpair ratio vs self-contraction ({vd_rec.label}): max relative discrepancy {ratio_err:.2e}.")

    d_res = pf.residual_d
    four_div = (d_res % 4 == 0)  # includes d_res == 0 (U(1))
    bd_needed = d_res != 1
    if four_div:
        log.append(f"Residual {residual_group_name(d_res)} contains 3pi/2 (4 | d): BD legality cannot close it; "
                   f"blind spots possible.")
    elif bd_needed:
        log.append(f"Residual {residual_group_name(d_res)} excludes 3pi/2 (4 does not divide d): "
                   f"BD legality forces all relative phases to 0.")
    else:
        log.append("Residual trivial: phases fully fixed by the vertex data; BD legality only decodes.")

    # --- 3. BD legality search (reduces to a single decode when the residual is trivial)
    sols_plus = bd_legality_search(omega, pf.columns, pf.candidates, scheme, +1, pair_tol,
                                   max_solutions=max(1, enumerate_solutions))
    sols_minus = bd_legality_search(omega, pf.columns, pf.candidates, scheme, -1, pair_tol,
                                    max_solutions=max(1, enumerate_solutions))
    if not sols_plus and not sols_minus:
        raise RuntimeError("no residual phase assignment yields a legal BD matrix")
    sol = sols_plus[0] if sols_plus else sols_minus[0]
    cols = sol.columns
    Bt = reconstruct_b_tilde(omega, cols)
    B_plus = snap(decode(Bt, +1), scheme, pair_tol)
    B_minus = snap(decode(Bt, -1), scheme, pair_tol)
    log.append(f"BD legality search: {len(sols_plus)} solution(s) under D_+, {len(sols_minus)} under D_- "
               f"(enumeration cap {enumerate_solutions}).")

    # --- 4. decode and checks
    causet_plus = causet_from_bd(B_plus, scheme)
    checks: Dict[str, object] = {}
    checks["B_plus_self_consistent"] = is_self_consistent_bd(B_plus, scheme, pair_tol)
    checks["B_minus_self_consistent"] = is_self_consistent_bd(B_minus, scheme, pair_tol)
    checks["D_minus_is_transpose_of_D_plus"] = bool(np.allclose(B_minus, B_plus.T, atol=pair_tol))

    # joint locality: every vertex must be diagonal in the recovered basis; s^{(3)}/s^{(4)} constant
    s_by_vertex: Dict[str, np.ndarray] = {}
    for vd in vertices:
        ops = [vd.T]
        subs = [LETTERS[: vd.order]]
        for t in range(vd.p):
            ops.append(cols.conj())
            subs.append(LETTERS[t] + "z")
        for t in range(vd.q):
            ops.append(cols)
            subs.append(LETTERS[vd.p + t] + "z")
        # each leg gets its own position index
        out = ""
        subs2 = [subs[0]]
        for t in range(vd.order):
            subs2.append(subs[t + 1][0] + LETTERS[8 + t])
            out += LETTERS[8 + t]
        T_pos = np.einsum(",".join(subs2) + "->" + out, *ops, optimize=True)
        diag = np.array([T_pos[(m,) * vd.order] for m in range(n)])
        total = np.linalg.norm(T_pos)
        offdiag = math.sqrt(max(total**2 - np.linalg.norm(diag) ** 2, 0.0))
        checks[f"{vd.label} off-diagonal fraction in recovered basis"] = float(offdiag / max(total, 1e-300))
        checks[f"{vd.label} self-contractions real & positive"] = bool(
            np.all(diag.real > 0) and np.max(np.abs(diag.imag)) < 1e-6 * np.max(np.abs(diag)))
        s_by_vertex[vd.name] = diag
        if vd.kappa_known is not None:
            checks[f"{vd.label} recovered volumes nu_m (= s_m/kappa)"] = (diag.real / vd.kappa_known)
    if "cubic" in s_by_vertex and "quartic" in s_by_vertex:
        r = s_by_vertex["cubic"] / s_by_vertex["quartic"]
        checks["s3/s4 ratio: mean, max relative deviation"] = (complex(np.mean(r)).real,
                                                                float(np.max(np.abs(r - np.mean(r))) / abs(np.mean(r))))

    return ReconstructionReport(vd_rec.label, cr, probe, attempt, d_res, bd_needed, four_div,
                                sols_plus, sols_minus, cols, Bt, B_plus, B_minus, causet_plus, checks, log)


# =============================================================================
# 7. Post-hoc validation (NOT used by the inverse)
# =============================================================================


def classify(original: CausalSet, recovered: CausalSet) -> Tuple[str, Optional[Dict[int, int]]]:
    iso, mapping = original.is_isomorphic_to(recovered)
    if iso:
        return "isomorphic to the input (same orientation)", mapping
    iso_r, mapping_r = original.reversed().is_isomorphic_to(recovered)
    if iso_r:
        return "isomorphic to the global time reversal of the input", mapping_r
    return "NOT isomorphic to the input or its reversal (blind spot)", None


def example_20_node_bd() -> np.ndarray:
    """A stored 20-element causal set in 3+1 dimensions, used when no causal set is given."""
    return np.array([
        [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [-9., 1., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [16., 1., 1., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [-9., 0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [16., -9., 0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [-8., -9., -9., 0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [16., 0., -9., 0., 0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [16., 0., -9., 0., 0., 1., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 16., -9., -9., 0., 1., 1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., -8., 0., 0., 16., 0., 0., 1., 1., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., -8., 16., 16., 0., -9., -9., 0., 0., 1., 0., -1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., -8., 0., 0., -9., -9., 0., 1., 0., -1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., -8., -8., 0., 16., 16., 16., 16., -9., -9., 1., 1., -1., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0.],
        [-9., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -1., 0., 0., 0.],
        [-9., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., -1., 0., 0.],
        [16., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -9., 1., 0., -1., 0.],
        [-8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 16., -9., 0., 1., -1.],
    ])


BLIND_SPOT_NAMES = ["p", "q", "r_1", "r_2", "r_3", "r_4"]  # element order of blind_spot_example()


def blind_spot_example() -> CausalSet:
    """Two minimal elements p,q below four maximal elements r_1..r_4 (thesis, Section 5.5.3)."""
    G = nx.DiGraph()
    G.add_nodes_from(range(6))
    for r in range(2, 6):
        G.add_edge(0, r)
        G.add_edge(1, r)
    return CausalSet.from_links(G, 6)


# =============================================================================
# 8. Optional figure
# =============================================================================


def plot_panels(panels: Sequence[Tuple[CausalSet, str]], path: str, max_cols: int = 3,
                labels: Optional[Sequence[Optional[Sequence[str]]]] = None, highlight: bool = False) -> None:
    """
    Draw each (causal set, title) in its own panel, wrapping into rows of at most max_cols.
    labels, if given, holds one list of element names per panel (None for integer labels).
    highlight: draw in colour the links that are not common to all panels, and the elements all of
    whose links differ. Only meaningful when element k is the same element in every panel
    (--match-labels).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure")
        return

    def layout(G: nx.DiGraph, sweeps: int = 20,
               ref: Optional[Dict[int, Tuple[float, float]]] = None) -> Dict[int, Tuple[float, float]]:
        """
        Layered DAG layout with barycentric crossing reduction (same as the paper's figure).
        If ref (the positions of another panel) is given, every element that is in the same row as in
        ref keeps the left-to-right order it has there, and only the elements whose row has changed are
        inserted, next to their neighbours. The panels then differ only where the causal sets differ.
        """
        gen: Dict[int, int] = {}
        for v in nx.topological_sort(G):
            gen[v] = max((gen[u] for u in G.predecessors(v)), default=-1) + 1
        # The row of an element is the length of the longest chain below it, which puts every minimal
        # element in the bottom row. A minimal element is unrelated to everything except its own future,
        # so its height is a drawing choice: draw it one row below its lowest child, which keeps its
        # links short.
        for v in G.nodes():
            if G.in_degree(v) == 0 and G.out_degree(v) > 0:
                gen[v] = min(gen[w] for w in G.successors(v)) - 1
        layers: Dict[int, List[int]] = {}
        for v, g_ in gen.items():
            layers.setdefault(g_, []).append(v)
        if ref is not None:
            def centred(nodes: List[int]) -> Dict[int, float]:
                return {v: i - (len(nodes) - 1) / 2.0 for i, v in enumerate(nodes)}
            kept = {g_: sorted((v for v in nodes if round(ref[v][1]) == g_), key=lambda v: ref[v][0])
                    for g_, nodes in layers.items()}
            for v in sorted(v for v in G.nodes() if round(ref[v][1]) != gen[v]):
                xs = {u: xu for nodes in kept.values() for u, xu in centred(nodes).items()}
                nbrs = [xs[u] for u in list(G.predecessors(v)) + list(G.successors(v)) if u in xs]
                target = float(np.mean(nbrs)) if nbrs else 0.0
                row = kept.setdefault(gen[v], [])
                best = min(range(len(row) + 1),
                           key=lambda i: abs(centred(row[:i] + [v] + row[i:])[v] - target))
                row.insert(best, v)
            return {v: (xv, float(g_)) for g_, nodes in kept.items() for v, xv in centred(nodes).items()}
        x: Dict[int, float] = {}
        for nodes in layers.values():
            for i, v in enumerate(sorted(nodes)):
                x[v] = float(i)
        for _ in range(sweeps):
            for g_ in sorted(layers):
                for v in layers[g_]:
                    preds = list(G.predecessors(v))
                    if preds:
                        x[v] = float(np.mean([x[u] for u in preds]))
                layers[g_].sort(key=lambda v: x[v])
                for i, v in enumerate(layers[g_]):
                    x[v] = float(i)
            for g_ in sorted(layers, reverse=True):
                for v in layers[g_]:
                    succs = list(G.successors(v))
                    if succs:
                        x[v] = float(np.mean([x[u] for u in succs]))
                layers[g_].sort(key=lambda v: x[v])
                for i, v in enumerate(layers[g_]):
                    x[v] = float(i)
        pos = {}
        for g_, nodes in layers.items():
            for i, v in enumerate(nodes):
                pos[v] = (i - (len(nodes) - 1) / 2.0, float(g_))
        return pos

    def tex_name(name: str) -> str:
        """Names with a subscript, e.g. r_1, are typeset as bold math so they match the plain labels."""
        if "_" not in name:
            return name
        base, sub = name.split("_", 1)
        return rf"$\mathbf{{{base}}}_{{\mathbf{{{sub}}}}}$"

    k = len(panels)
    ncols = min(k, max_cols)
    nrows = (k + ncols - 1) // ncols
    # Compact panels (about 5.6 x 6.4 inches each) keep the nodes and labels large relative to the
    # figure, so they stay readable when the figure is scaled down to the text width.
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 6.4 * nrows),
                             constrained_layout=True, squeeze=False)
    for ax in axes.flat[k:]:
        ax.axis("off")
    edge_sets = [set(cs.link_graph().edges()) for cs, _ in panels]
    common = set.intersection(*edge_sets) if (highlight and len(panels) > 1) else None
    ref_pos: Optional[Dict[int, Tuple[float, float]]] = None
    for i_panel, (ax, (cs, title)) in enumerate(zip(axes.flat, panels)):
        G = cs.link_graph()
        pos = layout(G, ref=ref_pos if highlight else None)
        if i_panel == 0:
            ref_pos = pos
        names = labels[i_panel] if labels is not None and i_panel < len(labels) else None
        node_labels = None if names is None else {v: tex_name(names[v]) for v in G.nodes()}
        changed = [] if common is None else [e for e in G.edges() if e not in common]
        same = [e for e in G.edges() if e not in set(changed)]
        moved = {v for v in G.nodes() if G.degree(v) > 0
                 and all(e in set(changed) for e in list(G.in_edges(v)) + list(G.out_edges(v)))} if changed else set()
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=560, edgecolors="black", linewidths=1.2,
                               node_color=(["#f6b26b" if v in moved else "lightblue" for v in G.nodes()]
                                           if moved else "lightblue"))
        nx.draw_networkx_labels(G, pos, ax=ax, labels=node_labels, font_size=11, font_weight="bold")
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=same, arrows=True, arrowsize=18, width=1.5,
                               edge_color="black", node_size=560)
        if changed:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=changed, arrows=True, arrowsize=20, width=2.6,
                                   edge_color="#cc0000", node_size=560)
        ax.set_title(title, fontsize=14, pad=10)
        ax.axis("off")
        if pos and i_panel % ncols == 0:  # time axis labels once per row, on the left-most panel
            ys = [p[1] for p in pos.values()]
            xs = [p[0] for p in pos.values()]
            x_text = min(xs) - 0.95
            ax.text(x_text, min(ys), "Past", rotation=90, va="center", ha="center", fontsize=12)
            ax.text(x_text, max(ys), "Future", rotation=90, va="center", ha="center", fontsize=12)
    fig.savefig(path, dpi=300, bbox_inches="tight")  # dpi applies to raster formats; .pdf is vector
    plt.close(fig)
    print(f"figure saved to {path}")


# =============================================================================
# 9. Command line
# =============================================================================


def parse_representation(s: str) -> Optional[Tuple[int, int]]:
    if s.lower() == "none":
        return None
    p, q = (int(x) for x in s.split(","))
    return p, q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--sprinkle", type=int, metavar="N", help="sprinkle N points into a unit box of Minkowski spacetime (dimension set by --scheme)")
    src.add_argument("--example20", action="store_true", help="use a stored 20-element causal set (the default if no causal set is given)")
    src.add_argument("--blindspot", action="store_true", help="use the 6-element blind-spot causal set")
    src.add_argument("--bd-file", type=str, help="load a normalized BD matrix from a .npy / .txt file")
    src.add_argument("--links", type=str, metavar="RELATIONS",
                     help="causal set from relations, e.g. \"a<b<e,a<c<e,a<d\" (x<y: x precedes y)")
    ap.add_argument("--scheme", choices=list(SCHEMES), default="3+1")
    ap.add_argument("--cubic", type=str, default="1,2", help="cubic representation p,q (p+q=3) or 'none'")
    ap.add_argument("--quartic", type=str, default="1,3", help="quartic representation p,q (p+q=4) or 'none'")
    ap.add_argument("--recover-from", choices=["cubic", "quartic"], default=None,
                    help="which vertex to contract for U (default: a (1,1) route, cubic preferred)")
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--random-volumes", action="store_true", help="use random nu_m instead of nu_m = 1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enumerate", type=int, default=1, help="largest number of consistent phase assignments returned for each decoder (the search is exhaustive unless this number is reached)")
    ap.add_argument("--figure", type=str, default=None, help="save a side-by-side figure to this path")
    ap.add_argument("--labels", choices=["letters", "numbers"], default="letters",
                    help="how elements are labelled in the output and figures (default: letters)")
    ap.add_argument("--match-labels", action="store_true",
                    help="label each recovered element by the input element it corresponds to, instead of by "
                         "its position in the recovered basis (post-hoc, for illustration; always on for "
                         "--blindspot)")
    ap.add_argument("--highlight", action="store_true",
                    help="with --match-labels (or --blindspot): draw in colour the links that differ between "
                         "the panels of the figure")
    ap.add_argument("--only-blind-spots", action="store_true",
                    help="in the figure, show the input and only the candidates that are not isomorphic to it "
                         "or to its time reversal")
    ap.add_argument("--draw-only", action="store_true",
                    help="print the BD matrix and draw the input causal set only; skip the reconstruction")
    args = ap.parse_args()

    scheme = SCHEMES[args.scheme]
    custom_names: Optional[List[str]] = None
    if args.links:
        causet, custom_names = causet_from_link_string(args.links)
    elif args.sprinkle:
        causet = sprinkle_minkowski(args.sprinkle, dim=4 if scheme is BD_3P1 else 2, seed=args.seed)
    elif args.blindspot:
        causet = blind_spot_example()
        custom_names = list(BLIND_SPOT_NAMES)
    elif args.bd_file:
        B = np.load(args.bd_file) if args.bd_file.endswith(".npy") else np.loadtxt(args.bd_file)
        causet = causet_from_bd(B, scheme)
    else:
        causet = causet_from_bd(example_20_node_bd(), BD_3P1)

    # Element names: the input uses the names given with --links (if any); recovered elements are
    # named by their position in the recovered basis.
    rec_names = element_names(causet.n, args.labels)
    in_names = custom_names if custom_names is not None else rec_names

    if args.draw_only:
        B = bd_matrix(causet, scheme)
        print(f"Causal set: N={causet.n}, links={causet.link_graph().number_of_edges()}, scheme {scheme.name}")
        print("Element order: " + ", ".join(in_names))
        print("Links: " + ", ".join(f"{in_names[u]}<{in_names[v]}" for u, v in sorted(causet.link_graph().edges())))
        print("Normalized BD matrix (row x, column y; nonzero only when y precedes x):")
        print(np.array2string(B, formatter={"float_kind": lambda x: f"{x:g}"}))
        if args.figure:
            plot_panels([(causet, "")], args.figure, labels=[in_names])
        return

    cubic = parse_representation(args.cubic)
    quartic = parse_representation(args.quartic)
    rng = np.random.default_rng(args.seed + 1)
    volumes = rng.uniform(0.5, 1.5, size=causet.n) if args.random_volumes else None

    omega, vertices, B_true, _U_true = forward_data(causet, scheme, cubic, quartic, args.g, args.lam,
                                                   volumes, basis_seed=args.seed + 2)
    print(f"Causal set: N={causet.n}, links={causet.link_graph().number_of_edges()}, scheme {scheme.name}")
    print("Input data: spectrum of B_tilde and " + ", ".join(vd.label for vd in vertices))
    print(f"Spectrum: min {omega.min():.4f}, max {omega.max():.4f}, sum {omega.sum():.4f}, "
          f"sum of squares {float(np.sum(omega ** 2)):.4f}")
    print("-" * 78)

    rep = reconstruct(omega, vertices, scheme, args.recover_from, args.seed, enumerate_solutions=args.enumerate)
    for line in rep.log:
        print(line)

    matched = args.match_labels or args.blindspot
    if matched:
        # Post-hoc only: this uses the true U, which the inverse stage never sees. Each recovered column
        # is a position vector of the input up to a phase, so recovered element m is the input element k
        # with |<e_k|e_m^(rec)>| = 1. Naming it after that element removes the relabeling from the
        # figure and leaves only the difference in causal order visible.
        overlap = np.abs(_U_true.conj().T @ rep.columns_final)
        owner = np.argmax(overlap, axis=0)
        if sorted(owner.tolist()) != list(range(causet.n)) or np.min(np.max(overlap, axis=0)) < 0.99:
            raise RuntimeError("could not match the recovered elements to the input elements")
        rec_names = list(in_names)

    def to_input_order(cs: CausalSet) -> CausalSet:
        """With matched labels, re-index a recovered causal set so element k is input element k."""
        if not matched:
            return cs
        prec = np.zeros_like(cs.prec)
        prec[np.ix_(owner, owner)] = cs.prec
        return CausalSet(prec)
    print("-" * 78)
    for k, v in rep.checks.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: {np.round(v, 4)}")
        else:
            print(f"{k}: {v}")
    print("-" * 78)
    verdict, mapping = classify(causet, rep.causet_plus)
    print(f"Post-hoc: D_+ reconstruction is {verdict}.")
    if matched:
        print("Recovered elements are named after the input elements they correspond to (post-hoc matching).")
    elif mapping is not None:
        if args.labels == "numbers" and custom_names is None:
            print("Relabeling (input -> recovered):", [mapping[i] for i in range(causet.n)])
        else:
            print("Relabeling (input -> recovered): "
                  + ", ".join(f"{in_names[i]}->{rec_names[mapping[i]]}" for i in range(causet.n)))
    if volumes is not None:
        rec = [v for k, v in rep.checks.items() if "recovered volumes" in k][0]
        print(f"Post-hoc: recovered volumes match the sprinkled nu_m up to relabeling: "
              f"{np.allclose(np.sort(rec), np.sort(volumes), atol=1e-8)}")
    # Collect the BD-legal solutions found (both decoders) that are pairwise distinct up to relabeling AND
    # global time reversal, which is the equivalence the reconstruction theorem is stated under. With a
    # trivial residual, or 4 ∤ d, there is exactly one; with 4 | d (0->4 or 2->2 alone) the blind spots
    # show up here as extra candidates.
    def same_up_to_reversal(a: CausalSet, b: CausalSet) -> bool:
        return a.is_isomorphic_to(b)[0] or a.reversed().is_isomorphic_to(b)[0]

    distinct: List[Tuple[CausalSet, str]] = []
    for sign, sols in ((+1, rep.solutions_plus), (-1, rep.solutions_minus)):
        for s in sols:
            cs = to_input_order(causet_from_bd(s.B_decoded, scheme))
            if not any(same_up_to_reversal(cs, d) for d, _ in distinct):
                distinct.append((cs, "D_+" if sign > 0 else "D_-"))
    n_found = len(rep.solutions_plus) + len(rep.solutions_minus)
    if args.enumerate > 1:
        print(f"Among {n_found} BD-legal solutions found (D_+ and D_-), {len(distinct)} are pairwise "
              f"distinct up to relabeling and global time reversal.")
        for i, (cs, dec) in enumerate(distinct):
            verdict_i, _ = classify(causet, cs)
            print(f"  candidate {i + 1} ({dec}, {cs.link_graph().number_of_edges()} links): {verdict_i}")
            if matched:
                print("    links: " + ", ".join(sorted(f"{rec_names[u]}<{rec_names[v]}"
                                                        for u, v in cs.link_graph().edges())))

    if args.figure:
        panels: List[Tuple[CausalSet, str]] = [(causet, "(a) Input causal set")]
        panel_names: List[Sequence[str]] = [in_names]
        blind = [(cs, dec) for cs, dec in distinct if classify(causet, cs)[0].startswith("NOT")]
        if args.only_blind_spots and not blind:
            print("--only-blind-spots: no non-isomorphic candidate was found; drawing the recovered causal set.")
        if args.only_blind_spots and blind:
            for i, (cs, dec) in enumerate(blind):
                number = f" {i + 1}" if len(blind) > 1 else ""
                panels.append((cs, f"({chr(ord('b') + i)}) BD-legal reconstruction{number}, ${dec}$: "
                                   f"not isomorphic to (a)"))
                panel_names.append(rec_names)
        elif len(distinct) <= 1:
            panels.append((to_input_order(rep.causet_plus), r"(b) Recovered $D_+$ branch"))
            panel_names.append(rec_names)
        else:
            for i, (cs, dec) in enumerate(distinct):
                verdict_i, _ = classify(causet, cs)
                short = ("non-isomorphic (blind spot)" if verdict_i.startswith("NOT")
                         else "time reversal of input" if "reversal" in verdict_i
                         else "isomorphic to input")
                panels.append((cs, f"({chr(ord('b') + i)}) BD-legal candidate {i + 1}, ${dec}$: {short}"))
                panel_names.append(rec_names)
        plot_panels(panels, args.figure, labels=panel_names, highlight=args.highlight and matched)


if __name__ == "__main__":
    main()