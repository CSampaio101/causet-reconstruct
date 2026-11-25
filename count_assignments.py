"""
count_assignments.py
====================

Counts the assignments of column phases that are left by the Lemma (BD legality applied pair by pair) when
the vertex data leave Z_4 or U(1), and how many of them are legal.

For every pair of elements with a nonzero Benincasa-Dowker coefficient, the Lemma allows the relative phase
0, which keeps the entry where it is, or 3pi/2, which moves it to the transposed position. The script
enumerates every assignment of phases in Z_4 (one phase per connected component held fixed) that satisfies
this on all pairs, and tests each one globally: the decoded matrix must be the Benincasa-Dowker matrix of
the causal set defined by its own links.

Put this file next to causet_reconstruct.py and run, for the causal set of Chapter 7:

    python count_assignments.py --links "a<b,a<c,a<p,b<d,b<e,c<e,c<f,d<g,e<h,f<i,f<j,g<k,h<k,i<l,j<l,k<m,l<n,m<o,n<o,p<q,p<r,q<s,s<t,f<v<l,f<w<l,i<u,j<u,v<u,w<u"

which prints 3222 assignments, 2 of them legal.
"""
import argparse

import numpy as np

import causet_reconstruct as cr


def count(causet: "cr.CausalSet", scheme: "cr.BDScheme"):
    """Returns (number of assignments allowed by the Lemma, list of the distinct legal decoded matrices)."""
    n = causet.n
    B = cr.bd_matrix(causet, scheme)
    pairs = [(k, j, B[j, k]) for j in range(n) for k in range(n) if j != k and B[j, k] != 0]  # k precedes j
    adj = {i: [] for i in range(n)}
    for x, y, _ in pairs:
        adj[x].append((y, +1))
        adj[y].append((x, -1))
    order, seen, roots = [], set(), set()
    for start in range(n):
        if start in seen:
            continue
        roots.add(start)
        seen.add(start)
        queue = [start]
        while queue:
            v = queue.pop(0)
            order.append(v)
            for w, _ in adj[v]:
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
    theta = [None] * n  # phases in units of pi/2
    allowed = 0
    legal = []

    def visit(i: int) -> None:
        nonlocal allowed
        if i == n:
            allowed += 1
            Bd = np.zeros((n, n))
            np.fill_diagonal(Bd, scheme.diagonal)
            for x, y, c in pairs:
                if (theta[y] - theta[x]) % 4 == 0:
                    Bd[y, x] = c
                else:
                    Bd[x, y] = c
            if cr.is_self_consistent_bd(Bd, scheme) and not any(np.array_equal(Bd, M) for M in legal):
                legal.append(Bd)
            return
        v = order[i]
        for k in ([0] if v in roots else range(4)):
            if all(theta[w] is None or ((k - theta[w]) * s) % 4 in (0, 3) for w, s in adj[v]):
                theta[v] = k
                visit(i + 1)
                theta[v] = None

    visit(0)
    return allowed, legal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--links", type=str, help='relations, e.g. "a<b<e,a<c<e,a<d,e<g"')
    src.add_argument("--blindspot", action="store_true", help="the six-element blind-spot causal set")
    ap.add_argument("--scheme", choices=list(cr.SCHEMES), default="3+1")
    args = ap.parse_args()
    scheme = cr.SCHEMES[args.scheme]
    causet = cr.blind_spot_example() if args.blindspot else cr.causet_from_link_string(args.links)[0]
    allowed, legal = count(causet, scheme)
    nonzero = int(np.sum(cr.bd_matrix(causet, scheme) != 0)) - causet.n
    print(f"Causal set: N={causet.n}, pairs with a nonzero coefficient: {nonzero}")
    print(f"Phase assignments that satisfy the Lemma on all pairs: {allowed}")
    print(f"Of these, legal (decoded matrix is the BD matrix of its own links): {len(legal)}")
    sets = [cr.causet_from_bd(M, scheme) for M in legal]
    distinct = []
    for cs in sets:
        if not any(cs.is_isomorphic_to(d)[0] for d in distinct):
            distinct.append(cs)
    print(f"Distinct causal sets among them, up to relabeling: {len(distinct)}")


if __name__ == "__main__":
    main()
