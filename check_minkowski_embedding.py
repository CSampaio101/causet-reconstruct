"""
check_minkowski_embedding.py
============================

Checks that the 23 points listed in example23_minkowski_coordinates.txt, taken as points of 3+1
dimensional Minkowski spacetime, have exactly the causal relations of the causal set of Chapter 7
of the thesis. A point x precedes a point y if y lies inside the future light cone of x.

Put this file next to causet_reconstruct.py and run

    python check_minkowski_embedding.py
"""
import numpy as np

import causet_reconstruct as cr

LINKS = ("a<b,a<c,a<p,b<d,b<e,c<e,c<f,d<g,e<h,f<i,f<j,g<k,h<k,i<l,j<l,k<m,l<n,m<o,n<o,"
         "p<q,p<r,q<s,s<t,f<v<l,f<w<l,i<u,j<u,v<u,w<u")


def main() -> None:
    causet, names = cr.causet_from_link_string(LINKS)
    points = np.loadtxt("example23_minkowski_coordinates.txt")  # rows: t x y z, elements a..w
    n = causet.n
    assert points.shape == (n, 4) and names == sorted(names)
    order = causet.prec.T.copy()           # order[x, y]: x precedes y in the causal set
    cones = np.zeros((n, n), dtype=bool)   # cones[x, y]: y is inside the future light cone of x
    for x in range(n):
        for y in range(n):
            if x != y:
                dt = points[y, 0] - points[x, 0]
                dx = np.linalg.norm(points[y, 1:] - points[x, 1:])
                cones[x, y] = dt > dx
    print(f"relations of the causal set: {int(order.sum())}")
    print(f"relations of the 23 points:  {int(cones.sum())}")
    print(f"identical: {bool(np.array_equal(order, cones))}")


if __name__ == "__main__":
    main()
