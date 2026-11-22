"""Periodic-consistent site descriptors.

Anchors of a site are placed in one compact set of periodic images (the image
combination minimising the summed pairwise anchor distances), neighbours are
tracked as (atom, image) pairs relative to those images, and the anchors'
own images are excluded.  Non-periodic structures reduce to the usual GCN.
"""
from itertools import product
import numpy as np

GCN_CN_MAX = {"T": 12.0, "B": 18.0, "H": 22.0, "F": 26.0}


def compact_anchor_images(anchors, positions, cell, pbc, tol=1e-6):
    """All minimal-spread image assignments (first anchor kept in the home cell)."""
    zero = (0, 0, 0)
    if not np.any(pbc) or len(anchors) == 1:
        return [[zero] * len(anchors)]
    cell = np.asarray(cell, float)
    axes = [range(-1, 2) if p else range(0, 1) for p in pbc]
    images = list(product(*axes))
    best, combos = np.inf, []
    for choice in product(images, repeat=len(anchors) - 1):
        shifts = [zero] + list(choice)
        pts = np.array([positions[a] + np.dot(s, cell) for a, s in zip(anchors, shifts)])
        spread = sum(np.linalg.norm(pts[i] - pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts)))
        if spread < best - tol:
            best, combos = spread, [shifts]
        elif abs(spread - best) <= tol:
            combos.append(shifts)
    return combos


def gcn_for_images(anchors, shifts, site_type, neighbors, cn):
    images = set()
    for anchor, s in zip(anchors, shifts):
        for j, shift in neighbors[anchor]:
            images.add((j, tuple(int(a) + int(b) for a, b in zip(shift, s))))
    for anchor, s in zip(anchors, shifts):
        images.discard((anchor, tuple(s)))
    return sum(float(cn[j]) for j, _ in images) / GCN_CN_MAX[site_type]


def site_gcn_pbc(anchors, site_type, neighbors, cn, positions, cell, pbc):
    """GCN averaged over symmetry-tied compact image choices; also returns their spread."""
    values = [gcn_for_images(anchors, s, site_type, neighbors, cn)
              for s in compact_anchor_images(anchors, positions, cell, pbc)]
    return float(np.mean(values)), float(max(values) - min(values))


def compact_centroid(anchors, positions, cell, pbc):
    shifts = compact_anchor_images(anchors, positions, cell, pbc)[0]
    cell = np.asarray(cell, float)
    return np.mean([positions[a] + np.dot(s, cell) for a, s in zip(anchors, shifts)], axis=0)


def site_centroid_distance_pbc(positions, o_anchors, h_anchors, cell, pbc):
    from ase.geometry import find_mic
    vector = compact_centroid(h_anchors, positions, cell, pbc) - compact_centroid(o_anchors, positions, cell, pbc)
    if np.any(pbc):
        vector, _ = find_mic(vector, cell=cell, pbc=pbc)
    return float(np.linalg.norm(vector))
