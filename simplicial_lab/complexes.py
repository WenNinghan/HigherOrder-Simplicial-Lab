"""Oriented Simplicial Complex Engine from first principles.

Provides canonical representation of 0-, 1-, and 2-simplices, exact signed
boundary operators (B1, B2) satisfying B1 @ B2 == 0 identically, and geometric,
random, and chemical reaction simplicial complex generators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _canonical_edge(u: int, v: int) -> Tuple[int, int]:
    iu, iv = int(u), int(v)
    if iu == iv:
        raise ValueError(f"Self-loop ({iu}, {iv}) is not a valid 1-simplex.")
    return (iu, iv) if iu < iv else (iv, iu)


def _canonical_triangle(u: int, v: int, w: int) -> Tuple[int, int, int]:
    tri = sorted((int(u), int(v), int(w)))
    if tri[0] == tri[1] or tri[1] == tri[2]:
        raise ValueError(f"Degenerate 2-simplex ({u}, {v}, {w}) with repeated vertices.")
    return (tri[0], tri[1], tri[2])


class SimplicialComplex:
    """Oriented 2-dimensional abstract simplicial complex.

    All 1-simplices are stored in canonical orientation ``(i, j)`` with ``i < j``,
    and all 2-simplices in canonical orientation ``(i, j, k)`` with ``i < j < k``.
    Downward simplicial closure is enforced automatically by default.
    """

    def __init__(
        self,
        num_nodes: int,
        edges: Iterable[Sequence[int]] = (),
        triangles: Iterable[Sequence[int]] = (),
        coords: Optional[np.ndarray] = None,
        enforce_closure: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        tri_set = {_canonical_triangle(t[0], t[1], t[2]) for t in triangles}
        edge_set = {_canonical_edge(e[0], e[1]) for e in edges}

        if enforce_closure:
            for i, j, k in tri_set:
                edge_set.add((i, j))
                edge_set.add((i, k))
                edge_set.add((j, k))
        else:
            for i, j, k in tri_set:
                missing = [e for e in ((i, j), (i, k), (j, k)) if e not in edge_set]
                if missing:
                    raise ValueError(
                        f"2-simplex {(i, j, k)} missing boundary edges {missing} with enforce_closure=False."
                    )

        max_idx = -1
        for i, j in edge_set:
            max_idx = max(max_idx, j)
        for i, j, k in tri_set:
            max_idx = max(max_idx, k)

        n0 = max(int(num_nodes), max_idx + 1)
        if n0 <= 0:
            raise ValueError("SimplicialComplex must have at least one 0-simplex.")

        self.num_nodes: int = n0
        self.nodes: np.ndarray = np.arange(n0, dtype=np.int64)

        sorted_edges = sorted(edge_set)
        self.edges: np.ndarray = (
            np.array(sorted_edges, dtype=np.int64)
            if sorted_edges
            else np.empty((0, 2), dtype=np.int64)
        )
        self.edge_to_idx: Dict[Tuple[int, int], int] = {
            e: idx for idx, e in enumerate(sorted_edges)
        }

        sorted_tris = sorted(tri_set)
        self.triangles: np.ndarray = (
            np.array(sorted_tris, dtype=np.int64)
            if sorted_tris
            else np.empty((0, 3), dtype=np.int64)
        )
        self.triangle_to_idx: Dict[Tuple[int, int, int], int] = {
            t: idx for idx, t in enumerate(sorted_tris)
        }

        if coords is not None:
            coords_arr = np.asarray(coords, dtype=np.float64)
            if coords_arr.shape[0] != n0:
                raise ValueError(
                    f"coords first dimension ({coords_arr.shape[0]}) must match num_nodes ({n0})."
                )
            self.coords: Optional[np.ndarray] = coords_arr
        else:
            self.coords = None

        self.metadata: Dict[str, Any] = dict(metadata) if metadata is not None else {}
        self._b1: Optional[np.ndarray] = None
        self._b2: Optional[np.ndarray] = None

    @property
    def num_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def num_triangles(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def shape_tuple(self) -> Tuple[int, int, int]:
        """Return ``(N0, N1, N2)`` simplex counts."""
        return (self.num_nodes, self.num_edges, self.num_triangles)

    @property
    def euler_characteristic(self) -> int:
        """Euler characteristic chi = N0 - N1 + N2."""
        return self.num_nodes - self.num_edges + self.num_triangles

    @property
    def B1(self) -> np.ndarray:
        """Signed node-edge boundary operator B1 of shape ``(N0, N1)``.

        For canonically oriented edge ``e = [i, j]`` with ``i < j``,
        ``partial_1([i, j]) = [j] - [i]``, so ``B1[i, e] = -1`` and ``B1[j, e] = +1``.
        """
        if self._b1 is None:
            b1 = np.zeros((self.num_nodes, self.num_edges), dtype=np.float64)
            if self.num_edges > 0:
                idx = np.arange(self.num_edges)
                b1[self.edges[:, 0], idx] = -1.0
                b1[self.edges[:, 1], idx] = 1.0
            self._b1 = b1
        return self._b1

    @property
    def B2(self) -> np.ndarray:
        """Signed edge-triangle boundary operator B2 of shape ``(N1, N2)``.

        For canonically oriented triangle ``t = [i, j, k]`` with ``i < j < k``,
        ``partial_2([i, j, k]) = [j, k] - [i, k] + [i, j]``.
        """
        if self._b2 is None:
            b2 = np.zeros((self.num_edges, self.num_triangles), dtype=np.float64)
            for p, (i, j, k) in enumerate(self.triangles):
                ii, jj, kk = int(i), int(j), int(k)
                b2[self.edge_to_idx[(ii, jj)], p] = 1.0
                b2[self.edge_to_idx[(ii, kk)], p] = -1.0
                b2[self.edge_to_idx[(jj, kk)], p] = 1.0
            self._b2 = b2
        return self._b2

    def boundary_operator_1(self) -> np.ndarray:
        return self.B1

    def boundary_operator_2(self) -> np.ndarray:
        return self.B2

    def verify_boundary_identity(self) -> float:
        """Return max |B1 @ B2|, which must vanish identically by partial^2 = 0."""
        if self.num_triangles == 0 or self.num_edges == 0:
            return 0.0
        prod = self.B1 @ self.B2
        return float(np.max(np.abs(prod)))

    def mean_degrees(self) -> Tuple[float, float]:
        """Return mean 1-simplex degree <k1> and mean 2-simplex degree <k2>."""
        k1 = (2.0 * self.num_edges) / max(self.num_nodes, 1)
        k2 = (3.0 * self.num_triangles) / max(self.num_nodes, 1)
        return float(k1), float(k2)

    def with_simplices(
        self,
        edges: Optional[Iterable[Sequence[int]]] = None,
        triangles: Optional[Iterable[Sequence[int]]] = None,
    ) -> "SimplicialComplex":
        """Create a derived SimplicialComplex sharing nodes and coordinates."""
        new_edges = self.edges if edges is None else edges
        new_tris = self.triangles if triangles is None else triangles
        return SimplicialComplex(
            num_nodes=self.num_nodes,
            edges=new_edges,
            triangles=new_tris,
            coords=None if self.coords is None else self.coords.copy(),
            enforce_closure=True,
            metadata=dict(self.metadata),
        )

    def __repr__(self) -> str:
        return (
            f"SimplicialComplex(N0={self.num_nodes}, N1={self.num_edges}, "
            f"N2={self.num_triangles}, chi={self.euler_characteristic})"
        )


def cycle_complex(n_nodes: int = 6, filled: bool = False) -> SimplicialComplex:
    """Construct a canonical polygon cycle C_n or fan-filled disk (when filled=True)."""
    if n_nodes < 3:
        raise ValueError("cycle_complex requires n_nodes >= 3.")
    angles = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
    coords = np.column_stack([np.cos(angles), np.sin(angles)])
    edges = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    triangles: List[Tuple[int, int, int]] = []
    if filled:
        for i in range(1, n_nodes - 1):
            triangles.append((0, i, i + 1))
    return SimplicialComplex(
        num_nodes=n_nodes,
        edges=edges,
        triangles=triangles,
        coords=coords,
        enforce_closure=True,
    )


def _extract_boundary_loops(
    edges: np.ndarray,
    triangles: np.ndarray,
    coords: np.ndarray,
    hole_centers: Sequence[Tuple[float, float]],
) -> List[List[int]]:
    """Find ordered cyclic vertex lists of inner hole boundaries."""
    edge_tri_count: Dict[Tuple[int, int], int] = {}
    for u, v in edges:
        edge_tri_count[(int(u), int(v))] = 0
    for i, j, k in triangles:
        ii, jj, kk = int(i), int(j), int(k)
        for e in ((ii, jj), (ii, kk), (jj, kk)):
            edge_tri_count[e] = edge_tri_count.get(e, 0) + 1

    boundary_adj: Dict[int, List[int]] = {}
    for (u, v), cnt in edge_tri_count.items():
        if cnt == 1:
            boundary_adj.setdefault(u, []).append(v)
            boundary_adj.setdefault(v, []).append(u)

    visited_edges = set()
    cycles: List[List[int]] = []
    for start in sorted(boundary_adj.keys()):
        for nxt in boundary_adj[start]:
            e0 = _canonical_edge(start, nxt)
            if e0 in visited_edges:
                continue
            loop = [start]
            prev, curr = start, nxt
            visited_edges.add(e0)
            while curr != start:
                loop.append(curr)
                nbrs = boundary_adj.get(curr, [])
                candidates = [w for w in nbrs if w != prev]
                if not candidates:
                    break
                chosen = candidates[0]
                if len(candidates) > 1:
                    for w in candidates:
                        if _canonical_edge(curr, w) not in visited_edges:
                            chosen = w
                            break
                visited_edges.add(_canonical_edge(curr, chosen))
                prev, curr = curr, chosen
            if curr == start and len(loop) >= 3:
                cycles.append(loop)

    if not cycles:
        return []

    # Exclude the outermost perimeter cycle (largest bounding area) and assign remaining to holes
    def _poly_area(cyc: List[int]) -> float:
        pts = coords[cyc]
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    cycles_sorted = sorted(cycles, key=_poly_area, reverse=True)
    inner_cycles = cycles_sorted[1:]

    matched_loops: List[List[int]] = []
    for cx, cy in hole_centers:
        center = np.array([cx, cy], dtype=np.float64)
        if not inner_cycles:
            break
        best = min(
            inner_cycles,
            key=lambda cyc: float(np.linalg.norm(coords[cyc].mean(axis=0) - center)),
        )
        pts = coords[best] - center
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        # Sort or orient counter-clockwise
        signed_area = 0.5 * np.sum(
            pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]
        )
        ordered = list(best) if signed_area > 0 else list(reversed(best))
        matched_loops.append(ordered)
    return matched_loops


def planar_punctured_complex(
    nx: int = 13,
    ny: int = 9,
    hole_centers: Sequence[Tuple[float, float]] = ((-0.44, 0.0), (0.44, 0.0)),
    hole_radius: float = 0.25,
    seed: int = 42,
) -> SimplicialComplex:
    """Generate a 2D planar simplicial complex with 2 topological holes (beta_0=1, beta_1=2).

    Constructs a structured triangular lattice over ``[-1.05, 1.05] x [-0.70, 0.70]``,
    excises vertices and triangles intersecting two interior circular obstacles, and
    triangulates the complement completely into filled 2-simplices.
    """
    if nx < 9 or ny < 7:
        raise ValueError("planar_punctured_complex requires nx >= 9 and ny >= 7 to separate 2 holes.")

    rng = np.random.default_rng(seed)
    xs = np.linspace(-1.05, 1.05, nx)
    ys = np.linspace(-0.70, 0.70, ny)
    dx = xs[1] - xs[0]

    grid_coords = np.zeros((nx, ny, 2), dtype=np.float64)
    active = np.ones((nx, ny), dtype=bool)
    centers = [np.array(c, dtype=np.float64) for c in hole_centers]

    for ix in range(nx):
        for iy in range(ny):
            shift = 0.28 * dx if (iy % 2 == 1) else -0.28 * dx
            # Keep outer boundary aligned cleanly
            if ix == 0 or ix == nx - 1:
                shift = 0.0
            pos = np.array([xs[ix] + shift, ys[iy]], dtype=np.float64)
            # Tiny deterministic interior jitter for generic edge lengths
            if 0 < ix < nx - 1 and 0 < iy < ny - 1:
                pos += rng.uniform(-0.012, 0.012, size=2)
            grid_coords[ix, iy] = pos

            for c in centers:
                if np.linalg.norm(pos - c) < hole_radius:
                    active[ix, iy] = False
                    break

    # Map active grid vertices to contiguous node IDs 0 .. N0-1
    vid_map = -np.ones((nx, ny), dtype=np.int64)
    coords_list: List[np.ndarray] = []
    for iy in range(ny):
        for ix in range(nx):
            if active[ix, iy]:
                vid_map[ix, iy] = len(coords_list)
                coords_list.append(grid_coords[ix, iy])

    coords = np.array(coords_list, dtype=np.float64)

    def _tri_valid(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
        centroid = (p1 + p2 + p3) / 3.0
        m12 = 0.5 * (p1 + p2)
        m23 = 0.5 * (p2 + p3)
        m31 = 0.5 * (p3 + p1)
        for c in centers:
            if (
                np.linalg.norm(centroid - c) < 0.92 * hole_radius
                or np.linalg.norm(m12 - c) < 0.88 * hole_radius
                or np.linalg.norm(m23 - c) < 0.88 * hole_radius
                or np.linalg.norm(m31 - c) < 0.88 * hole_radius
            ):
                return False
        return True

    triangles: List[Tuple[int, int, int]] = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            v00 = vid_map[ix, iy]
            v10 = vid_map[ix + 1, iy]
            v01 = vid_map[ix, iy + 1]
            v11 = vid_map[ix + 1, iy + 1]

            # Choose shorter diagonal of the cell so triangles are well-shaped
            if iy % 2 == 0:
                cand_tris = [(v00, v10, v11), (v00, v11, v01)]
            else:
                cand_tris = [(v00, v10, v01), (v10, v11, v01)]

            for a, b, d in cand_tris:
                if a >= 0 and b >= 0 and d >= 0:
                    if _tri_valid(coords[a], coords[b], coords[d]):
                        triangles.append(_canonical_triangle(a, b, d))

    sc_temp = SimplicialComplex(
        num_nodes=len(coords),
        edges=(),
        triangles=triangles,
        coords=coords,
        enforce_closure=True,
    )

    hole_loops = _extract_boundary_loops(
        sc_temp.edges, sc_temp.triangles, coords, hole_centers
    )
    hole_caps: List[List[Tuple[int, int, int]]] = []
    for loop in hole_loops:
        cap = [
            _canonical_triangle(loop[0], loop[m], loop[m + 1])
            for m in range(1, len(loop) - 1)
        ]
        hole_caps.append(cap)

    metadata = {
        "name": "planar_punctured_complex",
        "hole_centers": [tuple(map(float, c)) for c in hole_centers],
        "hole_radius": float(hole_radius),
        "hole_boundary_loops": hole_loops,
        "hole_cap_triangles": hole_caps,
    }
    return SimplicialComplex(
        num_nodes=len(coords),
        edges=sc_temp.edges,
        triangles=sc_temp.triangles,
        coords=coords,
        enforce_closure=True,
        metadata=metadata,
    )


def random_simplicial_complex(
    n_nodes: int = 80,
    p1: float = 0.14,
    p2: float = 0.45,
    seed: int = 42,
    independent_triples: bool = False,
) -> SimplicialComplex:
    """Generate a random 2-simplicial complex G_Delta(N, p1, p2).

    Parameters
    ----------
    n_nodes : int
        Number of 0-simplices.
    p1 : float
        Edge probability on pairs ``(i, j)``.
    p2 : float
        If ``independent_triples=False`` (default clique-filtered random complex),
        conditional probability of promoting each 3-clique ``(i, j, k)`` into a filled
        2-simplex. If ``independent_triples=True`` (Iacopini et al. RSC mode),
        unconditional probability over all ``binom(N, 3)`` triples with downward closure.
    seed : int
        Random generator seed.
    """
    if n_nodes < 3:
        raise ValueError("random_simplicial_complex requires n_nodes >= 3.")
    rng = np.random.default_rng(seed)

    # Circular layout with mild radial jitter for optional visualization
    angles = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
    radii = 1.0 + 0.08 * rng.standard_normal(n_nodes)
    coords = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    adj = np.zeros((n_nodes, n_nodes), dtype=bool)
    upper_i, upper_j = np.triu_indices(n_nodes, k=1)
    edge_mask = rng.random(len(upper_i)) < float(p1)
    edges_i = upper_i[edge_mask]
    edges_j = upper_j[edge_mask]
    adj[edges_i, edges_j] = True
    adj[edges_j, edges_i] = True
    edges = list(zip(edges_i.tolist(), edges_j.tolist()))

    triangles: List[Tuple[int, int, int]] = []
    if independent_triples:
        if p2 > 0.0:
            for i in range(n_nodes - 2):
                for j in range(i + 1, n_nodes - 1):
                    r = rng.random(n_nodes - j - 1)
                    ks = np.nonzero(r < p2)[0] + (j + 1)
                    for k in ks:
                        triangles.append((i, j, int(k)))
    else:
        if p2 > 0.0:
            for i in range(n_nodes - 2):
                nbrs_i = np.nonzero(adj[i, i + 1 :])[0] + (i + 1)
                for idx_a in range(len(nbrs_i)):
                    j = int(nbrs_i[idx_a])
                    common = nbrs_i[idx_a + 1 :]
                    common = common[adj[j, common]]
                    if len(common) > 0:
                        keep = common[rng.random(len(common)) < float(p2)]
                        for k in keep:
                            triangles.append((i, j, int(k)))

    return SimplicialComplex(
        num_nodes=n_nodes,
        edges=edges,
        triangles=triangles,
        coords=coords,
        enforce_closure=True,
        metadata={
            "name": "random_simplicial_complex",
            "p1": float(p1),
            "p2": float(p2),
            "independent_triples": bool(independent_triples),
        },
    )


def iacopini_rsc_complex(
    n_nodes: int = 160,
    mean_k1: float = 18.0,
    mean_k2: float = 6.0,
    seed: int = 42,
) -> SimplicialComplex:
    """Construct a Random Simplicial Complex matching target (<k1>, <k2>) per Iacopini et al. (2019)."""
    if n_nodes < 10:
        raise ValueError("iacopini_rsc_complex requires n_nodes >= 10.")
    p2 = (2.0 * mean_k2) / ((n_nodes - 1) * (n_nodes - 2))
    denom = (n_nodes - 1) - 2.0 * mean_k2
    p1 = max(0.0, (mean_k1 - 2.0 * mean_k2) / max(denom, 1.0))
    p1 = min(1.0, p1)
    p2 = min(1.0, max(0.0, p2))
    return random_simplicial_complex(
        n_nodes=n_nodes,
        p1=p1,
        p2=p2,
        seed=seed,
        independent_triples=True,
    )


def trimolecular_reaction_complex(
    n_pollutants: int = 6,
    seed: int = 42,
) -> SimplicialComplex:
    """Generate a higher-order catalytic oxidation complex (O3 + OH* + Pollutant synergy).

    Models radical chain propagation and 3-body peroxone/ozonation synergy:
    - Core oxidant & radical pool: O3 (0), H2O2 (1), OH* (2), HO2* (3), O2*- (4), Mineralized_CO2 (5).
    - Organic pollutant intermediates: P_1 .. P_m.
    - Filled 2-simplices encode synergistic 3-body radical attack triads ``(O3, OH*, P_r)``
      and peroxone radical generation ``(O3, H2O2, OH*)``, while unfilled multi-step degradation
      pathways form non-trivial stoichiometric reaction cycles (beta_1 >= 1).
    """
    if n_pollutants < 2:
        raise ValueError("n_pollutants must be >= 2.")
    rng = np.random.default_rng(seed)

    core_labels = ["O3", "H2O2", "OH*", "HO2*", "O2*-", "CO2_H2O"]
    pollutant_labels = [f"Pollutant_{m + 1}" for m in range(n_pollutants)]
    labels = core_labels + pollutant_labels
    n_total = len(labels)

    coords = np.zeros((n_total, 2), dtype=np.float64)
    core_angles = np.linspace(0.0, 2.0 * np.pi, len(core_labels), endpoint=False)
    coords[: len(core_labels)] = np.column_stack(
        [0.45 * np.cos(core_angles), 0.45 * np.sin(core_angles)]
    )
    pol_angles = np.linspace(0.0, 2.0 * np.pi, n_pollutants, endpoint=False) + 0.25
    coords[len(core_labels) :] = np.column_stack(
        [0.95 * np.cos(pol_angles), 0.95 * np.sin(pol_angles)]
    )
    coords += 0.015 * rng.standard_normal(coords.shape)

    # Unfilled radical cycle O3 - HO2* - O2*- - H2O2 - O3 plus filled peroxone triad
    edges: List[Tuple[int, int]] = [
        (0, 3),
        (3, 4),
        (4, 1),
        (1, 0),  # Radical propagation cycle (unfilled 4-cycle -> harmonic mode)
        (2, 5),
        (0, 5),  # Mineralization channels
    ]
    triangles: List[Tuple[int, int, int]] = [
        (0, 1, 2),  # Peroxone initiation triad: O3 + H2O2 -> OH*
        (0, 2, 3),  # Radical chain triad: O3 + OH* <-> HO2*
    ]

    offset = len(core_labels)
    for m in range(n_pollutants):
        p_node = offset + m
        next_p = offset + ((m + 1) % n_pollutants)
        # 3-body synergistic attack: O3 (0) + OH* (2) + Pollutant_m
        triangles.append((0, 2, p_node))
        # Pairwise oxidation to mineralized sink CO2_H2O (5) and cyclic isomerization
        edges.append((p_node, 5))
        edges.append((p_node, next_p))

    return SimplicialComplex(
        num_nodes=n_total,
        edges=edges,
        triangles=triangles,
        coords=coords,
        enforce_closure=True,
        metadata={
            "name": "trimolecular_reaction_complex",
            "node_labels": labels,
            "core_species": core_labels,
            "pollutant_species": pollutant_labels,
        },
    )
