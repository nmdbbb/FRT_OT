"""
FRT module — Global tree construction, Tree-Wasserstein, and full pipeline.

Highlights
----------
- Local normalized time index per series.
- Auto time weighting (median ratio heuristic) with configurable factor.
- Auto depth_shift based on data spread (optional).
- Deterministic-friendly 2-HST FRT builder.
- Closed-form Tree-Wasserstein computation with local-subtree pruning.
- Unified pipeline `frt_knn()` for training/testing distance matrices.
"""
from __future__ import annotations
import time
import numpy as np
from tqdm.auto import tqdm
from typing import List, Tuple, Optional, Dict


# ============================================================
# 1) Time & stacking utilities (with membership mapping)
# ============================================================

def add_time_index(series: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Append a normalized time column [0..1] to a time series."""
    S = np.asarray(series, dtype=float)
    m = S.shape[0]
    if m == 0:
        return np.zeros((0, S.shape[1] + 1))
    t = np.arange(m, dtype=float)
    if normalize and m > 1:
        t /= (m - 1.0)
    return np.hstack([t.reshape(-1, 1), S])


def stack_union_points_with_time(X_train_series: List[np.ndarray],
                                 X_test_series: List[np.ndarray],
                                 normalize_time: bool = True):
    """
    Stack all series (train + test) with local time indices and return:
    - coords_all        : (N, d) union of all points
    - seq_membership    : (N,) series id each point belongs to (0..n_tr+n_te-1)
    - seq_boundaries    : list of (start, end) for each series in concatenation order
    """
    all_chunks, seq_membership, seq_boundaries = [], [], []
    all_series = list(X_train_series) + list(X_test_series)
    offset = 0
    for sid, s in enumerate(all_series):
        arr = add_time_index(s, normalize=normalize_time)
        all_chunks.append(arr)
        seq_membership.extend([sid] * len(arr))
        seq_boundaries.append((offset, offset + len(arr)))
        offset += len(arr)
    coords_all = np.vstack(all_chunks) if all_chunks else np.zeros((0, 0))
    return coords_all, np.array(seq_membership, np.int32), seq_boundaries


# ============================================================
# 2) Metric utilities
# ============================================================

def _pairwise_weighted_euclidean(A: np.ndarray, B: np.ndarray,
                                 w: Optional[np.ndarray] = None,
                                 block_size: int = 444) -> np.ndarray:
    """Compute weighted Euclidean distances blockwise for memory efficiency."""
    A, B = np.asarray(A, np.float32), np.asarray(B, np.float32)
    if w is not None:
        ws = np.sqrt(np.asarray(w, np.float32))
        A, B = A * ws, B * ws
    n, m = A.shape[0], B.shape[0]
    D = np.zeros((n, m), np.float32)
    BB = np.sum(B * B, axis=1, keepdims=True).T
    B_T = B.T
    for i in range(0, n, block_size):
        Ai = A[i:i + block_size]
        AA = np.sum(Ai * Ai, axis=1, keepdims=True)
        D_block = AA + BB - 2.0 * (Ai @ B_T)
        np.maximum(D_block, 0.0, out=D_block)
        D[i:i + block_size] = np.sqrt(D_block)
    return D


def estimate_time_weight(coords: np.ndarray, time_col: int = 0,
                         feature_cols: Optional[List[int]] = None,
                         seed: int = 42,
                         time_factor: float = 64.0) -> float:
    """Estimate time-weight using median ratio heuristic, scaled by `time_factor` (default 64)."""
    X = np.asarray(coords, float)
    n, d = X.shape
    if n < 2:
        return 1.0
    if feature_cols is None:
        feature_cols = [j for j in range(d) if j != time_col]
    t, f = X[:, time_col], X[:, feature_cols]
    rng = np.random.default_rng(seed)
    m = int(min(max(0.44 * n, 2000), n))
    ii, jj = rng.integers(0, n, m), rng.integers(0, n, m)
    dt, df = np.abs(t[ii] - t[jj]), np.linalg.norm(f[ii] - f[jj], axis=1)
    med_dt, med_df = np.median(dt), np.median(df)
    if med_dt <= 1e-12 or med_df <= 1e-12:
        return 1.0
    return float(time_factor) * ((med_df / med_dt) ** 2)


def auto_depth_shift(coords: np.ndarray, wdim: np.ndarray) -> int:
    """
    Automatically determine suitable depth_shift based on data spread:
    depth_shift = clip(round(3 + log2(Δ/μ)), 4, 10),
    where Δ is diameter and μ is mean positive pairwise distance under wdim.
    """
    if coords.size == 0:
        return 8
    D_full = _pairwise_weighted_euclidean(coords, coords, w=wdim)
    if D_full.size == 0:
        return 8
    np.fill_diagonal(D_full, 0.0)
    Δ = float(np.max(D_full))
    pos = D_full[D_full > 0]
    μ = float(np.mean(pos)) if pos.size else 0.0
    if μ <= 1e-12 or Δ <= 1e-12:
        return 8
    r = Δ / μ
    shift = int(np.clip(round(3 + np.log2(r)), 4, 10))
    return shift


# ============================================================
# 3) 2-HST (FRT) construction
# ============================================================

class HST:
    """Compact representation of a 2-HST tree with per-series leaf buckets."""
    __slots__ = ("parent_w", "leaf_of_point", "leaves_per_series")

    def __init__(self, parent: np.ndarray, w: np.ndarray, leaf_of_point: np.ndarray,
                 seq_membership: Optional[np.ndarray] = None):
        self.parent_w = np.rec.fromarrays(
            [np.asarray(parent, np.int32), np.asarray(w, np.float32)],
            names=("parent", "w")
        )
        self.leaf_of_point = np.asarray(leaf_of_point, np.int32)

        # Group leaves by series id for fast local-subtree access.
        self.leaves_per_series: Dict[int, np.ndarray] = {}
        if seq_membership is not None and len(seq_membership) == len(leaf_of_point):
            # Each point has a unique leaf node id.
            from collections import defaultdict
            buckets = defaultdict(list)
            for pt_idx, leaf_node in enumerate(leaf_of_point):
                sid = int(seq_membership[pt_idx])
                buckets[sid].append(int(leaf_node))
            self.leaves_per_series = {sid: np.array(leaves, np.int32)
                                      for sid, leaves in buckets.items()}

    @property
    def parent(self): return self.parent_w["parent"]

    @property
    def w(self): return self.parent_w["w"]

    @property
    def n_nodes(self): return self.parent_w.shape[0]


def _frt_hst_from_metric_fast(D: np.ndarray, random_state=None,
                              alpha=None, level_edge_shift=0) -> HST:
    """Build a deterministic-friendly 2-HST (FRT tree) from a metric matrix."""
    rng = np.random.default_rng(random_state)
    n = D.shape[0]
    diam = np.max(D) if n else 0.0
    if n == 0 or diam <= 0:
        return HST(np.array([-1]), np.array([0.0]), np.zeros(0, int))
    L = int(np.ceil(np.log2(diam)))
    alpha = rng.uniform(1.0, 2.0) if alpha is None else float(alpha)
    order = rng.permutation(n)

    parent, w, levels = [-1], [0.0], [L + 1]
    root, node_id = 0, 1
    current_parent = np.full(n, root, int)
    leaf_nodes = np.full(n, -1, int)

    radii = [alpha * (2.0 ** k) for k in range(L, -1, -1)]
    for ell, rad in enumerate(radii):
        k = L - ell
        assigned = np.full(n, -1, int)
        cid = 0
        for i in order:
            if assigned[i] != -1:
                continue
            in_ball = np.where(D[i] <= rad)[0]
            unassigned = in_ball[assigned[in_ball] == -1]
            assigned[unassigned] = cid
            parent.append(int(current_parent[i]))
            w.append(float(2.0 ** (k + level_edge_shift)))
            levels.append(k)
            node_here = node_id
            node_id += 1
            current_parent[assigned == cid] = node_here
            cid += 1
        if k == 0:
            for p in range(n):
                parent.append(int(current_parent[p]))
                w.append(0.0)
                levels.append(-1)
                leaf_nodes[p] = node_id
                node_id += 1
    return HST(np.asarray(parent, np.int32), np.asarray(w, np.float32), leaf_nodes)


# ============================================================
# 4) Tree-Wasserstein: full-tree and local-subtree
# ============================================================

def tw_closed_form_on_tree_full(tree: HST, leaf_weights_dense_points: np.ndarray) -> float:
    """
    Closed-form W1 on full tree.
    leaf_weights_dense_points has length N_points (coords_all rows).
    """
    parent, w = tree.parent, tree.w
    n_nodes = parent.shape[0]
    n_points = leaf_weights_dense_points.shape[0]
    first_leaf = n_nodes - n_points
    mass = np.zeros(n_nodes, dtype=float)
    mass[first_leaf:] = leaf_weights_dense_points
    cost = 0.0
    for v in range(n_nodes - 1, 0, -1):
        val = mass[v]
        if val != 0.0:
            cost += abs(val) * w[v]
            p = parent[v]
            if p >= 0:
                mass[p] += val
    return float(cost)


def tw_closed_form_on_tree_local(tree: HST,
                                 leaf_nodes: np.ndarray,
                                 leaf_weights: np.ndarray) -> float:
    """
    Closed-form W1 but restricted to the induced subtree formed by `leaf_nodes`.
    - leaf_nodes: array of leaf node ids (node ids in HST, not point indices)
    - leaf_weights: weights for corresponding leaves (same length)
    Only nodes on paths from these leaves to the root are traversed.
    """
    parent, w = tree.parent, tree.w
    if leaf_nodes.size == 0:
        return 0.0

    # Collect induced set of nodes (union of paths to root).
    # Also build a topological order bottom-up for those nodes.
    mark = {}
    nodes = []
    for leaf in leaf_nodes:
        v = int(leaf)
        while v not in mark:
            mark[v] = True
            nodes.append(v)
            p = int(parent[v])
            if p < 0:  # reached root
                break
            v = p

    # Bottom-up traversal: nodes are appended leaf->root, but may interleave.
    # Sort descending so that children processed before parents.
    nodes = np.array(nodes, np.int32)
    # A simple heuristic: process in descending node id often respects build order
    # (leaves have larger ids). For correctness, we still do the parent-add after.
    nodes.sort()
    nodes = nodes[::-1]

    # Build sparse mass only on induced nodes.
    mass = {}
    # Map provided leaf node weights.
    for ln, wt in zip(leaf_nodes, leaf_weights):
        if wt == 0.0:
            continue
        mass[int(ln)] = mass.get(int(ln), 0.0) + float(wt)

    cost = 0.0
    for v in nodes:
        val = mass.get(int(v), 0.0)
        if val != 0.0:
            cost += abs(val) * float(w[v])
            p = int(parent[v])
            if p >= 0:
                mass[p] = mass.get(p, 0.0) + val
    return float(cost)


# ============================================================
# 5) Global FRT builder (+ membership in meta & tree)
# ============================================================

class GlobalMeta:
    """Metadata for global FRT trees and series mapping."""
    def __init__(self, coords_all, time_weight, wdim, time_col, feature_cols, w_avg,
                 seq_membership=None, seq_boundaries=None,
                 time_factor: float = 64.0, depth_shift: int = 8):
        self.coords_all = coords_all
        self.time_weight = time_weight
        self.wdim = wdim
        self.time_col = time_col
        self.feature_cols = feature_cols
        self.w_avg = w_avg
        self.seq_membership = np.asarray(seq_membership) if seq_membership is not None else None
        self.seq_boundaries = seq_boundaries or []
        self.time_factor = float(time_factor)
        self.depth_shift = int(depth_shift)
        self.build_tree_sec = 0.0
        self.distance_calc_sec = 0.0


def build_global_frt_trees_from_coords(coords_all: np.ndarray,
                                       n_trees: int = 1,
                                       time_weight: float | str = "auto",
                                       time_col: int = 0,
                                       feature_cols: Optional[List[int]] = None,
                                       random_state=None,
                                       alpha=None,
                                       level_edge_shift: int = 0,
                                       depth_shift: int | str = 8,
                                       time_factor: float = 64.0,
                                       seq_membership: Optional[np.ndarray] = None,
                                       ) -> Tuple[List[HST], GlobalMeta]:
    """
    Build multiple global FRT trees from unified coordinates.
    If seq_membership is provided, the resulting HSTs will store per-series leaf buckets.
    depth_shift can be an int or "auto".
    """
    X = np.asarray(coords_all, float)
    N, d = X.shape
    if feature_cols is None:
        feature_cols = [j for j in range(d) if j != time_col]

    # --- time weight ---
    if time_weight == "auto":
        tw = estimate_time_weight(X, time_col=time_col, feature_cols=feature_cols,
                                  seed=42, time_factor=time_factor)
    else:
        tw = float(time_weight)

    wdim = np.ones(d, float)
    wdim[time_col] = tw

    # --- auto depth shift if requested ---
    if isinstance(depth_shift, str) and depth_shift.lower() == "auto":
        ds_val = auto_depth_shift(X, wdim)
    else:
        ds_val = int(depth_shift)

    # Apply depth scaling
    wdim *= 2 ** ds_val

    # Build metric among all points (under final weights)
    D = _pairwise_weighted_euclidean(X, X, w=wdim)

    # Build trees
    trees: List[HST] = []
    base_seed = None if random_state is None else int(random_state)
    for t in range(n_trees):
        rs = None if base_seed is None else base_seed + t
        T = _frt_hst_from_metric_fast(D, random_state=rs, alpha=alpha, level_edge_shift=level_edge_shift)
        # Attach membership into tree for fast per-series leaf access
        if seq_membership is not None and seq_membership.size == N:
            T = HST(T.parent, T.w, T.leaf_of_point, seq_membership=seq_membership)
        trees.append(T)

    ws = [np.mean(T.w[1:]) if T.w.size > 1 else 0.0 for T in trees]
    meta = GlobalMeta(X, tw, wdim, time_col, feature_cols, float(np.mean(ws)),
                      seq_membership=seq_membership,
                      seq_boundaries=None,  # (filled by caller that knows boundaries)
                      time_factor=time_factor, depth_shift=ds_val)
    return trees, meta


# ============================================================
# 6) Series measure & distance matrices with local-subtree
# ============================================================

def series_indices_from_meta(meta: GlobalMeta, sid: int) -> np.ndarray:
    """
    Return point indices in coords_all that belong to series `sid`.
    Requires meta.seq_membership to be available.
    """
    if meta.seq_membership is None:
        return np.array([], np.int32)
    return np.where(meta.seq_membership == int(sid))[0].astype(np.int32)


def measure_from_series_selfpoints(meta: GlobalMeta, sid: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a probability measure over the series' own points (no KNN):
    - idx_pts: indices into coords_all
    - probs  : uniform probabilities over those points
    """
    idx_pts = series_indices_from_meta(meta, sid)
    if idx_pts.size == 0:
        return idx_pts, np.zeros(0, float)
    probs = np.full(idx_pts.size, 1.0 / idx_pts.size, float)
    return idx_pts, probs


def tw_series_pair_local(tree: HST,
                         meta: GlobalMeta,
                         sid_a: int, sid_b: int,
                         idx_a: np.ndarray, p_a: np.ndarray,
                         idx_b: np.ndarray, p_b: np.ndarray) -> float:
    """
    Compute TWD between two series using only the induced subtree of their support.
    - Convert point indices -> leaf node ids via tree.leaf_of_point
    - Build weights on those leaves (+ for A, - for B)
    - Run local closed-form on induced subtree
    """
    if idx_a.size == 0 and idx_b.size == 0:
        return 0.0
    # Map point indices to leaf node ids
    leaf_a = tree.leaf_of_point[idx_a] if idx_a.size else np.zeros(0, np.int32)
    leaf_b = tree.leaf_of_point[idx_b] if idx_b.size else np.zeros(0, np.int32)
    # Merge supports
    leaves = np.concatenate([leaf_a, leaf_b])
    if leaves.size == 0:
        return 0.0
    # Build aligned weights per leaf
    # We'll aggregate duplicates (if any) by sorting+grouping.
    weights = np.concatenate([p_a, -p_b]).astype(float)
    order = np.argsort(leaves, kind="mergesort")
    leaves_sorted = leaves[order]
    weights_sorted = weights[order]

    # Reduce by key (leaf id)
    uniq_leaves = []
    uniq_weights = []
    prev = None
    acc = 0.0
    for lid, wt in zip(leaves_sorted, weights_sorted):
        if prev is None:
            prev = int(lid)
            acc = float(wt)
        elif int(lid) == prev:
            acc += float(wt)
        else:
            uniq_leaves.append(prev)
            uniq_weights.append(acc)
            prev = int(lid)
            acc = float(wt)
    if prev is not None:
        uniq_leaves.append(prev)
        uniq_weights.append(acc)

    leaves_arr = np.array(uniq_leaves, np.int32)
    weights_arr = np.array(uniq_weights, float)
    if leaves_arr.size == 0:
        return 0.0

    return tw_closed_form_on_tree_local(tree, leaves_arr, weights_arr)


def compute_distance_matrices_global_frt(X_train_series: List[np.ndarray],
                                         X_test_series: List[np.ndarray],
                                         trees: List[HST], meta: GlobalMeta,
                                         desc_prefix: str = "FRT"):
    """
    Compute pairwise TW distances (train–train and test–train) using local-subtree pruning.
    Mapping uses the series' own points (no KNN).
    """
    n_tr, n_te = len(X_train_series), len(X_test_series)

    # Pre-build measures for all series (train + test) using own points (uniform mass).
    all_idx: List[Tuple[np.ndarray, np.ndarray]] = []
    for sid in tqdm(range(n_tr + n_te), desc=f"{desc_prefix} measures"):
        all_idx.append(measure_from_series_selfpoints(meta, sid))

    # Train–Train
    D_tr = np.zeros((n_tr, n_tr), float)
    for i in tqdm(range(n_tr), desc=f"{desc_prefix} D_tr"):
        idx_i, p_i = all_idx[i]
        for j in range(i + 1, n_tr):
            idx_j, p_j = all_idx[j]
            vals = [tw_series_pair_local(T, meta, i, j, idx_i, p_i, idx_j, p_j) for T in trees]
            D_tr[i, j] = D_tr[j, i] = float(np.mean(vals) if len(vals) else 0.0)

    # Test–Train
    D_te = np.zeros((n_te, n_tr), float)
    for p in tqdm(range(n_te), desc=f"{desc_prefix} D_te"):
        sid_p = n_tr + p
        idx_p, p_p = all_idx[sid_p]
        for j in range(n_tr):
            idx_j, p_j = all_idx[j]
            vals = [tw_series_pair_local(T, meta, sid_p, j, idx_p, p_p, idx_j, p_j) for T in trees]
            D_te[p, j] = float(np.mean(vals) if len(vals) else 0.0)

    return D_tr, D_te


# ============================================================
# 7) Unified FRT pipeline (public entry)
# ============================================================

def frt_knn(X_train, X_test, n_trees=16, time_weight="auto",
            random_state=123, level_edge_shift=1, n_jobs: int = -1,
            depth_shift: int | str = "auto", time_factor: float = 64.0):
    """
    Complete FRT pipeline: build trees + compute normalized distance matrices.

    Changes vs. old version:
    - Uses own-points mapping (no KNN) for measures.
    - Uses local-subtree TW to avoid traversing the full tree.
    - Supports auto depth_shift (default "auto") and configurable time_factor (default 64).
    """
    # 1) Stack data with membership mapping
    coords_all, seq_membership, seq_boundaries = stack_union_points_with_time(
        X_train, X_test, normalize_time=True
    )

    # 2) Build trees (attach membership)
    t0 = time.perf_counter()
    trees, meta = build_global_frt_trees_from_coords(
        coords_all,
        n_trees=n_trees,
        time_weight=time_weight,
        random_state=random_state,
        level_edge_shift=level_edge_shift,
        depth_shift=depth_shift,
        time_factor=time_factor,
        seq_membership=seq_membership,
    )
    meta.build_tree_sec = time.perf_counter() - t0
    meta.seq_boundaries = seq_boundaries  # record boundaries (useful for debugging/logs)

    print(f"[FRT] time_weight={meta.time_weight:.6f} | depth_shift={meta.depth_shift} | "
          f"w_avg={meta.w_avg:.6f} | points={coords_all.shape[0]}")

    # 3) Distances with local-subtree pruning
    t1 = time.perf_counter()
    D_tr, D_te = compute_distance_matrices_global_frt(X_train, X_test, trees, meta)
    meta.distance_calc_sec = time.perf_counter() - t1

    # 4) Normalize by max(D_tr) for stability
    maxv = float(np.max(D_tr)) if D_tr.size else 0.0
    if maxv > 0:
        D_tr /= maxv
        D_te /= maxv

    return D_tr, D_te, meta
