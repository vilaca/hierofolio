import cvxpy as cp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.covariance import LedoitWolf
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
import warnings
import hashlib


def _project_to_psd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project a covariance matrix to positive semi-definite via eigenvalue clipping."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


class RiskModel(ABC):
    """Abstract base class for risk models consumed by optimizers."""
    
    @abstractmethod
    def covariance(self) -> pd.DataFrame:
        """Return shrunk covariance matrix with asset names."""
        pass
    
    @abstractmethod
    def leaf_clusters(self) -> Dict[int, List[str]]:
        """Return leaf nodes (single assets)."""
        pass
    
    @abstractmethod
    def internal_clusters(self) -> Dict[int, List[str]]:
        """Return internal nodes (clusters of 2+ assets)."""
        pass
    
    @abstractmethod
    def cut(self, level: int) -> Dict[int, List[str]]:
        """
        Return a non-overlapping partition at a given tree level.
        
        For fixed mode: level corresponds to number of clusters.
        For full mode: level corresponds to depth from root.
        """
        pass


@dataclass(frozen=True, eq=False)
class HRPRiskModel(RiskModel):
    """
    Immutable Hierarchical Risk Parity Risk Model.
    
    All computations are precomputed at initialization. The object is designed
    to be immutable and cacheable for parallel backtesting.
    
    Features:
    - Stable correlation and distance matrices
    - Flexible shrinkage (Ledoit-Wolf, constant-correlation, identity)
    - Fixed or full dendrogram cluster modes
    - DataFrame-based API with asset name preservation
    - Tree structure traversal for recursive HRP
    - Clear separation of leaves, internal nodes, and root
    - Cut-level partition for constraints and reporting
    """
    
    # --- Configuration (immutable via dataclass frozen=True) ---
    returns: pd.DataFrame
    shrinkage_method: str = "ledoit_wolf"
    shrinkage_intensity: float = 0.5
    cluster_mode: str = "fixed"  # "fixed" or "full"
    n_clusters: Optional[int] = None
    linkage_method: str = "ward"
    
    # --- Computed attributes (set in __post_init__) ---
    _assets: List[str] = field(init=False, default_factory=list)
    _n_assets: int = field(init=False, default=0)
    _returns_values: np.ndarray = field(init=False, default=None)
    _corr_matrix: pd.DataFrame = field(init=False, default=None)
    _distance_matrix: pd.DataFrame = field(init=False, default=None)
    _linkage_matrix: np.ndarray = field(init=False, default=None)
    _leaf_clusters: Dict[int, List[str]] = field(init=False, default_factory=dict)
    _internal_clusters: Dict[int, List[str]] = field(init=False, default_factory=dict)
    _all_clusters: Dict[int, List[str]] = field(init=False, default_factory=dict)
    _shrunk_cov: pd.DataFrame = field(init=False, default=None)
    _root_node_id: int = field(init=False, default=0)
    _children_map: Dict[int, Tuple[int, int]] = field(init=False, default_factory=dict)
    _node_assets_map: Dict[int, List[str]] = field(init=False, default_factory=dict)
    _parent_map: Dict[int, int] = field(init=False, default_factory=dict)
    _levels: Dict[int, int] = field(init=False, default_factory=dict)
    _leaf_order: List[str] = field(init=False, default_factory=list)
    
    def __post_init__(self):
        """Initialize immutable computed attributes."""
        # --- Validation ---
        if self.returns.isna().any().any():
            raise ValueError("Returns contain NaNs. Please clean data first.")
        
        if self.returns.shape[1] < 2:
            raise ValueError(f"Need at least 2 assets, got {self.returns.shape[1]}")
        
        if self.shrinkage_method not in ["ledoit_wolf", "constant_correlation", "identity"]:
            raise ValueError(f"Unknown shrinkage_method: {self.shrinkage_method}")
        
        if self.cluster_mode not in ["fixed", "full"]:
            raise ValueError(f"Unknown cluster_mode: {self.cluster_mode}")
        
        if self.linkage_method not in ["ward", "average", "complete", "single"]:
            raise ValueError(f"Unknown linkage_method: {self.linkage_method}")
        
        # --- Cache for performance ---
        object.__setattr__(self, "_assets", self.returns.columns.tolist())
        object.__setattr__(self, "_n_assets", len(self._assets))
        object.__setattr__(self, "_returns_values", self.returns.values)
        
        # --- Step 1: Stable correlation and distance (DataFrames) ---
        corr_matrix = self.returns.corr().clip(-1, 1)
        dist_values = np.sqrt(np.maximum(0, 0.5 * (1 - corr_matrix.values)))
        distance_matrix = pd.DataFrame(
            dist_values,
            index=self._assets,
            columns=self._assets
        )
        
        object.__setattr__(self, "_corr_matrix", corr_matrix)
        object.__setattr__(self, "_distance_matrix", distance_matrix)
        
        # --- Step 2: Hierarchical clustering (cache full linkage) ---
        condensed = squareform(distance_matrix.values, checks=False)
        linkage_matrix = linkage(condensed, method=self.linkage_method)
        object.__setattr__(self, "_linkage_matrix", linkage_matrix)
        
        # --- Step 3: Extract clusters based on mode ---
        if self.cluster_mode == "fixed":
            n_clusters = self.n_clusters or int(np.ceil(np.sqrt(self._n_assets)))
            leaf_clusters, internal_clusters, all_clusters, root_id = self._extract_fixed_clusters(n_clusters)
        else:  # "full" mode
            leaf_clusters, internal_clusters, all_clusters, root_id, children_map, node_assets_map, parent_map, levels = (
                self._extract_full_clusters()
            )
            object.__setattr__(self, "_children_map", children_map)
            object.__setattr__(self, "_node_assets_map", node_assets_map)
            object.__setattr__(self, "_parent_map", parent_map)
            object.__setattr__(self, "_levels", levels)
        
        object.__setattr__(self, "_leaf_clusters", leaf_clusters)
        object.__setattr__(self, "_internal_clusters", internal_clusters)
        object.__setattr__(self, "_all_clusters", all_clusters)
        object.__setattr__(self, "_root_node_id", root_id)
        
        # --- Step 4: Get leaf order for dendrogram mapping ---
        d = dendrogram(linkage_matrix, labels=self._assets, no_plot=True)
        object.__setattr__(self, "_leaf_order", d["ivl"])
        
        # --- Step 5: Shrunk covariance (DataFrame) ---
        sample_cov = self.returns.cov().values
        shrunk_values = self._apply_shrinkage(sample_cov)
        shrunk_cov = pd.DataFrame(
            shrunk_values,
            index=self._assets,
            columns=self._assets
        )
        object.__setattr__(self, "_shrunk_cov", shrunk_cov)
    
    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------
    
    def _extract_fixed_clusters(self, n_clusters: int) -> Tuple[Dict[int, List[str]], Dict[int, List[str]], Dict[int, List[str]], int]:
        """Extract fixed number of clusters."""
        cluster_labels = fcluster(self._linkage_matrix, n_clusters, criterion="maxclust")
        
        internal_clusters = {}
        all_clusters = {}

        # Offset cluster IDs past the leaf ID range (0..n-1) so leaf and
        # cluster node IDs never collide, mirroring the full-mode convention.
        for cluster_id in set(cluster_labels):
            indices = np.where(cluster_labels == cluster_id)[0].tolist()
            asset_names = [self._assets[i] for i in indices]
            node_id = int(cluster_id) + self._n_assets
            all_clusters[node_id] = asset_names.copy()
            if len(asset_names) > 1:
                internal_clusters[node_id] = asset_names.copy()

        # Leaf clusters: each asset is its own cluster
        leaf_clusters = {i: [asset] for i, asset in enumerate(self._assets)}

        # Root cluster: the one containing all assets
        root_id = None
        for node_id, assets in all_clusters.items():
            if set(assets) == set(self._assets):
                root_id = node_id
                break

        if root_id is None:
            # Synthetic root
            root_id = max(all_clusters.keys()) + 1 if all_clusters else self._n_assets
            all_clusters[root_id] = self._assets.copy()
            internal_clusters[root_id] = self._assets.copy()
        
        return leaf_clusters, internal_clusters, all_clusters, root_id
    
    def _extract_full_clusters(self) -> Tuple[
        Dict[int, List[str]],  # leaf_clusters
        Dict[int, List[str]],  # internal_clusters
        Dict[int, List[str]],  # all_clusters
        int,  # root_id
        Dict[int, Tuple[int, int]],  # children_map
        Dict[int, List[str]],  # node_assets_map
        Dict[int, int],  # parent_map
        Dict[int, int]  # levels
    ]:
        """Extract full dendrogram as a tree structure."""
        n = self._n_assets
        linkage = self._linkage_matrix
        n_nodes = len(linkage)
        
        # Initialize
        all_clusters = {}
        node_assets_map = {}
        children_map = {}
        parent_map = {}
        levels = {}
        
        # Leaf nodes: IDs 0 to n-1
        leaf_clusters = {}
        for i, asset in enumerate(self._assets):
            leaf_clusters[i] = [asset]
            all_clusters[i] = [asset]
            node_assets_map[i] = [asset]
            levels[i] = 0
        
        # Internal nodes: IDs n to n + n_nodes - 1
        current_id = n
        for i, (left, right, dist, count) in enumerate(linkage):
            left = int(left)
            right = int(right)
            
            # Get assets under left and right
            left_assets = node_assets_map[left]
            right_assets = node_assets_map[right]
            merged_assets = left_assets + right_assets
            
            all_clusters[current_id] = merged_assets.copy()
            node_assets_map[current_id] = merged_assets.copy()
            children_map[current_id] = (left, right)
            parent_map[left] = current_id
            parent_map[right] = current_id
            levels[current_id] = max(levels.get(left, 0), levels.get(right, 0)) + 1
            
            current_id += 1
        
        # Root is the last internal node
        root_id = current_id - 1
        
        # Validate root contains all assets
        if root_id not in all_clusters or set(all_clusters[root_id]) != set(self._assets):
            raise RuntimeError(
                f"Linkage root does not contain all assets. "
                f"Expected {len(self._assets)} assets, got {len(all_clusters.get(root_id, []))}."
            )
        
        # Internal clusters: all nodes except leaves
        internal_clusters = {
            node_id: assets.copy()
            for node_id, assets in all_clusters.items()
            if len(assets) > 1
        }
        
        return leaf_clusters, internal_clusters, all_clusters, root_id, children_map, node_assets_map, parent_map, levels
    
    def _apply_shrinkage(self, sample_cov: np.ndarray) -> np.ndarray:
        """Apply covariance shrinkage with multiple methods."""
        if self.shrinkage_method == "ledoit_wolf":
            return LedoitWolf().fit(self._returns_values).covariance_
        
        elif self.shrinkage_method == "constant_correlation":
            return self._constant_correlation_target(sample_cov)
        
        elif self.shrinkage_method == "identity":
            target = np.eye(self._n_assets) * np.mean(np.diag(sample_cov))
            return (1 - self.shrinkage_intensity) * sample_cov + self.shrinkage_intensity * target
        
        else:
            raise ValueError(f"Unknown shrinkage method: {self.shrinkage_method}")
    
    def _constant_correlation_target(self, sample_cov: np.ndarray) -> np.ndarray:
        """Build constant correlation target matrix."""
        std = np.sqrt(np.diag(sample_cov))
        corr = np.corrcoef(self._returns_values.T)
        corr = np.clip(corr, -1, 1)
        
        triu_indices = np.triu_indices_from(corr, k=1)
        avg_corr = np.mean(corr[triu_indices])
        
        target = np.outer(std, std) * avg_corr
        np.fill_diagonal(target, np.diag(sample_cov))
        
        return (1 - self.shrinkage_intensity) * sample_cov + self.shrinkage_intensity * target
    
    def __hash__(self) -> int:
        """Custom hash for memoization and caching."""
        try:
            returns_hash = pd.util.hash_pandas_object(self.returns, index=True).sum()
        except Exception:
            returns_hash = hashlib.sha256(
                self.returns.values.tobytes()
            ).hexdigest()
        
        return hash((
            tuple(self._assets),
            self.shrinkage_method,
            self.shrinkage_intensity,
            self.cluster_mode,
            self.linkage_method,
            returns_hash
        ))

    def __eq__(self, other: object) -> bool:
        """Value equality on configuration (mirrors __hash__).

        Defined explicitly because the auto-generated dataclass __eq__ would
        compare DataFrame/ndarray fields and raise on ambiguous truth values.
        """
        if not isinstance(other, HRPRiskModel):
            return NotImplemented
        return (
            self.shrinkage_method == other.shrinkage_method
            and self.shrinkage_intensity == other.shrinkage_intensity
            and self.cluster_mode == other.cluster_mode
            and self.linkage_method == other.linkage_method
            and self.n_clusters == other.n_clusters
            and self.returns.equals(other.returns)
        )
    
    # ------------------------------------------------------------------------
    # Public API - Core Methods
    # ------------------------------------------------------------------------
    
    def covariance(self) -> pd.DataFrame:
        """Return shrunk covariance matrix with asset names."""
        return self._shrunk_cov
    
    def correlation(self) -> pd.DataFrame:
        """Return correlation matrix with asset names."""
        return self._corr_matrix
    
    def distance(self) -> pd.DataFrame:
        """Return distance matrix with asset names."""
        return self._distance_matrix
    
    def leaf_clusters(self) -> Dict[int, List[str]]:
        """Return leaf nodes (single assets)."""
        return self._leaf_clusters.copy()
    
    def internal_clusters(self) -> Dict[int, List[str]]:
        """Return internal nodes (clusters of 2+ assets)."""
        return self._internal_clusters.copy()
    
    def all_clusters(self) -> Dict[int, List[str]]:
        """Return all nodes (leaves + internal)."""
        return self._all_clusters.copy()
    
    def cut(self, level: int) -> Dict[int, List[str]]:
        """
        Return a non-overlapping partition at a given tree level.
        
        For fixed mode: level = number of clusters.
        For full mode: level = depth from root (0 = root, deeper = more clusters).
        
        Returns a partition (each asset appears exactly once).
        """
        if self.cluster_mode == "fixed":
            # level = number of clusters
            n_clusters = min(level, len(self._assets))
            cluster_labels = fcluster(self._linkage_matrix, n_clusters, criterion="maxclust")
            
            clusters = {}
            for cluster_id in set(cluster_labels):
                indices = np.where(cluster_labels == cluster_id)[0].tolist()
                asset_names = [self._assets[i] for i in indices]
                clusters[cluster_id] = asset_names
            return clusters
        
        else:  # "full" mode
            # level = depth from root (0 = root, 1 = root's children, etc.)
            # Find all nodes at this level that partition the assets
            
            # Start from root
            nodes_at_level = [self._root_node_id]
            
            for _ in range(level):
                new_nodes = []
                for node_id in nodes_at_level:
                    if node_id in self._children_map:
                        left, right = self._children_map[node_id]
                        new_nodes.extend([left, right])
                    else:
                        # Leaf node: keep it
                        new_nodes.append(node_id)
                nodes_at_level = new_nodes
                if all(self.is_leaf(node) for node in nodes_at_level):
                    break
            
            # Build partition from nodes at this level
            partition = {}
            for node_id in nodes_at_level:
                assets = self.assets_under(node_id)
                partition[node_id] = assets
            
            return partition
    
    @property
    def linkage_matrix(self) -> np.ndarray:
        """Return hierarchical linkage matrix for dendrogram analysis."""
        return self._linkage_matrix
    
    @property
    def cluster_covariances(self) -> Dict[int, pd.DataFrame]:
        """Return covariance matrix for each cluster using SHRUNK covariance."""
        return {
            node_id: self._shrunk_cov.loc[assets, assets]
            for node_id, assets in self._all_clusters.items()
        }
    
    @property
    def cluster_correlations(self) -> Dict[int, pd.DataFrame]:
        """Return correlation matrix for each cluster for diagnostics."""
        return {
            node_id: self._corr_matrix.loc[assets, assets]
            for node_id, assets in self._all_clusters.items()
        }
    
    # ------------------------------------------------------------------------
    # Public API - Tree Traversal (full mode only)
    # ------------------------------------------------------------------------
    
    def children(self, node_id: int) -> Optional[Tuple[int, int]]:
        """Get children of a node. Requires cluster_mode='full'."""
        if self.cluster_mode != "full":
            raise ValueError(
                f"children() requires cluster_mode='full'. Current mode: {self.cluster_mode}"
            )
        return self._children_map.get(node_id, None)
    
    def parent(self, node_id: int) -> Optional[int]:
        """Get parent of a node. Requires cluster_mode='full'."""
        if self.cluster_mode != "full":
            raise ValueError(
                f"parent() requires cluster_mode='full'. Current mode: {self.cluster_mode}"
            )
        return self._parent_map.get(node_id, None)
    
    def level(self, node_id: int) -> int:
        """Get depth level of a node. Requires cluster_mode='full'."""
        if self.cluster_mode != "full":
            raise ValueError(
                f"level() requires cluster_mode='full'. Current mode: {self.cluster_mode}"
            )
        return self._levels.get(node_id, 0)
    
    def assets_under(self, node_id: int) -> List[str]:
        """Get all assets under a specific node."""
        if self.cluster_mode != "full":
            return self._all_clusters.get(node_id, [])
        return self._node_assets_map.get(node_id, [])
    
    def path_of(self, asset: str) -> List[int]:
        """Get the full path from root to leaf for an asset. Requires cluster_mode='full'."""
        if self.cluster_mode != "full":
            raise ValueError(
                f"path_of() requires cluster_mode='full'. Current mode: {self.cluster_mode}"
            )
        
        leaf_id = None
        for node_id, assets in self._leaf_clusters.items():
            if assets == [asset]:
                leaf_id = node_id
                break
        
        if leaf_id is None:
            raise KeyError(f"Asset '{asset}' not found")
        
        path = [leaf_id]
        current = leaf_id
        while current in self._parent_map:
            current = self._parent_map[current]
            path.append(current)
        
        return path[::-1]
    
    @property
    def root(self) -> int:
        """Get the root node ID."""
        return self._root_node_id
    
    def is_leaf(self, node_id: int) -> bool:
        """Check if a node is a leaf (single asset)."""
        return node_id in self._leaf_clusters
    
    def is_internal(self, node_id: int) -> bool:
        """Check if a node is internal (has children)."""
        return not self.is_leaf(node_id)
    
    # ------------------------------------------------------------------------
    # Public API - Convenience Methods
    # ------------------------------------------------------------------------
    
    def assets_in_cluster(self, cluster_id: int) -> List[str]:
        """Get all assets in a specific cluster."""
        return self._all_clusters.get(cluster_id, []).copy()
    
    def cluster_of(self, asset: str) -> int:
        """Find which cluster an asset belongs to (fixed mode only)."""
        if self.cluster_mode == "full":
            warnings.warn(
                "cluster_of() is ambiguous in full mode. Use path_of() instead.",
                UserWarning
            )
            for node_id, assets in self._internal_clusters.items():
                if asset in assets:
                    return node_id
            for node_id, assets in self._leaf_clusters.items():
                if asset in assets:
                    return node_id
            raise KeyError(f"Asset '{asset}' not found")
        
        for cluster_id, assets in self._all_clusters.items():
            if asset in assets:
                return cluster_id
        raise KeyError(f"Asset '{asset}' not found")
    
    def subtree_weights(self, weights: Dict[str, float]) -> Dict[int, float]:
        """Aggregate asset weights to cluster level."""
        return {
            node_id: sum(weights.get(asset, 0.0) for asset in assets)
            for node_id, assets in self._all_clusters.items()
        }
    
    # ------------------------------------------------------------------------
    # Diagnostics and Visualization
    # ------------------------------------------------------------------------
    
    def plot_dendrogram(self, **kwargs):
        """Plot dendrogram of the hierarchical clustering."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Compute dendrogram with positions
        d = dendrogram(
            self._linkage_matrix,
            labels=self._assets,
            ax=ax,
            orientation="top",
            no_plot=False,
            **kwargs
        )
        
        ax.set_title(f"Hierarchical Clustering Dendrogram ({self.cluster_mode} mode)")
        ax.set_xlabel("Assets")
        ax.set_ylabel("Distance")
        
        # Mark clusters in full mode using the actual leaf order
        if self.cluster_mode == "full":
            leaf_order = d["ivl"]
            y_max = ax.get_ylim()[1]
            
            for node_id, assets in self._internal_clusters.items():
                if len(assets) >= 3:  # Only mark larger clusters
                    # Find positions using leaf order
                    positions = [leaf_order.index(a) for a in assets if a in leaf_order]
                    if positions:
                        x_min = min(positions) - 0.5
                        x_max = max(positions) + 0.5
                        ax.axvspan(x_min, x_max, alpha=0.05, color='blue')
        
        plt.tight_layout()
        return fig, ax
    
    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of all clusters."""
        summary_data = []
        for node_id, assets in self._all_clusters.items():
            n_assets = len(assets)
            
            # Compute equal-weight cluster volatility
            if n_assets > 0:
                cluster_cov = self._shrunk_cov.loc[assets, assets].values
                w = np.ones(n_assets) / n_assets
                cluster_risk = np.sqrt(w @ cluster_cov @ w)
            else:
                cluster_risk = 0.0
            
            # Average correlation within cluster
            if n_assets > 1:
                cluster_corr = self._corr_matrix.loc[assets, assets].values
                triu_indices = np.triu_indices_from(cluster_corr, k=1)
                avg_corr = np.mean(cluster_corr[triu_indices])
            else:
                avg_corr = np.nan
            
            node_type = "leaf" if n_assets == 1 else "root" if node_id == self._root_node_id else "internal"
            
            summary_data.append({
                "node_id": node_id,
                "node_type": node_type,
                "n_assets": n_assets,
                "assets": ", ".join(assets[:5]) + ("..." if n_assets > 5 else ""),
                "cluster_std": cluster_risk,
                "avg_correlation": avg_corr,
                "level": self._levels.get(node_id, 0) if self.cluster_mode == "full" else 0
            })
        
        return pd.DataFrame(summary_data)


# -----------------------------------------------------------------------------
# Optimizer Interface
# -----------------------------------------------------------------------------

class PortfolioOptimizer(ABC):
    """Abstract base class for portfolio optimizers."""
    
    def __init__(
        self,
        risk_model: RiskModel,
        alpha: pd.Series,
        current_weights: Optional[pd.Series] = None
    ):
        self.risk_model = risk_model
        self.alpha = alpha
        self.current_weights = current_weights
        
        if set(alpha.index) != set(risk_model.covariance().index):
            raise ValueError("Alpha index must match risk model assets")
        if current_weights is not None:
            if set(current_weights.index) != set(risk_model.covariance().index):
                raise ValueError("Current weights index must match risk model assets")
    
    @abstractmethod
    def solve(
        self,
        max_cluster_exposure: Optional[float] = None,
        turnover_limit: Optional[float] = None,
        max_weight: Optional[float] = None,
        **kwargs
    ) -> pd.Series:
        """Solve the optimization problem."""
        pass


class ConstrainedMVOOptimizer(PortfolioOptimizer):
    """
    Constrained Mean-Variance Optimization with cluster constraints.
    
    Solves:
    max   alpha^T w - λ/2 * w^T Σ w
    s.t.  sum(w) = 1
          w >= 0
          w <= max_weight
          sum(w_{cluster}) <= max_cluster_exposure (fixed mode only)
          turnover <= turnover_limit
    """
    
    def __init__(
        self,
        risk_model: RiskModel,
        alpha: pd.Series,
        current_weights: Optional[pd.Series] = None,
        risk_aversion: float = 1.0,
        psd_eps: float = 1e-10
    ):
        super().__init__(risk_model, alpha, current_weights)
        self.risk_aversion = risk_aversion
        self.psd_eps = psd_eps
    
    def solve(
        self,
        max_cluster_exposure: Optional[float] = None,
        turnover_limit: Optional[float] = None,
        max_weight: Optional[float] = 0.20,
        **kwargs
    ) -> pd.Series:
        """Solve constrained mean-variance optimization."""
        n = len(self.alpha)
        assets = self.alpha.index.tolist()

        # --- PSD projection for numerical safety ---
        # Reindex to alpha's ordering so cov rows align with w and alpha.
        cov_mat = self.risk_model.covariance().reindex(index=assets, columns=assets).values
        cov_mat = _project_to_psd(cov_mat, self.psd_eps)

        # --- Variables ---
        w = cp.Variable(n, nonneg=True)

        # --- Objective ---
        alpha_vec = self.alpha.values
        objective = cp.Maximize(
            alpha_vec @ w - self.risk_aversion * 0.5 * cp.quad_form(w, cov_mat)
        )
        
        # --- Constraints ---
        constraints = [cp.sum(w) == 1.0]
        
        # Max weight constraint
        if max_weight is not None:
            constraints.append(w <= max_weight)
        
        # Cluster exposure constraints (only in fixed mode)
        if max_cluster_exposure is not None:
            if self.risk_model.cluster_mode != "fixed":
                raise ValueError(
                    "Cluster exposure constraints require cluster_mode='fixed'. "
                    f"Current mode: {self.risk_model.cluster_mode}"
                )
            
            # Cap each multi-asset cluster. internal_clusters() excludes
            # singletons, so single-asset holdings are left uncapped here
            # (they are still bounded by max_weight above). Skip the whole-
            # universe (root) cluster: sum(w) == 1 already governs it, and
            # capping it below 1 would make the problem infeasible.
            universe = set(assets)
            clusters = self.risk_model.internal_clusters()
            for cluster_id, assets_list in clusters.items():
                if set(assets_list) == universe:
                    continue
                indices = [assets.index(a) for a in assets_list]
                constraints.append(cp.sum(w[indices]) <= max_cluster_exposure)
        
        # Turnover constraint (half the L1 norm)
        if turnover_limit is not None and self.current_weights is not None:
            current_vec = self.current_weights.reindex(assets).fillna(0).values
            constraints.append(0.5 * cp.norm(w - current_vec, 1) <= turnover_limit)
        
        # --- Solve ---
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)
        
        if w.value is None:
            raise RuntimeError("Optimization failed to converge")
        
        return pd.Series(w.value, index=assets)


class RobustOptimizer(PortfolioOptimizer):
    """
    Genuinely Robust Portfolio Optimizer with alpha uncertainty.
    
    Solves:
    max   α^T w - ρ * ||D w||₂ - λ/2 * w^T Σ w
    s.t.  sum(w) = 1
          w >= 0
          w <= max_weight
          turnover <= turnover_limit
    
    where D = diag(σ_α) (standard errors of alpha estimates).
    """
    
    def __init__(
        self,
        risk_model: RiskModel,
        alpha: pd.Series,
        alpha_uncertainty: Optional[pd.Series] = None,
        current_weights: Optional[pd.Series] = None,
        risk_aversion: float = 1.0,
        robustness_penalty: float = 1.0,
        psd_eps: float = 1e-10
    ):
        """
        Parameters
        ----------
        risk_model : RiskModel
            Precomputed risk model
        alpha : pd.Series
            Expected returns signal
        alpha_uncertainty : pd.Series, optional
            Standard errors of alpha estimates. If None, uses sqrt(diag(Σ)).
        current_weights : pd.Series, optional
            Current portfolio weights for turnover constraints
        risk_aversion : float
            Risk aversion parameter λ
        robustness_penalty : float
            Robustness penalty ρ for alpha uncertainty
        psd_eps : float
            Small constant for PSD projection
        """
        super().__init__(risk_model, alpha, current_weights)
        self.risk_aversion = risk_aversion
        self.robustness_penalty = robustness_penalty
        self.psd_eps = psd_eps
        
        if alpha_uncertainty is None:
            # Use sqrt of diagonal of covariance as default
            cov_diag = np.diag(risk_model.covariance().values)
            alpha_uncertainty = pd.Series(np.sqrt(cov_diag), index=risk_model.covariance().index)
        
        if set(alpha_uncertainty.index) != set(risk_model.covariance().index):
            raise ValueError("Alpha uncertainty index must match risk model assets")
        
        self.alpha_uncertainty = alpha_uncertainty
    
    def solve(
        self,
        max_cluster_exposure: Optional[float] = None,
        turnover_limit: Optional[float] = None,
        max_weight: Optional[float] = 0.20,
        **kwargs
    ) -> pd.Series:
        """
        Solve robust optimization with alpha uncertainty.
        
        Note: Cluster exposure constraints are NOT applied in full mode
        due to overlapping clusters. Use cut() for a partition if needed.
        """
        n = len(self.alpha)
        assets = self.alpha.index.tolist()

        # --- PSD projection for numerical safety ---
        # Reindex to alpha's ordering so cov rows align with w and alpha.
        cov_mat = self.risk_model.covariance().reindex(index=assets, columns=assets).values
        cov_mat = _project_to_psd(cov_mat, self.psd_eps)

        # --- Variables ---
        w = cp.Variable(n, nonneg=True)

        # --- Objective: alpha + robustness penalty + risk penalty ---
        alpha_vec = self.alpha.values
        uncertainty_vec = self.alpha_uncertainty.reindex(assets).values
        
        objective = cp.Maximize(
            alpha_vec @ w
            - self.robustness_penalty * cp.norm(cp.multiply(uncertainty_vec, w), 2)
            - self.risk_aversion * 0.5 * cp.quad_form(w, cov_mat)
        )
        
        # --- Constraints ---
        constraints = [cp.sum(w) == 1.0]
        
        # Max weight constraint
        if max_weight is not None:
            constraints.append(w <= max_weight)
        
        # Turnover constraint (half the L1 norm)
        if turnover_limit is not None and self.current_weights is not None:
            current_vec = self.current_weights.reindex(assets).fillna(0).values
            constraints.append(0.5 * cp.norm(w - current_vec, 1) <= turnover_limit)
        
        # --- Cluster constraints using a non-overlapping partition ---
        if max_cluster_exposure is not None:
            # Fixed mode: the multi-asset fcluster groups (excluding the
            # whole-universe root, which sum(w) == 1 already governs).
            # Full mode: cut(level=1) gives root's children (a partition).
            if self.risk_model.cluster_mode == "fixed":
                partition = self.risk_model.internal_clusters()
            else:
                partition = self.risk_model.cut(level=1)

            universe = set(assets)
            for cluster_id, assets_list in partition.items():
                if set(assets_list) == universe:
                    continue
                indices = [assets.index(a) for a in assets_list if a in assets]
                if indices:
                    constraints.append(cp.sum(w[indices]) <= max_cluster_exposure)
        
        # --- Solve ---
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)
        
        if w.value is None:
            raise RuntimeError("Optimization failed to converge")
        
        return pd.Series(w.value, index=assets)


