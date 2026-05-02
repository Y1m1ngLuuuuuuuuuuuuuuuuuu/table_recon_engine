from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from table_recon_engine.data_structures import CellBox, DetectionBox


def projection_overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    shorter = max(1e-6, min(abs(a1 - a0), abs(b1 - b0)))
    return inter / shorter


@dataclass(slots=True)
class ClusterConfig:
    row_tolerance_factor: float = 0.55
    col_tolerance_factor: float = 0.55
    min_tolerance: float = 4.0
    min_projection_overlap: float = 0.35
    drop_low_score: float = 0.0


@dataclass(slots=True)
class AxisCluster:
    index: int
    item_indexes: list[int] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    center: float = 0.0

    def add(self, item_index: int, start: float, end: float) -> None:
        self.item_indexes.append(item_index)
        if len(self.item_indexes) == 1:
            self.start = start
            self.end = end
            self.center = (start + end) * 0.5
            return
        self.start = min(self.start, start)
        self.end = max(self.end, end)
        old_count = len(self.item_indexes) - 1
        self.center = (self.center * old_count + (start + end) * 0.5) / len(self.item_indexes)


class CellCluster:
    """Hand-written row/column clustering for detected table-cell boxes."""

    def __init__(self, config: ClusterConfig | None = None) -> None:
        self.config = config or ClusterConfig()

    def cluster(self, detections: list[DetectionBox]) -> tuple[list[CellBox], list[AxisCluster], list[AxisCluster]]:
        filtered = [box.ordered() for box in detections if box.score >= self.config.drop_low_score]
        if not filtered:
            return [], [], []

        row_clusters = self._cluster_axis(filtered, axis="y")
        col_clusters = self._cluster_axis(filtered, axis="x")

        row_of: dict[int, int] = {}
        col_of: dict[int, int] = {}
        for cluster in row_clusters:
            for item_index in cluster.item_indexes:
                row_of[item_index] = cluster.index
        for cluster in col_clusters:
            for item_index in cluster.item_indexes:
                col_of[item_index] = cluster.index

        cells = [
            CellBox.from_detection(
                box,
                row=row_of.get(idx, -1),
                col=col_of.get(idx, -1),
            )
            for idx, box in enumerate(filtered)
        ]
        cells.sort(key=lambda cell: (cell.row, cell.col, cell.y0, cell.x0))
        return cells, row_clusters, col_clusters

    def _cluster_axis(self, boxes: list[DetectionBox], axis: str) -> list[AxisCluster]:
        if axis == "y":
            intervals = [(idx, box.y0, box.y1) for idx, box in enumerate(boxes)]
            sizes = [box.height for box in boxes if box.height > 0]
            tolerance = max(self.config.min_tolerance, median(sizes) * self.config.row_tolerance_factor)
        elif axis == "x":
            intervals = [(idx, box.x0, box.x1) for idx, box in enumerate(boxes)]
            sizes = [box.width for box in boxes if box.width > 0]
            tolerance = max(self.config.min_tolerance, median(sizes) * self.config.col_tolerance_factor)
        else:
            raise ValueError(f"Unknown axis: {axis}")

        intervals.sort(key=lambda item: ((item[1] + item[2]) * 0.5, item[1]))
        clusters: list[AxisCluster] = []
        for item_index, start, end in intervals:
            center = (start + end) * 0.5
            best_cluster: AxisCluster | None = None
            best_score = -1.0
            for cluster in clusters:
                overlap = projection_overlap_ratio(start, end, cluster.start, cluster.end)
                center_distance = abs(center - cluster.center)
                if overlap >= self.config.min_projection_overlap or center_distance <= tolerance:
                    score = overlap - center_distance / max(tolerance, 1e-6)
                    if score > best_score:
                        best_score = score
                        best_cluster = cluster
            if best_cluster is None:
                best_cluster = AxisCluster(index=len(clusters))
                clusters.append(best_cluster)
            best_cluster.add(item_index, start, end)

        clusters.sort(key=lambda cluster: cluster.center)
        for idx, cluster in enumerate(clusters):
            cluster.index = idx
        return clusters
