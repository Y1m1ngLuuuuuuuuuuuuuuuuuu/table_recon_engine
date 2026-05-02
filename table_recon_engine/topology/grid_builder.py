from __future__ import annotations

from dataclasses import dataclass

from table_recon_engine.data_structures import CellBox
from table_recon_engine.topology.cell_cluster import AxisCluster


@dataclass(slots=True)
class GridSlot:
    cell: CellBox | None = None
    is_anchor: bool = True


@dataclass(slots=True)
class TableGrid:
    slots: list[list[GridSlot]]
    row_centers: list[float]
    col_centers: list[float]

    @property
    def n_rows(self) -> int:
        return len(self.slots)

    @property
    def n_cols(self) -> int:
        return len(self.slots[0]) if self.slots else 0


class GridBuilder:
    """Builds a dense virtual table grid from clustered cell boxes."""

    def __init__(self, span_center_margin: float = 0.08) -> None:
        self.span_center_margin = span_center_margin

    def build(
        self,
        cells: list[CellBox],
        row_clusters: list[AxisCluster],
        col_clusters: list[AxisCluster],
    ) -> TableGrid:
        row_count = len(row_clusters)
        col_count = len(col_clusters)
        slots = [[GridSlot() for _ in range(col_count)] for _ in range(row_count)]
        row_centers = [cluster.center for cluster in row_clusters]
        col_centers = [cluster.center for cluster in col_clusters]

        for cell in sorted(cells, key=lambda item: (item.row, item.col, -item.area)):
            covered_rows = self._covered_indexes(cell.y0, cell.y1, row_centers)
            covered_cols = self._covered_indexes(cell.x0, cell.x1, col_centers)
            if not covered_rows:
                covered_rows = [max(cell.row, 0)]
            if not covered_cols:
                covered_cols = [max(cell.col, 0)]

            row = min(covered_rows)
            col = min(covered_cols)
            if not (0 <= row < row_count and 0 <= col < col_count):
                continue

            cell.row = row
            cell.col = col
            cell.rowspan = max(1, len(covered_rows))
            cell.colspan = max(1, len(covered_cols))

            if slots[row][col].cell is not None and slots[row][col].is_anchor:
                row, col = self._nearest_empty(slots, row, col)
                cell.row = row
                cell.col = col

            for r in range(row, min(row + cell.rowspan, row_count)):
                for c in range(col, min(col + cell.colspan, col_count)):
                    slots[r][c] = GridSlot(cell=cell, is_anchor=(r == row and c == col))

        return TableGrid(slots=slots, row_centers=row_centers, col_centers=col_centers)

    def _covered_indexes(self, start: float, end: float, centers: list[float]) -> list[int]:
        if not centers:
            return []
        width = max(1e-6, end - start)
        margin = width * self.span_center_margin
        return [idx for idx, center in enumerate(centers) if start - margin <= center <= end + margin]

    @staticmethod
    def _nearest_empty(slots: list[list[GridSlot]], row: int, col: int) -> tuple[int, int]:
        for radius in range(max(len(slots), len(slots[0]) if slots else 0) + 1):
            for r in range(max(0, row - radius), min(len(slots), row + radius + 1)):
                for c in range(max(0, col - radius), min(len(slots[r]), col + radius + 1)):
                    if slots[r][c].cell is None:
                        return r, c
        return row, col
