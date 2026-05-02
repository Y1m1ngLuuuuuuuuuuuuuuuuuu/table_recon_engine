from __future__ import annotations

from table_recon_engine.topology.grid_builder import TableGrid


class LaTeXGenerator:
    """Serializes a virtual table grid into a LaTeX tabular environment."""

    def __init__(self, default_text: str = "") -> None:
        self.default_text = default_text

    def generate(self, grid: TableGrid) -> str:
        if grid.n_rows == 0 or grid.n_cols == 0:
            return "\\begin{tabular}{}\n\\end{tabular}"

        alignment = "c" * grid.n_cols
        lines = [f"\\begin{{tabular}}{{{alignment}}}"]
        for row in grid.slots:
            parts: list[str] = []
            col = 0
            while col < grid.n_cols:
                slot = row[col]
                if slot.cell is None:
                    parts.append("")
                    col += 1
                    continue
                if not slot.is_anchor:
                    col += 1
                    continue

                text = self._cell_text(slot.cell.text)
                if slot.cell.rowspan > 1:
                    text = f"\\multirow{{{slot.cell.rowspan}}}{{*}}{{{text}}}"
                if slot.cell.colspan > 1:
                    parts.append(f"\\multicolumn{{{slot.cell.colspan}}}{{c}}{{{text}}}")
                    col += slot.cell.colspan
                else:
                    parts.append(text)
                    col += 1
            lines.append(" & ".join(parts) + r" \\")
        lines.append("\\end{tabular}")
        return "\n".join(lines)

    def _cell_text(self, text: str) -> str:
        value = text if text else self.default_text
        return self._escape_latex(value)

    @staticmethod
    def _escape_latex(value: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(char, char) for char in value)
