from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DetectionBox:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float = 1.0
    class_id: int = 0
    text: str = ""

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) * 0.5

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) * 0.5

    @property
    def area(self) -> float:
        return self.width * self.height

    def clamp(self, image_width: int, image_height: int) -> "DetectionBox":
        return DetectionBox(
            x0=min(max(self.x0, 0.0), float(image_width)),
            y0=min(max(self.y0, 0.0), float(image_height)),
            x1=min(max(self.x1, 0.0), float(image_width)),
            y1=min(max(self.y1, 0.0), float(image_height)),
            score=self.score,
            class_id=self.class_id,
            text=self.text,
        ).ordered()

    def ordered(self) -> "DetectionBox":
        return DetectionBox(
            x0=min(self.x0, self.x1),
            y0=min(self.y0, self.y1),
            x1=max(self.x0, self.x1),
            y1=max(self.y0, self.y1),
            score=self.score,
            class_id=self.class_id,
            text=self.text,
        )

    def to_yolo(self, image_width: int, image_height: int) -> str:
        cx = self.cx / image_width
        cy = self.cy / image_height
        width = self.width / image_width
        height = self.height / image_height
        return f"{self.class_id} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}"

    @classmethod
    def from_yolo(
        cls,
        row: str,
        image_width: int,
        image_height: int,
        score: float = 1.0,
    ) -> "DetectionBox":
        parts = row.strip().split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO row: {row!r}")
        class_id = int(float(parts[0]))
        cx = float(parts[1]) * image_width
        cy = float(parts[2]) * image_height
        width = float(parts[3]) * image_width
        height = float(parts[4]) * image_height
        return cls(
            x0=cx - width * 0.5,
            y0=cy - height * 0.5,
            x1=cx + width * 0.5,
            y1=cy + height * 0.5,
            score=score,
            class_id=class_id,
        ).ordered()


@dataclass(slots=True)
class CellBox(DetectionBox):
    row: int = -1
    col: int = -1
    rowspan: int = 1
    colspan: int = 1

    @classmethod
    def from_detection(
        cls,
        box: DetectionBox,
        row: int = -1,
        col: int = -1,
        rowspan: int = 1,
        colspan: int = 1,
    ) -> "CellBox":
        return cls(
            x0=box.x0,
            y0=box.y0,
            x1=box.x1,
            y1=box.y1,
            score=box.score,
            class_id=box.class_id,
            text=box.text,
            row=row,
            col=col,
            rowspan=rowspan,
            colspan=colspan,
        )
