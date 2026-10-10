from collections.abc import Sequence
from pathlib import Path

from matplotlib.artist import Artist
from matplotlib.axes import Axes

class Figure:
    axes: list[Axes]
    def __init__(self, figsize: tuple[float, float], layout: str) -> None: ...
    def add_subplot(self, nrows: int, ncols: int, index: int) -> Axes: ...
    def legend(self, handles: Sequence[Artist] | None = None, **kwargs: object) -> Artist: ...
    def savefig(
        self,
        path: Path,
        format: str,
        dpi: float | None = None,
        metadata: dict[str, None] | None = None,
    ) -> None: ...
