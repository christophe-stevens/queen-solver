"""Research extras: puzzle generation and the learned queen-placement model.
Needs Tool/ on sys.path too (for queens.drawer, used by pipeline.solve_image)."""
from .grid import Grid
from .queen_model import QueenModel, SinkhornQueenModel, ZoneSinkhornQueenModel
from .pipeline import solve_image

__all__ = []
