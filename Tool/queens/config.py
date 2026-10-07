"""Settings shared across the project (no heavy imports here)."""
SIZES = list(range(4, 15))  # grid sizes 4x4 to 14x14
MAX_ZONES = max(SIZES)  # one zone per row, so at most 14