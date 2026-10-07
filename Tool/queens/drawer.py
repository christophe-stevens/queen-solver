from PIL import Image, ImageDraw

class GridDrawer:
    """Renders a Grid as a Queens-style image."""
    SIZE = 490            # image width and height in pixels
    THIN, THICK = 1, 4    # cell line and zone border widths
    # RGB in the same order as Grid.COLOURS: Y R B P G W O C K N L
    RGB = [(250, 227, 120), (255, 123, 96), (150, 190, 255), (190, 160, 220),
           (179, 223, 160), (230, 230, 230), (255, 180, 110), (150, 220, 230),
           (240, 170, 200), (200, 160, 120), (215, 235, 120),
           (60, 160, 140), (210, 90, 200), (235, 215, 170)]  # teal, magenta, sand

    def __init__(self, grid):
        self.grid = grid
        self.cell = self.SIZE / grid.m

    def draw(self, show_queens=False):
        img = Image.new("RGB", (self.SIZE, self.SIZE), "white")
        d = ImageDraw.Draw(img)
        z, q, s, m = self.grid.zones, self.grid.queens, self.cell, self.grid.m
        for r in range(m):              # cells, thin lines, queens
            for c in range(m):
                x, y = c * s, r * s
                d.rectangle([x, y, x + s, y + s], fill=self.RGB[z[r, c]], outline="grey", width=self.THIN)
                if show_queens and q[r, c]:
                    d.ellipse([x + s / 3, y + s / 3, x + 2 * s / 3, y + 2 * s / 3], fill="black")
        for r in range(m):              # thick borders between zones
            for c in range(m):
                x, y = c * s, r * s
                if c + 1 < m and z[r, c] != z[r, c + 1]:
                    d.line([x + s, y, x + s, y + s], fill="black", width=self.THICK)
                if r + 1 < m and z[r, c] != z[r + 1, c]:
                    d.line([x, y + s, x + s, y + s], fill="black", width=self.THICK)
        d.rectangle([0, 0, self.SIZE - 1, self.SIZE - 1], outline="black", width=self.THICK)
        return img
