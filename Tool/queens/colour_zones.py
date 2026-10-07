"""Read zones by grouping cells of matching colour: works with any palette."""
import numpy as np


def zones_by_colour(img, m):
    # PIL image of an m x m grid -> (m,m) zone numbers, or None if no clean grouping
    a = np.array(img.convert("RGB")).astype(float)
    h, w, _ = a.shape
    # Median colour of each cell's central patch (avoids borders, X marks, queen icons)
    cells = np.array([np.median(a[int((r + .3) * h / m):int((r + .7) * h / m),
                                  int((c + .3) * w / m):int((c + .7) * w / m)].reshape(-1, 3), axis=0)
                      for r in range(m) for c in range(m)])
    for tol in range(5, 80, 5):  # loosen the colour match until exactly m groups remain
        centres, labels = [], []
        for col in cells:
            d = [np.linalg.norm(col - c) for c in centres]
            if d and min(d) < tol:
                labels.append(int(np.argmin(d)))  # joins the closest existing group
            else:
                centres.append(col)               # starts a new group
                labels.append(len(centres) - 1)
        if len(centres) == m:
            return np.array(labels).reshape(m, m)
    return None