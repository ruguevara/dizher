"""Renders for judging by eye: pictures enlarged without smoothing, with cell rulers so that a cell can be named by
its (row, column), a faint cell grid, eye views, contact sheets of several columns and crops of a few cells."""
import cv2
import numpy as np

MARGIN = 16         # px of ruler above and to the left of a picture
GRID = 0.25         # opacity of the cell grid over the picture
GAP = 6             # px between tiles of a sheet
TITLE = 18          # px of a tile's title strip
FONT = cv2.FONT_HERSHEY_PLAIN
RULER_BG = (236, 236, 236)


def to_ubyte(img) -> np.ndarray:
    """float 0..1 or uint8, (H, W, 3) or (H, W) -> (H, W, 3) uint8."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    return np.ascontiguousarray(a)


def up(img, zoom: int) -> np.ndarray:
    """Nearest-neighbour enlargement: every pixel a zoom x zoom square."""
    return np.ascontiguousarray(np.repeat(np.repeat(to_ubyte(img), zoom, 0), zoom, 1))


def label_step(cell_px: int) -> int:
    """Every how many cells the ruler carries a number: a two-digit number needs ~16 px."""
    return 1 if cell_px >= 20 else 2 if cell_px >= 10 else 4


def ruled(img, zoom: int = 3, cell=(8, 8), origin=(0, 0), grid: bool = True) -> np.ndarray:
    """The picture at zoom with rulers: column numbers above, row numbers to the left, ticks at every cell (longer at
    every fourth), a faint grid on the cell borders. origin (row, col) is the cell the picture's top-left cell is on
    the screen, so a crop's numbers continue the screen's. The picture's pixels sit at [MARGIN:, MARGIN:] and are
    untouched but for the grid lines."""
    h, w = cell
    pic = up(img, zoom)
    H, W = pic.shape[:2]
    R, C = img.shape[0] // h, img.shape[1] // w
    if grid:
        for r in range(1, R):
            y = r * h * zoom
            pic[y] = (pic[y] * (1 - GRID) + 255 * GRID).astype(np.uint8)
        for c in range(1, C):
            x = c * w * zoom
            pic[:, x] = (pic[:, x] * (1 - GRID) + 255 * GRID).astype(np.uint8)
    out = np.full((H + MARGIN, W + MARGIN, 3), RULER_BG, np.uint8)
    out[MARGIN:, MARGIN:] = pic
    r0, c0 = origin
    scale, thickness = 0.7, 1
    step_c, step_r = label_step(w * zoom), label_step(h * zoom)
    for c in range(C):
        x = MARGIN + c * w * zoom
        n = c0 + c
        cv2.line(out, (x, MARGIN - 1), (x, MARGIN - (6 if n % 4 == 0 else 3)), (90, 90, 90), 1)
        if n % step_c == 0:
            (tw, th), _ = cv2.getTextSize(str(n), FONT, scale, thickness)
            cv2.putText(out, str(n), (x + 2, th + 1), FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    for r in range(R):
        y = MARGIN + r * h * zoom
        n = r0 + r
        cv2.line(out, (MARGIN - 1, y), (MARGIN - (6 if n % 4 == 0 else 3), y), (90, 90, 90), 1)
        if n % step_r == 0:
            (tw, th), _ = cv2.getTextSize(str(n), FONT, scale, thickness)
            cv2.putText(out, str(n), (max(0, MARGIN - 2 - tw), y + th + 2), FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return out


def titled(tile: np.ndarray, text: str) -> np.ndarray:
    """A white strip with the text above the tile."""
    strip = np.full((TITLE, tile.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, text, (3, TITLE - 5), FONT, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([strip, tile])


def hstack(tiles, gap: int = GAP, fill=128) -> np.ndarray:
    """Tiles side by side, shorter ones padded below."""
    H = max(t.shape[0] for t in tiles)
    row = []
    for t in tiles:
        if t.shape[0] < H:
            t = np.concatenate([t, np.full((H - t.shape[0], t.shape[1], 3), fill, np.uint8)])
        row += [t, np.full((H, gap, 3), fill, np.uint8)]
    return np.concatenate(row[:-1], axis=1)


def vstack(rows, gap: int = GAP, fill=128) -> np.ndarray:
    """Rows one under another, narrower ones padded to the right."""
    W = max(r.shape[1] for r in rows)
    col = []
    for r in rows:
        if r.shape[1] < W:
            r = np.concatenate([r, np.full((r.shape[0], W - r.shape[1], 3), fill, np.uint8)], axis=1)
        col += [r, np.full((gap, W, 3), fill, np.uint8)]
    return np.concatenate(col[:-1])


def crop_cells(img, r0: int, c0: int, r1: int, c1: int, cell=(8, 8)):
    """The cells r0..r1 and c0..c1, inclusive, of a picture (not enlarged)."""
    h, w = cell
    return img[r0 * h:(r1 + 1) * h, c0 * w:(c1 + 1) * w]


def outline(img, cells, colour=(255, 0, 0), zoom: int = 1, cell=(8, 8), margin: int = 0):
    """Cells outlined on a rendered (enlarged, maybe ruled) picture."""
    out = img.copy()
    h, w = cell
    for r, c in cells:
        y, x = margin + r * h * zoom, margin + c * w * zoom
        cv2.rectangle(out, (x, y), (x + w * zoom - 1, y + h * zoom - 1), colour, 1)
    return out


def sheet(columns, zoom: int = 3, eye=None, crops=(), cell=(8, 8), marks=None):
    """Contact sheet: a titled, ruled tile per (title, picture) column; under them, when eye is given (a function
    picture -> eye view), the same columns through the eye model; then each crop (r0, c0, r1, c1) of every column at
    twice the zoom. marks: title -> cells to outline in red on that column."""
    marks = marks or {}
    tile = lambda title, pic, z, origin=(0, 0): outline(ruled(pic, z, cell, origin), marks.get(title, ()), (255, 0, 0), z, cell, MARGIN)
    rows = [hstack([titled(tile(t, p, zoom), t) for t, p in columns])]
    if eye is not None:
        rows.append(hstack([titled(ruled(eye(p), zoom, cell), f'{t} / eye') for t, p in columns]))
    for r0, c0, r1, c1 in crops:
        z = zoom * 2
        rows.append(hstack([titled(tile(t, crop_cells(p, r0, c0, r1, c1, cell), z, (r0, c0)),
                                   f'{t} cells {r0}..{r1}, {c0}..{c1}') for t, p in columns]))
        if eye is not None:
            rows.append(hstack([titled(ruled(crop_cells(eye(p), r0, c0, r1, c1, cell), z, cell, (r0, c0)),
                                       f'{t} / eye') for t, p in columns]))
    return vstack(rows)


def save(path, img) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(to_ubyte(img), cv2.COLOR_RGB2BGR))
