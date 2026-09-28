"""Artists' screens from zxart.ee as a reference set: the top standard pictures by votes, each a project whose source
is the artist's screen seen through a wider eye blur (the picture behind the dots) and whose reference is the screen
itself. The artists' attributes are known good colourings of those sources, so `run` and `compare` score pair selection
against them with no hand painting; `zxart stats` gives what pairs artists use and how often they change them.

tests/images/zxart/index.json (in git) lists the pictures; the screens, sources and projects under
tests/images/zxart/<id>/ are downloaded and regenerated (the artists' works stay out of git).
"""
import html
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from mokit.project import save_project

from dizher import ops
from dizher.converter.energy import LRGB2OPP, dot_contrast
from dizher.converter.eye import eye_kernel
from dizher.platforms.zxspectrum import ZXPalette

from . import render as R
from .project import IMAGES
from .scr import read_scr, has_flash, shown, seams, same_region, pair_name, render_scr

FOLDER = IMAGES / 'zxart'
API = ('https://zxart.ee/api/action:filter/export:zxPicture/language:eng/start:{start}/limit:{limit}/'
       'filter:zxPictureType=standard;zxPictureMinRating={rating}/order:votes,desc/')
GITIGNORE = '*\n!.gitignore\n!index.json\n'
LUMA_SIGMA, CHROMA_SIGMA = 1.0, 2.5    # px of the source blur: the dots' lightness texture stays, their colour averages


def download(url: str, tries: int = 4) -> bytes:
    """The URL's bytes, retried with backoff (2, 4, 8 s): the tunnel drops now and then."""
    url = urllib.parse.quote(url, safe=":/?=;,&()'")           # file names come with non-ASCII letters
    for i in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def api(start: int, limit: int, rating: float, fetch=None) -> list:
    """Picture entries from the API: id, title, rating, year, url, original (the .scr)."""
    fetch = fetch or download
    data = json.loads(fetch(API.format(start=start, limit=limit, rating=rating)))
    assert data.get('responseStatus') == 'success', data
    return [dict(id=int(p['id']), title=html.unescape(p['title']), rating=float(p['rating']), year=p.get('year'), url=p['url'],
                 authors=p.get('authorIds', []), original=p['originalUrl'])
            for p in data['responseData']['zxPicture'] if p.get('originalUrl', '').lower().endswith('.scr')]


def folder(pid) -> Path:
    return FOLDER / str(pid)


def name(pid) -> str:
    """The bench's project name of a picture."""
    return f'zxart/{pid}'


def load_index() -> list:
    f = FOLDER / 'index.json'
    return json.loads(f.read_text()) if f.exists() else []


def save_index(entries) -> None:
    FOLDER.mkdir(parents=True, exist_ok=True)
    (FOLDER / '.gitignore').write_text(GITIGNORE)
    (FOLDER / 'index.json').write_text(json.dumps(entries, indent=1, ensure_ascii=False))


def fetch(n: int, rating: float = 4.0, get=None, log=print) -> list:
    """The first n pictures by votes whose screen is a plain 6912-byte file without flash, downloaded to their
    folders as reference.scr; the index saved. get(url) -> bytes stands in for the network in tests."""
    get = get or download
    entries, start, page = [], 0, 40
    while len(entries) < n:
        batch = api(start, page, rating, get)
        if not batch:
            break
        start += page
        for e in batch:
            if len(entries) >= n:
                break
            f = folder(e['id']) / 'reference.scr'
            if not f.exists():
                data = get(e['original'])
                if len(data) != 6912:
                    log(f'{e["id"]} {e["title"]}: {len(data)} bytes, skipped')
                    continue
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(data)
            if has_flash(f):
                log(f'{e["id"]} {e["title"]}: flash, skipped')
                continue
            entries.append(e)
            log(f'{e["id"]} {e["title"]} ({e["year"]}, {e["rating"]})')
    save_index(entries)
    return entries


def blurred(rgb, luma_sigma=LUMA_SIGMA, chroma_sigma=CHROMA_SIGMA, gamma=2.2):
    """The screen through a Gaussian eye blur in the opponent space: lightness by luma_sigma px, colour by
    chroma_sigma px; back in sRGB. The source a converter is given."""
    opp = (rgb.astype(np.float32) ** gamma) @ LRGB2OPP.T
    out = np.empty_like(opp)
    for k, sigma in enumerate((luma_sigma, chroma_sigma, chroma_sigma)):
        h = eye_kernel(sigma * np.sqrt(2), 2.0)
        out[..., k] = cv2.filter2D(np.ascontiguousarray(opp[..., k]), -1, h, borderType=cv2.BORDER_REFLECT_101)
    return (out @ np.linalg.inv(LRGB2OPP).T).clip(0, 1) ** (1 / gamma)


def prepare(entries=None, luma_sigma=LUMA_SIGMA, chroma_sigma=CHROMA_SIGMA, log=print) -> list:
    """Each picture's source.png (its screen blurred) and project.json (the pipeline's defaults, Fit framing, the
    whole palette), so `run zxart/<id>` scores a method against the artist. Returns the project names."""
    palette = ZXPalette()
    names = []
    for e in entries or load_index():
        d = folder(e['id'])
        bitmap, idx = read_scr(d / 'reference.scr')
        R.save(d / 'source.png', blurred(render_scr(bitmap, idx, palette), luma_sigma, chroma_sigma))
        graph = ops.make_graph()
        graph = graph.with_params('source', replace(graph['source'].params, path=d / 'source.png'))
        graph = graph.with_params('framing', replace(graph['framing'].params, fit='Fit'))
        save_project(d, graph, {})
        names.append(name(e['id']))
        log(f'{name(e["id"])}: {e["title"]}')
    return names


def stats(entries=None, palette=ZXPalette()) -> dict:
    """What the artists do: pair frequencies (names, blacks one, solid cells as one colour), the share of seams that
    change pair, of non-solid cells whose pair clashes in hue (dot_contrast chroma over CLASH), of bright cells,
    of cells with magenta, and the distinct pairs per picture."""
    pairs = np.array(list(palette.iter_idxs_pairs()))
    contrast = dict(zip((tuple(p) for p in pairs.tolist()), dot_contrast(palette.as_float()[pairs], 2.2)[:, 1]))
    counts, changes, total_seams, clash, mixed, bright, magenta, cells, distinct = {}, 0, 0, 0, 0, 0, 0, 0, []
    key = lambda p: (min(p), max(p)) if (min(p), max(p)) in contrast else (max(p), min(p))
    for e in entries or load_index():
        bitmap, idx = read_scr(folder(e['id']) / 'reference.scr')
        s = shown(bitmap, idx)
        for p in map(tuple, s.reshape(-1, 2).tolist()):
            counts[p] = counts.get(p, 0) + 1
        a, b = seams(s)
        changes += int((~same_region(a, b)).sum())
        total_seams += len(a)
        solid = s[..., 0] == s[..., 1]
        mixed += int((~solid).sum())
        clash += sum(contrast.get(key(tuple(p)), 0) > CLASH for p in s[~solid].tolist())
        bright += int((idx >= 8).any(-1).sum())
        magenta += int(((s % 8) == 3).any(-1).sum())
        cells += s.shape[0] * s.shape[1]
        distinct.append(len({tuple(p) for p in s.reshape(-1, 2).tolist()}))
    top = sorted(counts.items(), key=lambda kv: -kv[1])
    return dict(pictures=len(distinct), cells=cells,
                pairs=[(pair_name(*p), c / cells, contrast.get(key(p), 0.0)) for p, c in top],
                seam_changes=changes / max(total_seams, 1), clash=clash / max(mixed, 1), mixed=mixed / max(cells, 1),
                bright=bright / max(cells, 1), magenta=magenta / max(cells, 1),
                distinct=float(np.mean(distinct)) if distinct else float('nan'))


CLASH = 0.5    # chroma dot contrast (hue alone, CIELAB, see energy.dot_contrast) from which a pair clashes


def print_stats(s) -> None:
    print(f"{s['pictures']} pictures, {s['cells']} cells: {s['mixed']:.2f} mixed, {s['bright']:.2f} bright, "
          f"{s['magenta']:.3f} with magenta, {s['clash']:.3f} of the mixed cells with clashing hues, "
          f"{s['seam_changes']:.2f} of the seams change pair, {s['distinct']:.1f} distinct pairs per picture")
    print('pairs (share, hue clash): ' + ', '.join(f'{n} {f:.3f} ({c:.1f})' for n, f, c in s['pairs'][:24]))
