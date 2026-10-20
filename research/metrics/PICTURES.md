# The picture collection

PLAN.md, "The pictures": 40 new sources, tagged, prepared, frozen, split at random. This file is the tag list and the
manifest's format.

## Categories (one per picture)

| tag | what | count |
|---|---|---|
| `illus` | artists' illustrations: fantasy, sci-fi, painted portraits, book and pulp covers | 15 |
| `portrait` | artistic and processed photo portraits | 10 |
| `screen` | posters, film and game frames, game title screens, AI-made images | 8 |
| `land` | landscapes | 5 |
| `photo` | plain photographs | 2 |

## Difficulty tags (any number per picture; each on at least 6 pictures)

| tag | what makes a conversion hard |
|---|---|
| `gradient` | a large smooth gradient: sky, sunset, studio backdrop, glow |
| `face` | a face large enough to read (at least about 1/6 of the height) |
| `shadow` | large deep shadows with detail in them (low key) |
| `saturated` | saturated colours: neon, flowers, coloured light |
| `detail` | fine detail: hair, foliage, engraving, armour, machinery |
| `text` | lettering in the picture |
| `skin` | large areas of skin in mid tones |
| `pastel` | pale, low-saturation colours (hard for a saturated palette) |
| `hue` | colours with no near ZX colour: brown, orange, pink, olive, teal |
| `edges` | strong diagonal or curved edges between flat colours (staircase) |

## The manifest

`pictures.csv`, one row per picture, committed with the files' hashes before anything is converted:

```
id,file,category,tags,source_url,licence,author,publish,sha256
p01,p01.png,illus,gradient face hue,https://commons.wikimedia.org/...,CC BY-SA 4.0,Name,yes,...
```

- `file`: the prepared 256x192 PNG. The full-size original stays beside it as `pNN-orig.*`.
- `publish`: `yes` if the licence allows showing it in the article (own work, public domain, CC); `no` for film and
  game frames and anything else used for research only. The article's figures use only `yes` pictures.
- The files themselves stay out of git (`data/pictures/`); git holds the manifest and the hashes.
