"""Builds painted-style textures for the game from reference/reference.png and writes them to ./assets:

  leaves_atlas.png   4 colours (green / yellow / orange / red) x 2 variants of lobed leaf clusters, painted colours
                     sampled from the reference, transparent between leaves (used for tree and bush foliage cards)
  pine_atlas.png     2 variants of drooping needle tufts, painted in the deep teal-greens of the reference
  bark.png           tileable dark bark taken from the big trunks

Run once (or again after changing reference/reference.png):  py make_leaves.py
"""
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from scipy import ndimage as ndi

HERE = Path(__file__).parent
CELL = 256


def load():
    src = Image.open(HERE / "reference" / "reference.png").convert("RGB")
    hsv = np.asarray(src.convert("HSV")).astype(float)
    return src, hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255


def best_windows(mask, size, count, margin=0):
    """Top-left corners of the `count` non-overlapping size x size windows best covered by `mask`."""
    cov = ndi.uniform_filter(mask.astype(float), size, mode="constant")
    h, w = mask.shape
    half = size // 2
    cov[:half + margin], cov[-half - margin:] = 0, 0
    cov[:, :half + margin], cov[:, -half - margin:] = 0, 0
    picks = []
    for idx in np.argsort(cov, axis=None)[::-1]:
        cy, cx = divmod(int(idx), w)
        if cov[cy, cx] <= 0:
            break
        if all(abs(cy - py) >= size or abs(cx - px) >= size for py, px, _ in picks):
            picks.append((cy, cx, cov[cy, cx]))
            if len(picks) == count:
                break
    return [(cx - half, cy - half, c) for cy, cx, c in picks]


def mirror_tile(img):
    """Make a texture seamlessly tileable by mirroring it into a 2x2 block."""
    a = np.asarray(img)
    top = np.concatenate([a, a[:, ::-1]], axis=1)
    return Image.fromarray(np.concatenate([top, top[::-1]], axis=0))


def leaf_cluster(swatch, seed):
    """One 256x256 RGBA card: a cluster of overlapping lobed (oak/maple-like) leaves painted with the swatch colours."""
    rnd = random.Random(seed)
    tex = np.asarray(swatch.resize((CELL, CELL), Image.LANCZOS)).astype(float)
    rgb = np.empty_like(tex)
    rgb[:] = tex.mean(axis=(0, 1))
    alpha = np.zeros((CELL, CELL))
    leaves = []
    for _ in range(22):
        a, r = rnd.uniform(0, 2 * math.pi), 86 * math.sqrt(rnd.random())
        leaves.append((CELL / 2 + math.cos(a) * r, CELL / 2 + math.sin(a) * r * .9, rnd.uniform(22, 33), rnd.uniform(0, 6.28)))
    leaves.sort(key=lambda l: l[1])                       # top first, so lower leaves overlap them like shingles
    ss = 3                                                 # supersampling for smooth edges
    for cx, cy, R, rot in leaves:
        pts = [(cx + math.cos(t) * R * (1 + .16 * math.cos(7 * (t - rot))), cy + math.sin(t) * R * (1 + .16 * math.cos(7 * (t - rot))))
               for t in np.linspace(0, 2 * math.pi, 72, endpoint=False)]
        m = Image.new("L", (CELL * ss, CELL * ss), 0)
        ImageDraw.Draw(m).polygon([(x * ss, y * ss) for x, y in pts], fill=255)
        m = np.asarray(m.resize((CELL, CELL), Image.LANCZOS)).astype(float) / 255
        shadow = np.roll(ndi.gaussian_filter(m, 2.0), 4, axis=0)             # soft contact shadow on the leaves underneath
        rgb *= (1 - .38 * shadow * (alpha > 0))[..., None]
        shift = np.roll(tex, (rnd.randrange(CELL), rnd.randrange(CELL)), axis=(0, 1)) * rnd.uniform(.9, 1.12)
        rgb = rgb * (1 - m[..., None]) + shift * m[..., None]
        alpha = np.maximum(alpha, m)
    light = np.linspace(1.26, .92, CELL)[:, None, None]                      # lit from above
    rgb = np.clip(rgb * light, 0, 255)
    return np.dstack([rgb, alpha * 255]).astype(np.uint8)


def needle_card(swatch, seed):
    """One 256x256 RGBA card for a pine branch skirt: three overlapping drooping tufts with spiky needle tips."""
    rnd = random.Random(seed)
    tex = np.asarray(swatch.resize((CELL, CELL), Image.LANCZOS)).astype(float)
    rgb = np.empty_like(tex)
    rgb[:] = tex.mean(axis=(0, 1))
    alpha = np.zeros((CELL, CELL))
    ss = 3
    for y0, width, drop in ((62, 214, 150), (104, 188, 122), (146, 156, 92)):         # top tuft first, lower ones overlap it
        cx = CELL / 2 + rnd.uniform(-6, 6)
        teeth = 11
        pts = [(cx - width * .3 + width * .6 * t / 16, y0 - 14 * math.sin(math.pi * t / 16)) for t in range(17)]      # narrow curved top edge (fan shape)
        for i in range(teeth * 2 + 1):                                                                              # spiky bottom edge, right to left
            fx = cx + width / 2 - width * i / (teeth * 2)
            tip = i % 2 == 0
            pts.append((fx + rnd.uniform(-3, 3), y0 + drop * (rnd.uniform(.85, 1.15) if tip else rnd.uniform(.35, .55))))
        m = Image.new("L", (CELL * ss, CELL * ss), 0)
        ImageDraw.Draw(m).polygon([(x * ss, y * ss) for x, y in pts], fill=255)
        m = np.asarray(m.resize((CELL, CELL), Image.LANCZOS)).astype(float) / 255
        shadow = np.roll(ndi.gaussian_filter(m, 2.5), 6, axis=0)
        rgb *= (1 - .45 * shadow * (alpha > 0))[..., None]
        shift = np.roll(tex, (rnd.randrange(CELL), rnd.randrange(CELL)), axis=(0, 1)) * rnd.uniform(.92, 1.12)
        rgb = rgb * (1 - m[..., None]) + shift * m[..., None]
        alpha = np.maximum(alpha, m)
    light = np.linspace(1.3, .85, CELL)[:, None, None]
    return np.dstack([np.clip(rgb * light, 0, 255), alpha * 255]).astype(np.uint8)


def main():
    src, H, S, V = load()
    out = HERE / "assets"
    out.mkdir(exist_ok=True)
    masks = {
        "green": ((H >= 60) & (H <= 150) & (S > .3) & (V > .25) & (V < .85)),
        "yellow": ((H >= 39) & (H <= 56) & (S > .55) & (V > .75)),
        "orange": ((H >= 22) & (H <= 38) & (S > .6) & (V > .6)),
        "red": ((H >= 6) & (H <= 21) & (S > .55) & (V > .45) & (V < .9)),
    }
    atlas = Image.new("RGBA", (CELL * 4, CELL * 2), (0, 0, 0, 0))
    for col, (name, m) in enumerate(masks.items()):
        wins = best_windows(m, 70, 2)
        print(name, [(x, y, round(c, 2)) for x, y, c in wins])
        for row, (x, y, _) in enumerate(wins):
            swatch = src.crop((x, y, x + 70, y + 70))
            atlas.paste(Image.fromarray(leaf_cluster(swatch, col * 10 + row), "RGBA"), (col * CELL, row * CELL))
    atlas.save(out / "leaves_atlas.png")

    pine = (H >= 135) & (H <= 185) & (S > .2) & (S < .7) & (V > .18) & (V < .55)          # the deep teal-green foliage
    pine_atlas = Image.new("RGBA", (CELL * 2, CELL), (0, 0, 0, 0))
    for col, (x, y, c) in enumerate(best_windows(pine, 64, 2)):
        print("pine", (x, y, round(c, 2)))
        pine_atlas.paste(Image.fromarray(needle_card(src.crop((x, y, x + 64, y + 64)), 70 + col), "RGBA"), (col * CELL, 0))
    pine_atlas.save(out / "pine_atlas.png")

    def tile_from(mask, size, name, final=256, blur=0.0, gain=1.0):
        x, y, c = best_windows(mask, size, 1)[0]
        print(name, (x, y, round(c, 2)))
        img = mirror_tile(src.crop((x, y, x + size, y + size))).resize((final, final), Image.LANCZOS)
        if blur:
            img = img.filter(ImageFilter.GaussianBlur(blur))
        if gain != 1.0:
            img = ImageEnhance.Brightness(img).enhance(gain)
        img.save(out / name)

    tile_from((H >= 12) & (H <= 40) & (S > .12) & (S < .6) & (V > .12) & (V < .5), 64, "bark.png", gain=1.2)
    print("wrote", out)


if __name__ == "__main__":
    main()
