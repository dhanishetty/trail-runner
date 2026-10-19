"""Cuts reference/reference.png into two backdrop layers for the game and writes them to ./assets:

  backdrop_sky.png      the sky, sun and clouds only (the gaps behind mountains/trees are filled in)
  backdrop_scenery.png  everything that is not sky: mountain, lake, distant trees, the framing autumn trees

Run once (or again after changing reference/reference.png):  py make_backdrop.py
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).parent
CROP_H = 525            # keep the part above the far end of the trail; the game draws its own ground below
FADE_BOTTOM = 40        # px: soft fade into the game's horizon haze


def sky_mask(rgb):
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(float)
    hue, sat, val = hsv[..., 0] * 360 / 255, hsv[..., 1], hsv[..., 2]
    bluish = (hue >= 170) & (hue <= 235) & (val > 200) & (sat > 20)       # light blue sky
    whitish = (sat < 60) & (val > 225)                                    # clouds and the sun
    cand = bluish | whitish
    labels, n = ndi.label(cand)
    keep = [i for i in range(1, n + 1) if (labels == i).sum() > 3000 and (labels[0] == i).any()]   # big areas touching the top edge
    mask = np.isin(labels, keep)
    mask = ndi.binary_fill_holes(mask)                                    # cloud shadows are part of the sky
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    return mask


def fill_sky(rgb, mask):
    """Replace non-sky pixels with the colour of the lowest sky pixel in the same column."""
    h, w, _ = rgb.shape
    fill = np.full((w, 3), np.nan)
    for x in range(w):
        ys = np.where(mask[:, x])[0]
        if len(ys):
            fill[x] = rgb[ys.max(), x]
    good = ~np.isnan(fill[:, 0])
    xs = np.arange(w)
    for c in range(3):
        fill[:, c] = np.interp(xs, xs[good], fill[good, c])
    out = np.where(mask[..., None], rgb, fill[None, :, :]).astype(float)
    # soften the filled area so it reads as sky, not as copied columns
    blurred = np.stack([ndi.gaussian_filter(out[..., c], 14) for c in range(3)], axis=-1)
    soft = ndi.gaussian_filter((~mask).astype(float), 3)[..., None]
    return out * (1 - soft) + blurred * soft


def edge_fade(h, w, top=0.0, sides=0.0, bottom=0.0):
    a = np.ones((h, w))
    if top:
        n = int(h * top)
        a[:n] *= np.linspace(0, 1, n)[:, None]
    if sides:
        n = int(w * sides)
        ramp = np.linspace(0, 1, n)
        a[:, :n] *= ramp[None, :]
        a[:, -n:] *= ramp[::-1][None, :]
    if bottom:
        a[-bottom:] *= np.linspace(1, 0, bottom)[:, None]
    return a


def main():
    src = Image.open(HERE / "reference" / "reference.png").convert("RGB")
    rgb = np.asarray(src)[:CROP_H].copy()
    h, w, _ = rgb.shape
    mask = sky_mask(rgb)
    print(f"sky covers {mask.mean() * 100:.0f}% of the crop")

    sky = fill_sky(rgb.astype(float), mask)
    sky_alpha = edge_fade(h, w, top=0.10, sides=0.16, bottom=FADE_BOTTOM)
    sky_img = np.dstack([np.clip(sky, 0, 255), sky_alpha * 255]).astype(np.uint8)

    soft_mask = ndi.gaussian_filter(ndi.binary_erosion(mask, iterations=1).astype(float), 1.3)
    scenery_alpha = np.clip(1 - soft_mask, 0, 1) * edge_fade(h, w, top=0.08, sides=0.16, bottom=FADE_BOTTOM)
    scenery_img = np.dstack([rgb, scenery_alpha * 255]).astype(np.uint8)

    out = HERE / "assets"
    out.mkdir(exist_ok=True)
    Image.fromarray(sky_img, "RGBA").save(out / "backdrop_sky.png")
    Image.fromarray(scenery_img, "RGBA").save(out / "backdrop_scenery.png")
    # a quick preview to eyeball the cut: sky layer + scenery layer composited on grey
    prev = Image.new("RGBA", (w, h), (120, 120, 120, 255))
    prev.alpha_composite(Image.fromarray(sky_img, "RGBA"))
    prev.alpha_composite(Image.fromarray(scenery_img, "RGBA"))
    prev.convert("RGB").save(out / "backdrop_preview.png")
    print("wrote", out)


if __name__ == "__main__":
    main()
