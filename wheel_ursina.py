"""3D rolling ball on a forest trail (Ursina / Panda3D version, real lighting + shadows).
The ball rolls forward on its own; steer with Left/Right (A/D), jump with Space.
Cross the pits, hop over the rocks, reach the gate. R restarts, Esc quits."""
import sys
if getattr(sys, "frozen", False) and "--worker" in sys.argv:        # the packaged .exe starts itself again as the camera process
    import pose_control
    pose_control._worker()
    sys.exit()
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from panda3d.core import Texture as PandaTexture, Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, BitMask32, NodePath, TransparencyAttrib
from ursina import *
from ursina.shader import Shader
from ursina.shaders import lit_with_shadows_shader, unlit_shader
from PIL import Image, ImageChops, ImageDraw, ImageFilter

# ---------------------------------------------------------------- level / physics
TILE = 2                 # tile size
COLS = 3                   # trail is 3 tiles wide, x from -3 to 3
HALF = COLS * TILE / 2
R = 0.5                    # ball radius
FORWARD_START, FORWARD_END = 0.09, 0.125      # per 1/60 s frame: slow at the start, speeding up steadily to the finish gate
GRAVITY, JUMP_V = 0.00333, 0.125
STEER = 0.0625
CAM_PITCH = 7              # degrees the camera looks down
MIN_ASPECT, MAX_ASPECT = 4 / 3, 20 / 9      # picture shapes from an old monitor / tablet (4:3) to a long phone held sideways (20:9); anything else gets bars
DEPTH = 3.5                # how deep the ground blocks go (pit depth)

# What the constants above do (distances are game units, "per frame" = per 1/60 s):
#   TILE       size of one track square; obstacles snap to these squares. Best left alone.
#   COLS       trail width in tiles (3 lanes: left, centre, right). The obstacle layout assumes 3.
#   HALF       half the trail width (calculated); the left / right limit for the ball.
#   R          ball radius; bigger = harder to squeeze past rocks and pit edges.
#   FORWARD_START / FORWARD_END   forward distance per frame = game speed at the start / at the finish (it rises steadily in between).
#   GRAVITY    how fast the ball falls back down; with JUMP_V it sets jump height and length.
#   JUMP_V     take-off speed of a jump. A jump lasts the same time at any speed, so it flies further as the ball speeds up.
#   STEER      sideways speed; higher = more responsive, lower = floatier.
#   CAM_PITCH  degrees the camera looks down; higher shows more track ahead.
#   MIN_ASPECT / MAX_ASPECT   the narrowest / widest picture shape (width / height) the game draws; a window that is wider or taller
#              gets black bars, so the empty edge of the forest is never shown. Laptops and TVs are 16:9 (1.78), 16:10 (1.6).
#   DEPTH      how deep the ground blocks go (how deep the pits are).

def speed_at(z, length):
    """Forward distance per frame at distance z: FORWARD_START at the start, rising evenly to FORWARD_END at the finish."""
    return FORWARD_START + (FORWARD_END - FORWARD_START) * min(1.0, max(0.0, z / length))


def curve(z):
    """Sideways offset of the winding trail at distance z. Only the pictures follow it (the whole static world is bent with it,
    and the panda and camera are shifted to match); collisions keep using the straight grid. Starts straight, then meanders."""
    z = np.asarray(z, dtype=float)
    k = np.clip(z / 24.0, 0.0, 1.0)
    return (k * k * (3 - 2 * k)) * (3.0 * np.sin(z * 0.045) + 1.4 * np.sin(z * 0.11 + 1.0))


def curve_slope(z):
    return (curve(z + .5) - curve(z - .5)) / 1.0


class Pit:
    """A freeform muddy hollow. One irregular outline decides both where you fall in and how the ground is shaped."""

    def __init__(self, cx, cz, rx, rz, seed):
        self.cx, self.cz, self.rx, self.rz = cx, cz, rx, rz
        r = random.Random(seed)
        self.ph = [r.uniform(0, 6.28) for _ in range(3)]

    def edge(self, th):
        return 1 + .12 * np.sin(2 * th + self.ph[0]) + .08 * np.sin(3 * th + self.ph[1]) + .05 * np.sin(5 * th + self.ph[2])

    def s(self, x, z):
        """1 at the centre, 0 on the outline, negative outside (numpy friendly)."""
        u, v = (x - self.cx) / self.rx, (z - self.cz) / self.rz
        return 1 - np.sqrt(u * u + v * v) / self.edge(np.arctan2(v, u))

    def point(self, th, k=1.0):
        e = self.edge(th) * k
        return self.cx + self.rx * e * math.cos(th), self.cz + self.rz * e * math.sin(th)


TRACK_FILE = (Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent) / "track_layout.txt"      # edit this file to arrange the obstacles yourself
DEFAULT_LENGTH = 540.0
LANES = {"left": 0, "center": 1, "right": 2}


def generate_events(length=DEFAULT_LENGTH, seed=11):
    """The default for track_layout.txt. Boulders and side pits each cover two lanes (2/3 of the trail width) and leave one lane
    open; full-width pits cover all of it. The kinds are balanced: they are dealt like a shuffled hand of cards, so each comes up
    equally often but in a random-looking order, and the same kind never comes twice in a row. Boulders and side pits alternate
    left, right, left, ... (the open lane swaps sides), so no two in a row are on the same side; full-width pits can come anywhere.
    The gaps grow with the ball's speed, so there is always the same time to land and line up. Returns (events, length)."""
    r = random.Random(seed)
    kinds = ["rock", "pit_full", "pit_side"]
    ev, z, side, last, hand = [], 24.0, "left", None, []
    while True:
        if not hand:
            hand = kinds[:]
            r.shuffle(hand)
            while hand[-1] == last:                                # a new hand never starts with the kind just played
                r.shuffle(hand)
        kind = hand.pop()
        last = kind
        if kind == "pit_full":
            size = round(r.uniform(1.4, 1.6), 1)
            item, span = (z, "pit_full", {"size": size}), 2 * size
        else:
            if kind == "rock":
                item, span = (z, "rock", {"lanes": "left+center" if side == "left" else "center+right"}), 2.0
            else:
                size = round(r.uniform(1.4, 1.7), 1)
                item, span = (z, "pit_side", {"side": side, "size": size}), 2 * size
            side = "right" if side == "left" else "left"
        if z + span > length - 20:
            break
        ev.append(item)
        z += span + speed_at(z, length) * 60 * 2.5                 # about 2.5 s to the next one
    return ev, length


HEADER = """# TRACK LAYOUT - edit this file to arrange the obstacles, then start the game again.
#
# One obstacle per line:   <kind> <distance> [options]
# <distance> is how far from the start (in game units; the first obstacle is best placed after 15 or so).
# Lines starting with # are notes and are ignored. Delete this file to get a freshly generated layout.
#
#   length 540                     how long the whole track is (the finish gate is at this distance)
#   pit_full   <z> [size=1.7]      a muddy pit across the whole track. Jump it. size = half its length (1.3 - 2.2)
#   pit_side   <z> side=left|right [size=1.7]   a pit over two lanes (2/3 of the width) on one side, one lane open. Jump it, or steer round it
#   pit_mid    <z> [size=1.8]      a narrow pit in the middle. Steer round it, or jump
#   rock       <z> lanes=left|center|right   a boulder in that lane (several with + make a wider wall)
#   rocks_gap  <z> gap=left|center|right   a wall of boulders with one open lane
#   slalom     <z> side=left|right  three boulders in a zig-zag over about 14 units
#
# Rules used by the ready-made layout: boulders cover two lanes (2/3 of the width), side pits too, and the open lane alternates sides
# left, right, left ... The ball speeds up as it goes, so leave more room between obstacles towards the end (about 13 units at the
# start, 19 at the finish).
"""


def events_to_text(events, length):
    out = [HEADER, f"length {int(length)}", ""]
    for z, kind, opt in sorted(events, key=lambda e: e[0]):
        out.append(f"{kind:10s} {z:6.1f}  " + " ".join(f"{k}={v}" for k, v in opt.items()))
    return "\n".join(out) + "\n"


def parse_events(text):
    """Read track_layout.txt. Raises ValueError with the line number when something is wrong."""
    ev, length = [], DEFAULT_LENGTH
    kinds = ("pit_full", "pit_side", "pit_mid", "rock", "rocks_gap", "slalom")
    for n, line in enumerate(text.splitlines(), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        try:
            if parts[0] == "length":
                length = float(parts[1])
                continue
            if parts[0] not in kinds:
                raise ValueError(f"unknown obstacle '{parts[0]}'")
            z = float(parts[1])
            opt = dict(q.split("=", 1) for q in parts[2:])
            if "side" in opt and opt["side"] not in ("left", "right"):
                raise ValueError("side must be left or right")
            if "gap" in opt and opt["gap"] not in LANES:
                raise ValueError("gap must be left, center or right")
            if "lanes" in opt and any(c not in LANES for c in opt["lanes"].split("+")):
                raise ValueError("lanes must be left, center, right (join with +)")
            ev.append((z, parts[0], opt))
        except (IndexError, ValueError) as e:
            raise ValueError(f"track_layout.txt line {n}: {e}")
    return ev, length


def build_from_events(events, length):
    finish_row = int(length // TILE)
    rows = [["#", "#", "#"] for _ in range(finish_row + 8)]
    pits = []

    def cell(z, c):
        i = int(z // TILE)
        if 0 <= i < finish_row and 0 <= c < 3:
            rows[i][c] = "B"
    for k, (z, kind, o) in enumerate(sorted(events, key=lambda e: e[0])):
        if z > length - 6:
            continue
        size = float(o.get("size", 1.7))
        if kind == "pit_full":
            pits.append(Pit(0, z, 3.8, size, 100 + k))
        elif kind == "pit_side":
            side = -1 if o.get("side", "left") == "left" else 1
            pits.append(Pit(side * 1.0, z, 2.0, size, 100 + k))
        elif kind == "pit_mid":
            pits.append(Pit(0, z, 1.45, size, 100 + k))
        elif kind == "rock":
            for name in o.get("lanes", "center").split("+"):
                cell(z, LANES[name])
        elif kind == "rocks_gap":
            gap = LANES[o.get("gap", "center")]
            for c in range(3):
                if c != gap:
                    cell(z, c)
        elif kind == "slalom":
            first = LANES[o.get("side", "left")]
            cell(z, first); cell(z + 6, 2 - first); cell(z + 12, first)
    rows[finish_row] = ["F", "F", "F"]
    return ["".join(c) for c in rows], pits, finish_row


def load_track():
    """The obstacle layout comes from track_layout.txt (created on first run - edit it to arrange the track yourself)."""
    try:
        if TRACK_FILE.exists():
            events, length = parse_events(TRACK_FILE.read_text(encoding="utf-8"))
        else:
            events, length = generate_events()
            TRACK_FILE.write_text(events_to_text(events, length), encoding="utf-8")
    except Exception as e:                                         # bad file: say so, and fall back to a generated layout
        print(f"[track] {e} - using a generated layout instead")
        events, length = generate_events()
    return build_from_events(events, length)


LEVEL, PITS, FINISH_ROW = load_track()
TRACK_LENGTH = FINISH_ROW * TILE


def near_pit(x, z0, z1=None, margin=.35):
    """True if (x, z) - or any of 5 points along z0..z1 - is on or within `margin` of a pit's outline."""
    zlo, zhi = (z0, z0) if z1 is None else (z0, z1)
    for q in PITS:
        if abs(x - q.cx) > q.rx * 1.35 + margin + .3 or zhi < q.cz - q.rz * 1.35 - margin - .3 or zlo > q.cz + q.rz * 1.35 + margin + .3:
            continue                                           # far away: skip the maths
        zs = np.array([z0]) if z1 is None else np.linspace(z0, z1, 5)
        if float(np.max(q.s(x, zs))) > -margin:
            return True
    return False


ROW_MIN, ROW_MAX = -8, len(LEVEL) + 16     # rows that get built (trail continues past both ends)
FOG = color.rgb32(186, 216, 222)          # hazy blue-green horizon (matches the painted backdrop)


try:
    from pose_control import PoseCamera, STATUS_TEXT
except Exception:                                  # OpenCV / MediaPipe not installed: keyboard only
    PoseCamera = None
    STATUS_TEXT = {}
active_camera = [None]                             # the PoseCamera when the player chose camera control


# ---- Input layer: the game only sees actions (left / right / jump). Sources: the keyboard, and the body-tracking camera. ----
def keyboard_actions():
    return {
        "left": held_keys["left arrow"] or held_keys["a"],
        "right": held_keys["right arrow"] or held_keys["d"],
        "jump": held_keys["space"] or held_keys["up arrow"] or held_keys["w"],
    }


def camera_actions():
    cam = active_camera[0]
    return cam.mapper.actions() if cam is not None else {"left": 0, "right": 0, "jump": 0, "target": None}


def get_actions():
    k, c = keyboard_actions(), camera_actions()
    act = {key: max(k[key], c[key]) for key in k}         # the keyboard still works in camera mode, handy as a backup
    act["target"] = None if (k["left"] or k["right"]) else c["target"]
    return act


def tile_at(x, z):
    for q in PITS:
        if abs(z - q.cz) < q.rz * 1.4 and abs(x - q.cx) < q.rx * 1.4 and q.s(x, z) > 0:
            return "."
    row, col = int(z // TILE), min(COLS - 1, max(0, int((x + HALF) // TILE)))
    if z < 0 or row >= len(LEVEL):
        return "#"
    return LEVEL[row][col]


def rock_hit(x, z):
    """Is a boulder under the ball? Checked across a width a little narrower than the ball, so only a real head-on hit counts and
    brushing past the edge of a boulder does not."""
    e = R * .7
    return tile_at(x - e, z) == "B" or tile_at(x + e, z) == "B" or tile_at(x, z) == "B"


class BallState:
    """Pure game logic (no graphics). update() returns a list of events for effects."""

    def __init__(self):
        self.x, self.y, self.z = 0.0, 0.0, 1.0   # y = height of the ball's bottom above ground
        self.vx = self.vy = 0.0
        self.roll = 0.0                           # radians
        self.state = "play"                       # play / fell / crashed / win
        self.air = False

    def update(self, act, f=1.0):
        ev = []
        if self.state != "play":
            if self.state == "fell":
                self.vy -= GRAVITY * f
                self.y += self.vy * f
            return ev
        target = act.get("target")
        if target is None:
            want = (act["right"] - act["left"]) * STEER                     # keys: hold to move sideways
        else:                                                               # camera: the ball goes to where your body is
            want = max(-STEER, min(STEER, (target * (HALF - R) - self.x) * 0.3))
        self.vx += (want - self.vx) * (1 - 0.75 ** f)
        nx = max(-HALF + R, min(HALF - R, self.x + self.vx * f))
        low = self.y < 1.0 - 0.05                                  # below the top of a boulder
        if low and rock_hit(nx, self.z) and not rock_hit(self.x, self.z):
            nx, self.vx = self.x, 0.0                              # pushing sideways into a boulder just stops you: no crash, no bounce
        self.x = nx
        v = speed_at(self.z, TRACK_LENGTH)
        self.z += v * f
        self.roll += v * f / R

        t = tile_at(self.x, self.z)
        grounded = False
        if low and rock_hit(self.x, self.z):                       # only running into a boulder head-on ends the run
            self.state = "crashed"
            return ["crash"]
        if t in "#F":
            if self.y <= 0 and self.vy <= 0:
                self.y, self.vy, grounded = 0.0, 0.0, True
        elif t == "B" and self.y <= 1.0 and self.vy <= 0:
            self.y, self.vy, grounded = 1.0, 0.0, True

        if grounded:
            if self.air:
                self.air = False
                ev.append("land")
            ev.append("trail")
            if act["jump"]:
                self.vy = JUMP_V
                ev.append("jump")
        else:
            self.air = True
            self.vy -= GRAVITY * f
        self.y += self.vy * f
        if self.y < -.55:                                  # sank into the puddle
            self.state = "fell"
            ev.append("fell")
        elif self.z // TILE >= FINISH_ROW and self.y <= 0.01:
            self.state = "win"
            ev.append("win")
        return ev


# ---------------------------------------------------------------- procedural textures
def _tex(img):
    t = Texture(img)
    t.filtering = "mipmap"
    return t


def _blob(d, S, cx, cy, r, col, rnd):
    """A flat, irregular patch of colour, drawn wrapped so the texture tiles."""
    ph = [rnd.uniform(0, 6.28) for _ in range(3)]
    pts = [(cx + math.cos(a) * r * (1 + .22 * math.sin(2 * a + ph[0]) + .12 * math.sin(3 * a + ph[1]) + .07 * math.sin(5 * a + ph[2])),
            cy + math.sin(a) * r * .8 * (1 + .22 * math.sin(2 * a + ph[0]) + .12 * math.sin(3 * a + ph[1]))) for a in np.linspace(0, 2 * math.pi, 18, endpoint=False)]
    for ox in (-S, 0, S):
        for oy in (-S, 0, S):
            d.polygon([(px + ox, py + oy) for px, py in pts], fill=col)


def make_dirt_texture(S=256):
    """Clean cartoon path: warm tan with flat lighter and darker patches, outlined pebbles and a few light flecks."""
    rnd = random.Random(3)
    img = Image.new("RGB", (S, S), (208, 146, 88))
    d = ImageDraw.Draw(img)
    for _ in range(9):
        _blob(d, S, rnd.uniform(0, S), rnd.uniform(0, S), rnd.uniform(34, 62), (224, 166, 102), rnd)
    for _ in range(8):
        _blob(d, S, rnd.uniform(0, S), rnd.uniform(0, S), rnd.uniform(26, 50), (186, 124, 74), rnd)
    for _ in range(26):                                      # small light flecks
        _blob(d, S, rnd.uniform(0, S), rnd.uniform(0, S), rnd.uniform(3, 7), (232, 178, 116), rnd)
    for _ in range(12):                                      # pebbles: dark shadow, body, highlight
        x, y, r = rnd.uniform(0, S), rnd.uniform(0, S), rnd.uniform(5, 10)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                cx, cy = x + ox, y + oy
                d.ellipse((cx - r * 1.1, cy - r * .62 + 2, cx + r * 1.1, cy + r * .72 + 2), fill=(120, 84, 58))
                d.ellipse((cx - r, cy - r * .62, cx + r, cy + r * .62), fill=rnd.choice(((176, 150, 126), (156, 128, 104), (196, 168, 140))))
                d.ellipse((cx - r * .6, cy - r * .48, cx + r * .2, cy - r * .08), fill=(214, 192, 166))
    return _tex(img)


def make_grass_texture(S=256):
    """Clean cartoon meadow: yellow-green with flat lighter / darker patches and little blade tufts."""
    rnd = random.Random(4)
    img = Image.new("RGB", (S, S), (126, 176, 62))
    d = ImageDraw.Draw(img)
    for _ in range(10):
        _blob(d, S, rnd.uniform(0, S), rnd.uniform(0, S), rnd.uniform(34, 64), (156, 202, 78), rnd)
    for _ in range(9):
        _blob(d, S, rnd.uniform(0, S), rnd.uniform(0, S), rnd.uniform(26, 52), (98, 152, 54), rnd)
    for _ in range(60):                                      # blade tufts: three little pointed blades each
        x, y = rnd.uniform(0, S), rnd.uniform(0, S)
        col = rnd.choice(((76, 128, 46), (184, 216, 98), (92, 148, 52)))
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                for k in (-1, 0, 1):
                    bx = x + ox + k * 5
                    h = rnd.uniform(9, 15)
                    d.polygon([(bx - 2, y + oy), (bx + 2, y + oy), (bx + k * 4, y + oy - h)], fill=col)
    return _tex(img)


def make_sky_texture(W=4096, H=1024, sun=True, scenery=True):
    """360-degree backdrop painted after the reference image: blue sky + sun + puffy clouds,
    blue mountains, a lake, and hazy forest ridges dotted with green/yellow/orange/red autumn trees.
    Sky-dome mapping (measured): u = azimuth/360 clockwise from straight ahead (+z); the image bottom is the horizon
    and row = (1 - elevation/79) * H. scenery=False leaves out the mountains / lake / ridges (the painted backdrop has its own; the dome's
    ridges used to show as a thin coloured strip along the horizon)."""
    ppx, ppy = W / 360.0, H / 79.0                       # pixels per degree (horizontal, vertical)

    def row(el):
        return (1 - el / 79.0) * H

    def X(az):
        return (az % 360) / 360.0 * W

    el = (1 - np.arange(H) / H) * 79
    xs = (0, 6, 20, 45, 79)
    cols = ((190, 218, 232), (150, 200, 240), (100, 172, 240), (60, 140, 232), (40, 118, 224))
    grad = np.stack([np.interp(el, xs, [c[k] for c in cols]) for k in range(3)], axis=1)
    img = np.repeat(grad[:, None, :], W, axis=1)

    sun_az, sun_el = 20.0, 23.0                          # upper right of the view, like the reference
    sx, sy = X(sun_az), row(sun_el)
    y0, y1 = max(0, int(sy - 30 * ppy)), min(H, int(sy + 30 * ppy))
    yy, xx = np.mgrid[y0:y1, 0:W]
    dist = np.hypot(((xx - sx + W / 2) % W - W / 2) / ppx, (yy - sy) / ppy)      # wrap-safe angular distance
    g = np.clip(0.55 * np.exp(-dist / 5.5) + (dist < 2.4) * 1.0, 0, 1)[..., None] * (1.0 if sun else 0.0)
    img[y0:y1] = img[y0:y1] * (1 - g) + np.array((255, 250, 228)) * g
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    rnd = random.Random(33)

    # ---- puffy cumulus clouds with soft blue-grey undersides
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    clouds = [(-22, 24, 1.5), (8, 33, .8), (30, 14, .8), (40, 8, .6), (-48, 16, 1.1), (70, 22, 1.2), (120, 30, 1.4), (165, 14, 1.0),
              (205, 34, 1.3), (250, 20, 1.2), (300, 28, 1.5), (335, 12, .9)]
    for caz, cel, sc in clouds:
        puffs = [(rnd.uniform(-7, 7) * sc, rnd.uniform(-.2, .9) * sc, rnd.uniform(1.6, 3.0) * sc) for _ in range(7)]
        for off in (-W, 0, W):
            for dx, dy, r in puffs:
                cx, cy = X(caz) + off + dx * ppx, row(cel + dy)
                d.ellipse((cx - r * 1.5 * ppx, cy - r * 0.95 * ppy, cx + r * 1.5 * ppx, cy + r * 0.95 * ppy), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(2.2))
    edge = ImageChops.subtract(mask, ImageChops.offset(mask, 0, -int(1.3 * ppy))).point(lambda v: min(255, int(v * 1.2)))
    im.paste(Image.new("RGB", (W, H), (255, 255, 255)), mask=mask)
    im.paste(Image.new("RGB", (W, H), (176, 198, 232)), mask=edge.filter(ImageFilter.GaussianBlur(2)).point(lambda v: int(v * .75)))

    d = ImageDraw.Draw(im)
    haze = (186, 216, 222)

    def mix(c, f):
        return tuple(int(v + (h - v) * f) for v, h in zip(c, haze))

    def ridge(base, amp, harmonics, extra=None):
        out = []
        for x in range(0, W + 8, 8):
            az = x / W * 360
            m = .5 + sum(w * math.sin(math.radians(k * az) + ph) for k, w, ph in harmonics)
            e = base + amp * max(0.0, m) ** 1.2
            if extra:
                e += extra(az)
            out.append((x, row(e)))
        return out

    if scenery:
        # blue mountains (one big peak just left of straight ahead)
        mt = ridge(.6, 3.0, ((3, .28, .7), (5, .16, 2.0), (9, .08, 1.1), (13, .05, .3)),
                   lambda az: 2.4 * math.exp(-((((az + 180) % 360) - 180 + 6) / 7.5) ** 2))
        d.polygon(mt + [(W, H), (0, H)], fill=mix((92, 130, 184), .18))
        mt2 = ridge(.5, 2.0, ((4, .3, 2.4), (7, .15, .4), (11, .07, 1.9)),
                    lambda az: 1.5 * math.exp(-((((az + 180) % 360) - 180 - 16) / 9) ** 2))
        d.polygon(mt2 + [(W, H), (0, H)], fill=mix((70, 118, 160), .28))

        # lake, visible in a gap ahead
        lake_el0, lake_el1 = .45, 2.6
        d.polygon([(X(az), row(lake_el1 - .5 * math.sin(math.radians((az + 10) * 4.5)) ** 2 * 0)) for az in (-12, 30)] +
                  [(X(30), row(lake_el0)), (X(-12), row(lake_el0))], fill=(78, 152, 206))
        d.line([(X(-4), row(1.3)), (X(24), row(1.3))], fill=(150, 200, 232), width=3)

        # forested ridges dotted with autumn crowns
        def forest(base, amp, harmonics, dark, gap_w, seed):
            r = random.Random(seed)
            gap = lambda az: -gap_w * math.exp(-((((az + 180) % 360) - 180 - 10) / 18) ** 2)
            prof = ridge(base, amp, harmonics, gap)
            d.polygon(prof + [(W, H), (0, H)], fill=mix(dark, .22))
            palette = ((74, 142, 84), (74, 142, 84), (60, 120, 92), (118, 164, 62), (236, 172, 52), (240, 124, 44), (208, 74, 42), (250, 206, 76))
            for x, y in prof:
                for _ in range(2):
                    rad = r.uniform(.45, 1.05) * ppx
                    cx, cy = x + r.uniform(-4, 4), y + r.uniform(-.2, 1.1) * ppy
                    col = mix(r.choice(palette), .2)
                    d.ellipse((cx - rad, cy - rad * 1.05, cx + rad, cy + rad * 1.05), fill=col)
                if r.random() < .22:                                         # little conifers on the skyline
                    h = r.uniform(.8, 1.6) * ppy
                    d.polygon([(x - .35 * ppx, y + 6), (x + .35 * ppx, y + 6), (x, y - h)], fill=mix((44, 100, 76), .2))

        forest(.8, 2.4, ((3, .24, 1.2), (6, .14, .2), (10, .08, 2.6)), (70, 126, 98), 2.2, 5)
        forest(.3, 1.4, ((4, .26, 2.9), (8, .14, 1.0), (13, .07, .4)), (54, 112, 80), 1.0, 9)

    d.rectangle((0, row(.7), W, H), fill=haze)                          # hazy horizon band (matches the fog)
    return _tex(im.filter(ImageFilter.GaussianBlur(.7)))



def c3(r, g, b):
    return color.rgb32(r, g, b)


# ---------------------------------------------------------------- painted backdrop (layers cut from reference.png)
BACKDROP_HALF_FOV = 33          # each layer spans +-33 degrees ahead of the player (whole painting fits above the horizon)
ASSETS = Path(__file__).with_name("assets")


backdrop_rig = None


backdrop_layers = []           # (entity, parallax factor) pairs, shifted a little as the player moves sideways
cloud_layer = None


def make_cloud_texture(W=1024, H=256):
    """Soft puffy storybook clouds (white tops, pale blue-grey bellies), tileable sideways so they can drift forever."""
    rnd = random.Random(8)
    top = Image.new("L", (W, H), 0)
    belly = Image.new("L", (W, H), 0)
    dt, db = ImageDraw.Draw(top), ImageDraw.Draw(belly)
    for _ in range(7):
        cx, cy, sc = rnd.uniform(0, W), rnd.uniform(H * .3, H * .75), rnd.uniform(.7, 1.3)
        puffs = [(rnd.uniform(-70, 70) * sc, rnd.uniform(-14, 8) * sc, rnd.uniform(18, 32) * sc) for _ in range(7)]
        for off in (-W, 0, W):
            for dx, dy, r in puffs:
                dt.ellipse((cx + off + dx - r * 1.4, cy + dy - r * .8, cx + off + dx + r * 1.4, cy + dy + r * .8), fill=255)
                db.ellipse((cx + off + dx - r * 1.4, cy + dy - r * .55 + 5, cx + off + dx + r * 1.4, cy + dy + r * .95 + 5), fill=255)
    top = top.filter(ImageFilter.GaussianBlur(2.5))
    belly = belly.filter(ImageFilter.GaussianBlur(2.5))
    img = Image.new("RGBA", (W, H), (176, 198, 232, 0))
    img.paste(Image.new("RGBA", (W, H), (176, 198, 232, 255)), mask=belly)
    img.paste(Image.new("RGBA", (W, H), (255, 255, 255, 255)), mask=top)
    img.putalpha(ImageChops.lighter(top, belly).point(lambda v: int(v * .92)))
    return img


def make_haze_texture(H=128):
    """Vertical fade: fog colour at the horizon line fading to transparent upward, so the painting melts into the ground haze."""
    img = Image.new("RGBA", (4, H))
    for y in range(H):
        t = y / (H - 1)                                    # 0 = top, 1 = horizon
        a = int(255 * t ** 1.6 * .95)
        for x in range(4):
            img.putpixel((x, y), (186, 216, 222, a))
    return img


def build_backdrop():
    """Layers in front of the sky dome, all fixed in the world direction: painted sky, drifting clouds, the painted mountains / lake /
    trees (with transparent sky), and a haze band at the horizon. Made by make_backdrop.py. Returns False if the files are missing."""
    global backdrop_rig, cloud_layer
    names = (("backdrop_sky.png", 140, 5, .5), ("backdrop_scenery.png", 130, 7, 1.6))
    if not all((ASSETS / n).exists() for n, _, _, _ in names):
        return False
    backdrop_rig = Entity(parent=camera)           # turned against the camera each frame so the painting stays put

    def layer(img, dist, order, par, tex_repeat=False, h_scale=None):
        w = 2 * dist * math.tan(math.radians(BACKDROP_HALF_FOV))
        h = h_scale or w * img.height / img.width
        y = dist * math.tan(math.radians(CAM_PITCH)) + h / 2 - dist * math.tan(math.radians(.8))
        e = Entity(parent=backdrop_rig, model="quad", texture=_tex(img), position=(0, y, dist), scale=(w, h), shader=unlit_shader,
                   double_sided=True)
        e.set_bin("background", order)             # after the sky dome, before the 3D world
        e.set_depth_test(False)
        e.set_depth_write(False)
        e.set_transparency(TransparencyAttrib.M_alpha)
        e.base_x = 0.0
        backdrop_layers.append((e, par))
        return e, w, h, y

    layer_imgs = {n: Image.open(ASSETS / n).convert("RGBA") for n, _, _, _ in names}
    layer(layer_imgs["backdrop_sky.png"], 140, 5, .5)
    cl, w, h, y = layer(make_cloud_texture(), 136, 6, .8, h_scale=None)
    cl.scale = (w * 1.0, w * .22)
    cl.y = y + w * .1
    cloud_layer = cl
    layer(layer_imgs["backdrop_scenery.png"], 130, 7, 1.6)
    hz, w, h, y = layer(make_haze_texture(), 125, 8, 0.0)
    hz.scale = (w * 1.3, 24)
    hz.y = 125 * math.tan(math.radians(CAM_PITCH)) - 125 * math.tan(math.radians(.8)) + 12
    return True


# ---------------------------------------------------------------- fast static-mesh builder
class Tpl:
    """A primitive as flat (non-indexed) numpy arrays: positions, normals, uvs."""

    def __init__(self, V, N, UV, C=None):
        self.V, self.N, self.UV = (np.array(a, dtype=float) for a in (V, N, UV))
        self.C = None if C is None else np.array(C, dtype=float)       # optional rgb per vertex (0..1), multiplied by the colour given to add()


def _cube():
    V, N, UV = [], [], []
    faces = (((0, 1, 0), (1, 0, 0), (0, 0, 1)), ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
             ((1, 0, 0), (0, 0, 1), (0, 1, 0)), ((-1, 0, 0), (0, 0, -1), (0, 1, 0)),
             ((0, 0, 1), (-1, 0, 0), (0, 1, 0)), ((0, 0, -1), (1, 0, 0), (0, 1, 0)))
    uvs = ((0, 0), (1, 0), (1, 1), (0, 1))
    for n, u, v in faces:
        n, u, v = (np.array(q, dtype=float) for q in (n, u, v))
        c = n * .5
        cor = (c - .5 * u - .5 * v, c + .5 * u - .5 * v, c + .5 * u + .5 * v, c - .5 * u + .5 * v)
        for i in (0, 1, 2, 0, 2, 3):
            V.append(cor[i]); N.append(n); UV.append(uvs[i])
    return Tpl(V, N, UV)


def _plane():
    cor = ((-.5, 0, -.5), (.5, 0, -.5), (.5, 0, .5), (-.5, 0, .5))
    uvs = ((0, 0), (1, 0), (1, 1), (0, 1))
    idx = (0, 3, 2, 0, 2, 1)
    return Tpl([cor[i] for i in idx], [(0, 1, 0)] * 6, [uvs[i] for i in idx])


def _cone(res=8):                      # base on y=0 (radius .5), apex at y=1
    V, N, UV = [], [], []
    for k in range(res):
        a0, a1 = 2 * math.pi * k / res, 2 * math.pi * (k + 1) / res
        am = (a0 + a1) / 2
        for a, p in ((a0, (.5 * math.cos(a0), 0, .5 * math.sin(a0))), (a1, (.5 * math.cos(a1), 0, .5 * math.sin(a1))),
                     (am, (0, 1, 0))):
            V.append(p); N.append((math.cos(a), .5, math.sin(a))); UV.append((0, 0))
    return Tpl(V, N, UV)


def _cylinder(res=8):                  # y from 0 to 1, radius .5, no caps
    V, N, UV = [], [], []
    for k in range(res):
        a0, a1 = 2 * math.pi * k / res, 2 * math.pi * (k + 1) / res
        for a, y in ((a0, 0), (a1, 0), (a1, 1), (a0, 0), (a1, 1), (a0, 1)):
            V.append((.5 * math.cos(a), y, .5 * math.sin(a))); N.append((math.cos(a), 0, math.sin(a))); UV.append((a / (2 * math.pi), y))
    return Tpl(V, N, UV)


def _icosphere():
    m = load_model("icosphere")
    V = np.array([tuple(v) for v in m.vertices], dtype=float)
    N = np.array([tuple(n) for n in m.normals], dtype=float)
    tris = m.triangles
    if tris:
        idx = []
        for t in tris:
            idx += [t] if isinstance(t, int) else list(t)
        V, N = V[idx], N[idx]
    return Tpl(V, N, np.zeros((len(V), 2)))


def _card():                           # upright 1x1 quad facing +z (foliage cards)
    cor = ((-.5, -.5, 0), (.5, -.5, 0), (.5, .5, 0), (-.5, .5, 0))
    uvs = ((0, 0), (1, 0), (1, 1), (0, 1))
    idx = (0, 1, 2, 0, 2, 3)
    return Tpl([cor[i] for i in idx], [(0, 0, 1)] * 6, [uvs[i] for i in idx])


def _orient(d, roll=0.0):
    """Rotation matrix that turns +z towards direction d (then spins the card by `roll` degrees). Plain maths: this runs ~60k times."""
    dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    zx, zy, zz = dx / n, dy / n, dz / n
    ux, uy, uz = (0.0, 1.0, 0.0) if abs(zy) < .95 else (1.0, 0.0, 0.0)
    xx, xy, xz = uy * zz - uz * zy, uz * zx - ux * zz, ux * zy - uy * zx          # up x z
    m = math.sqrt(xx * xx + xy * xy + xz * xz)
    xx, xy, xz = xx / m, xy / m, xz / m
    yx, yy, yz = zy * xz - zz * xy, zz * xx - zx * xz, zx * xy - zy * xx          # z x x
    c, sn = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    return np.array(((xx * c + yx * sn, -xx * sn + yx * c, zx),
                     (xy * c + yy * sn, -xy * sn + yy * c, zy),
                     (xz * c + yz * sn, -xz * sn + yz * c, zz)))


def _disc(seed, seg=24):               # flat, irregular blob on y=0, radius ~.5 (dirt patches, shade, glints)
    rnd = random.Random(seed)
    ph = [rnd.uniform(0, 6.28) for _ in range(3)]
    ang = [2 * math.pi * k / seg for k in range(seg + 1)]
    ring = [(.5 * (1 + .14 * math.sin(2 * a + ph[0]) + .09 * math.sin(3 * a + ph[1]) + .05 * math.sin(5 * a + ph[2])) * math.cos(a), 0,
             .5 * (1 + .14 * math.sin(2 * a + ph[0]) + .09 * math.sin(3 * a + ph[1]) + .05 * math.sin(5 * a + ph[2])) * math.sin(a)) for a in ang]
    V = []
    for k in range(seg):
        V += [(0, 0, 0), ring[k], ring[k + 1]]
    return Tpl(V, [(0, 1, 0)] * len(V), np.zeros((len(V), 2)))


def _sphere(lon=10, lat=6):            # smooth-shaded blob, radius .5
    grid = []
    for i in range(lat + 1):
        phi = math.pi * i / lat
        grid.append([(.5 * math.sin(phi) * math.cos(2 * math.pi * j / lon), .5 * math.cos(phi), .5 * math.sin(phi) * math.sin(2 * math.pi * j / lon))
                     for j in range(lon + 1)])
    V = []
    for i in range(lat):
        for j in range(lon):
            a, b, c, d = grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]
            V += [a, b, c, a, c, d]
    V = np.array(V, dtype=float)
    return Tpl(V, V * 2, np.zeros((len(V), 2)))


def _rot(ry, rz):
    cy, sy = math.cos(math.radians(ry)), math.sin(math.radians(ry))
    cz, sz = math.cos(math.radians(rz)), math.sin(math.radians(rz))
    Ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    Rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return Ry @ Rz


class Builder:
    def __init__(self, texture=None, shader=None, role=None):
        self.texture, self.shader, self.role = texture, shader, role
        self.V, self.N, self.UV, self.C = [], [], [], []

    def add(self, tpl, pos, scale, col, ry=0, rz=0, uv=(1, 1), uv_off=(0, 0), rot=None, normal=None):
        sc = np.array(scale if hasattr(scale, "__len__") else (scale,) * 3, dtype=float)
        rot = _rot(ry, rz) if rot is None else rot
        if normal is None:
            n = (tpl.N / sc) @ rot.T
        else:                                                      # soft "spherical" lighting for foliage cards
            n = np.tile(np.array(normal, dtype=float), (len(tpl.V), 1))
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        self.V.append((tpl.V * sc) @ rot.T + np.array(pos, dtype=float))
        self.N.append(n)
        self.UV.append(tpl.UV * np.array(uv, dtype=float) + np.array(uv_off, dtype=float))
        cols = np.tile(np.array((col.r, col.g, col.b, 1.0)), (len(tpl.V), 1))
        if tpl.C is not None:
            cols[:, :3] *= tpl.C
        self.C.append(cols)

    def add_raw(self, V, N, UV, C):
        """Append ready-made per-vertex arrays (used for the heightfield ground)."""
        self.V.append(V); self.N.append(N); self.UV.append(UV); self.C.append(C)

    def finish(self):
        if not self.V:
            return None
        V, N, UV, C = (np.concatenate(a) for a in (self.V, self.N, self.UV, self.C))
        V[:, 0] += curve(V[:, 2])                  # bend the whole world along the winding trail
        n = len(V)
        data = np.empty(n, dtype=[("v", "<f4", 3), ("n", "<f4", 3), ("c", "u1", 4), ("t", "<f4", 2)])
        data["v"], data["n"], data["t"] = V, N, UV
        data["c"] = np.clip(C * 255, 0, 255).astype(np.uint8)
        vdata = GeomVertexData("static", GeomVertexFormat.get_v3n3c4t2(), Geom.UH_static)   # fast path: no per-vertex Python
        vdata.unclean_set_num_rows(n)
        memoryview(vdata.modify_array(0)).cast("B")[:] = data.tobytes()
        prim = GeomTriangles(Geom.UH_static)
        prim.add_consecutive_vertices(0, n)
        prim.close_primitive()
        geom = Geom(vdata)
        geom.add_primitive(prim)
        node = GeomNode("chunk")
        node.add_geom(geom)
        e = Entity(model=NodePath(node), texture=self.texture, double_sided=True)
        if self.shader is not None:
            e.shader = self.shader
        if self.role == "leaves":
            e.hide(BitMask32.bit(0))          # leaf cards are square: keep them out of the shadow map...
        elif self.role == "shadow":
            e.hide(BitMask32.bit(1))          # ...and let invisible blobs cast the canopy's shadow instead
        elif self.role == "overlay":
            e.set_transparency(TransparencyAttrib.M_alpha)   # grass that fades in over the dirt
            e.set_depth_write(False)
        return e


# Ready-made models: Kenney "Nature Kit" (CC0), unzipped into models/kenney_nature_kit. Plain OBJ files are read here directly;
# the flat material colours are swapped for the painted palette used everywhere else.
MODEL_DIR = Path(__file__).with_name("models") / "kenney_nature_kit"
HAVE_MODELS = (MODEL_DIR / "stone_largeA.obj").exists()
MODEL_COLORS = {"stone": (128, 128, 138), "dirt": (146, 138, 130), "grass": (108, 154, 66), "woodBark": (112, 80, 56),
                "woodInner": (224, 188, 140), "colorRed": (224, 72, 62), "_defaultMat": (238, 228, 208)}
_model_cache = {}


def load_obj(name):
    """Parse MODEL_DIR/<name>.obj (+ .mtl) into a Tpl with per-vertex colours; cached. Returns None if missing."""
    if name in _model_cache:
        return _model_cache[name]
    path = MODEL_DIR / f"{name}.obj"
    if not path.exists():
        _model_cache[name] = None
        return None
    mats, verts, norms, V, N, C = {}, [], [], [], [], []
    mat = "_defaultMat"
    mtl = MODEL_DIR / f"{name}.mtl"
    if mtl.exists():
        cur = None
        for line in mtl.read_text(errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "newmtl":
                cur = parts[1]
            elif len(parts) >= 4 and parts[0] == "Kd" and cur:
                mats[cur] = tuple(int(float(q) * 255) for q in parts[1:4])
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(q) for q in parts[1:4]])
        elif parts[0] == "vn":
            norms.append([float(q) for q in parts[1:4]])
        elif parts[0] == "usemtl":
            mat = parts[1]
        elif parts[0] == "f":
            idx = []
            for tok in parts[1:]:
                f = tok.split("/")
                idx.append((int(f[0]) - 1, int(f[2]) - 1 if len(f) > 2 and f[2] else None))
            col = MODEL_COLORS.get(mat) or mats.get(mat, (200, 200, 200))
            for k in range(1, len(idx) - 1):                      # fan-triangulate
                tri = (idx[0], idx[k], idx[k + 1])
                p3 = np.array([verts[t[0]] for t in tri])
                fn = np.cross(p3[1] - p3[0], p3[2] - p3[0])
                fn = fn / (np.linalg.norm(fn) + 1e-9)
                for vi, ni in tri:
                    V.append(verts[vi])
                    N.append(norms[ni] if ni is not None else fn)
                    C.append([c / 255 for c in col])
    if not V:
        _model_cache[name] = None
        return None
    V = np.array(V)
    tpl = Tpl(V, N, np.zeros((len(V), 2)), C)
    tpl.size = V.max(0) - V.min(0)
    _model_cache[name] = tpl
    return tpl


CUBE, PLANE, CONE, CYL, ICO, SPH, CARD = _cube(), _plane(), _cone(), _cylinder(), _icosphere(), _sphere(), _card()
DISCS = [_disc(k) for k in range(5)]

# painted foliage / bark textures made from reference.png by make_leaves.py (falls back to plain coloured blobs without them)
HAVE_LEAVES = all((ASSETS / f).exists() for f in ("leaves_atlas.png", "bark.png"))
HAVE_NEEDLES = HAVE_LEAVES and (ASSETS / "pine_atlas.png").exists()
foliage_shader = None          # the lit shader plus alpha cut-out, created in create_app()
LEAF_CELL = {"green": 0, "yellow": 1, "orange": 2, "red": 3}


def make_foliage_shader():
    lit = lit_with_shadows_shader
    sh = Shader(language=Shader.GLSL, name="foliage_shader", vertex=lit.vertex,
                fragment=lit.fragment.replace("fragment_color = cast_shadows(fragment_color);",
                                              "if (fragment_color.a < 0.5) discard;\n    fragment_color = cast_shadows(fragment_color);\n    fragment_color.rgb = min(fragment_color.rgb * 1.3, vec3(1.0));"),
                default_input=dict(lit.default_input))
    sh.continuous_input = dict(lit.continuous_input)
    return sh

# Clean cartoon trees (style B, after the vector-style references): three-tone foliage clumps, thick curved trunks with branches,
# stepped ("toon") lighting, in the autumn colours. Each theme is (light, mid, dark).
THEMES = {
    "green": ((156, 200, 78), (98, 156, 58), (58, 112, 54)),
    "yellow": ((255, 228, 112), (242, 192, 62), (204, 148, 44)),
    "orange": ((255, 188, 82), (240, 134, 46), (196, 90, 40)),
    "red": ((240, 122, 80), (206, 72, 46), (150, 46, 42)),
}
THEME_WEIGHTS = (("green", .30), ("yellow", .24), ("orange", .30), ("red", .16))
LEAVES = ((240, 130, 44), (226, 92, 42), (250, 190, 60), (210, 70, 40), (196, 150, 52))
WILD = ((240, 130, 50), (226, 90, 46), (250, 190, 60), (255, 236, 176))
BLADES = ((128, 170, 48), (168, 200, 64), (96, 150, 50), (206, 206, 84))
STONE = ((126, 126, 134), (142, 142, 150), (110, 112, 122))
toon_shader = None             # lit shader with stepped lighting, created in create_app()


def make_toon_shader():
    lit = lit_with_shadows_shader
    old = "float NdotL = clamp(dot(normal_vector, L), 0.001, 1.0);"
    assert old in lit.fragment
    sh = Shader(language=Shader.GLSL, name="toon_shader", vertex=lit.vertex,
                fragment=lit.fragment.replace(old, old + "\n        NdotL = floor(NdotL * 3.0 + 0.999) / 3.0 * 1.7;"),
                default_input=dict(lit.default_input))
    sh.continuous_input = dict(lit.continuous_input)
    return sh


def make_bark_texture(S=256):
    """Cartoon bark: long wobbly vertical strips in three browns with dark cracks and a few knots (tiles in both directions)."""
    rnd = random.Random(12)
    img = Image.new("RGB", (S, S), (118, 80, 54))
    d = ImageDraw.Draw(img)
    for _ in range(46):
        x, w = rnd.uniform(0, S), rnd.uniform(5, 15)
        col = rnd.choice(((146, 100, 66), (98, 66, 44), (130, 90, 60), (164, 118, 78)))
        left = [(x + math.sin(y * .05 + x) * 3, y) for y in range(-8, S + 9, 8)]
        poly = left + [(px + w + math.sin(py * .07 + x) * 2, py) for px, py in reversed(left)]
        for off in (-S, 0, S):
            d.polygon([(px + off, py) for px, py in poly], fill=col)
    for _ in range(16):
        x = rnd.uniform(0, S)
        line = [(x + math.sin(y * .06 + x) * 3, y) for y in range(-8, S + 9, 8)]
        for off in (-S, 0, S):
            d.line([(px + off, py) for px, py in line], fill=(66, 44, 32), width=2)
    for _ in range(5):
        x, y, r = rnd.uniform(20, S - 20), rnd.uniform(20, S - 20), rnd.uniform(5, 9)
        d.ellipse((x - r, y - r * 1.5, x + r, y + r * 1.5), fill=(70, 46, 34))
        d.ellipse((x - r * .55, y - r * .9, x + r * .55, y + r * .9), fill=(104, 70, 48))
    return _tex(img)


toon_leaf_shader = None        # toon lighting plus alpha cut-out for the leaf cards
GRASS_B = [None]               # the card builder tufts are added to (set per row while the world is built)


PLANTS_B = [None]              # the card builder undergrowth is added to (set per row while the world is built)


def make_plant_atlas(CELL=256):
    """Vector undergrowth drawn in code (4 x 2 cells): ferns (green / gold), broad leaves (green / gold), sword-leaf plants
    (teal / olive), flower sprigs (purple / pink-orange). Outlined, with a light edge and a dark rim."""
    ss = 3
    atlas = Image.new("RGBA", (CELL * 4, CELL * 2), (0, 0, 0, 0))
    GREEN = ((150, 210, 92), (82, 158, 60), (40, 102, 48))
    GOLD = ((240, 210, 104), (206, 160, 58), (140, 100, 44))
    TEAL = ((96, 170, 110), (40, 120, 82), (20, 70, 56))
    OLIVE = ((210, 190, 100), (150, 130, 60), (100, 84, 44))

    def lens(base, ang, length, width, bulge=1.0):
        """A pointed leaf shape starting at `base`, pointing along `ang` (radians, 0 = up)."""
        us = np.linspace(0, 1, 14)
        left, right = [], []
        for u in us:
            hw = width * .5 * math.sin(math.pi * u ** (.8 * bulge)) ** .9
            ax, ay = math.sin(ang), -math.cos(ang)
            px, py = base[0] + ax * length * u, base[1] + ay * length * u
            left.append((px - ay * hw, py + ax * hw))
            right.append((px + ay * hw, py - ax * hw))
        return left + right[::-1]

    def draw_leaf(d, base, ang, length, width, pal, vein=True):
        light, mid, dark = pal
        outline = tuple(int(v * .4) for v in dark)
        sc = lambda pts: [(x * ss, y * ss) for x, y in pts]
        d.polygon(sc(lens(base, ang, length * 1.04, width * 1.18)), fill=outline + (255,))
        d.polygon(sc(lens(base, ang, length, width)), fill=dark + (255,))
        shift = (math.cos(ang) * width * .05, math.sin(ang) * width * .05)
        d.polygon(sc(lens((base[0] - shift[0] * 0, base[1] - 1), ang, length * .96, width * .86)), fill=mid + (255,))
        d.polygon(sc(lens((base[0], base[1] - 2), ang, length * .8, width * .38)), fill=light + (255,))
        if vein:
            tip = (base[0] + math.sin(ang) * length * .9, base[1] - math.cos(ang) * length * .9)
            d.line([(base[0] * ss, base[1] * ss), (tip[0] * ss, tip[1] * ss)], fill=dark + (255,), width=ss * 2)

    def new():
        c = Image.new("RGBA", (CELL * ss, CELL * ss), (0, 0, 0, 0))
        return c, ImageDraw.Draw(c)

    def put(canvas, idx):
        atlas.paste(canvas.resize((CELL, CELL), Image.LANCZOS), ((idx % 4) * CELL, (idx // 4) * CELL))

    # 0, 1: ferns
    for idx, pal in ((0, GREEN), (1, GOLD)):
        c, d = new()
        P = lambda u: ((1 - u) ** 2 * 128 + 2 * u * (1 - u) * 118 + u * u * 160, (1 - u) ** 2 * 248 + 2 * u * (1 - u) * 120 + u * u * 22)
        pts = [P(u) for u in np.linspace(0, 1, 30)]
        d.line([(x * ss, y * ss) for x, y in pts], fill=tuple(int(v * .4) for v in pal[2]) + (255,), width=ss * 6)
        d.line([(x * ss, y * ss) for x, y in pts], fill=pal[2] + (255,), width=ss * 3)
        for u in np.linspace(.1, .96, 17):
            x0, y0 = P(u)
            x1, y1 = P(min(1, u + .03))
            heading = math.atan2(x1 - x0, -(y1 - y0))
            L = 66 * (1 - .72 * u)
            for side in (-1, 1):
                draw_leaf(d, (x0, y0), heading + side * math.radians(68), L, L * .36, pal, vein=False)
        put(c, idx)

    # 2, 3: broad-leaf plants
    for idx, pal in ((2, GREEN), (3, GOLD)):
        c, d = new()
        for ang, length, width, bx in ((-.5, 150, 96, 100), (.5, 150, 96, 156), (0.0, 196, 122, 128)):
            base = (bx, 248)
            d.line([(base[0] * ss, base[1] * ss), ((base[0] + math.sin(ang) * 46) * ss, (base[1] - math.cos(ang) * 46) * ss)],
                   fill=pal[2] + (255,), width=ss * 5)
            draw_leaf(d, (base[0] + math.sin(ang) * 40, base[1] - math.cos(ang) * 40), ang, length - 30, width, pal)
        put(c, idx)

    # 4, 5: sword-leaf plants
    for idx, pal in ((4, TEAL), (5, OLIVE)):
        c, d = new()
        for ang, length, width in ((-.9, 150, 30), (.9, 150, 30), (-.55, 196, 34), (.55, 196, 34), (-.2, 226, 36), (.2, 226, 36), (0.0, 238, 34)):
            draw_leaf(d, (128 + ang * 14, 250), ang, length, width, pal)
        put(c, idx)

    # 6, 7: flower sprigs
    for idx, petals in ((6, ((150, 110, 214), (186, 150, 236))), (7, ((255, 140, 170), (255, 178, 84)))):
        c, d = new()
        d.line([(128 * ss, 250 * ss), (134 * ss, 70 * ss)], fill=GREEN[2] + (255,), width=ss * 4)
        for ang, y in ((-.9, 190), (.9, 160), (-.8, 130), (.8, 110)):
            draw_leaf(d, (131, y + 60), ang, 56, 26, GREEN, vein=False)
        spots = [(128, 44), (96, 76), (166, 70), (112, 110), (150, 108), (134, 74)]
        for k, (fx, fy) in enumerate(spots):
            col = petals[k % 2]
            for t in range(5):
                a = t * 2 * math.pi / 5
                x, y = fx + math.cos(a) * 15, fy + math.sin(a) * 15
                d.ellipse(((x - 11) * ss, (y - 11) * ss, (x + 11) * ss, (y + 11) * ss), fill=tuple(int(v * .45) for v in col) + (255,))
            for t in range(5):
                a = t * 2 * math.pi / 5
                x, y = fx + math.cos(a) * 15, fy + math.sin(a) * 15
                d.ellipse(((x - 9) * ss, (y - 9) * ss, (x + 9) * ss, (y + 9) * ss), fill=col + (255,))
            d.ellipse(((fx - 8) * ss, (fy - 8) * ss, (fx + 8) * ss, (fy + 8) * ss), fill=(255, 214, 80, 255))
        put(c, idx)
    return _tex(atlas)


def _plant_card(idx, base, out, size, rnd, tilt=-.4):
    """One undergrowth card standing at `base`, leaning out along `out` with its top bending outward."""
    d = np.asarray(out, float) * .9 + np.array((0.0, tilt, 0.0))
    d /= np.linalg.norm(d)
    R = _orient(d, rnd.uniform(-8, 8))
    centre = np.asarray(base, float) + R[:, 1] * size * .5
    v = rnd.randint(88, 108)                                      # (the toon lighting brightens cards a lot)
    n = np.asarray(out, float) * .4 + np.array((0.0, .9, 0.0))
    PLANTS_B[0].add(CARD, tuple(centre), (size, size, 1), c3(v, v, v), uv=(.25, .5), uv_off=((idx % 4) * .25, .5 if idx < 4 else 0.0),
                    rot=R, normal=n)


def _ring(x, z, size, rnd, idxs, n, tilt, spread):
    a0 = rnd.uniform(0, 6.28)
    for k in range(n):
        a = a0 + k * 2 * math.pi / n + rnd.uniform(-.25, .25)
        out = np.array((math.cos(a), 0.0, math.sin(a)))
        _plant_card(rnd.choice(idxs), np.array((x, 0.0, z)) + out * spread, out, size * rnd.uniform(.85, 1.15), rnd, tilt)


def fern_clump(x, z, size, rnd):
    _ring(x, z, size, rnd, (0, 0, 0, 1), 6, -.55, .05)


def broad_clump(x, z, size, rnd):
    _ring(x, z, size, rnd, (2, 2, 3), 5, -.35, .08)


def spiky_clump(x, z, size, rnd):
    _ring(x, z, size, rnd, (4, 4, 5), 5, -.2, .06)


def flower_patch(x, z, size, rnd):
    idx, a = rnd.choice((6, 7)), rnd.uniform(0, 180)
    for k in (0, 90):
        R = _rot(a + k, 0)
        PLANTS_B[0].add(CARD, (x, size * .5, z), (size, size, 1), c3(118, 118, 118), ry=a + k, uv=(.25, .5),
                        uv_off=((idx % 4) * .25, .5 if idx < 4 else 0.0), normal=(0, 1, 0))


def make_grass_atlas(CELL=256):
    """Vector grass tufts: nine curved, pointed blades with a dark outline, a light edge, in fresh green and dry autumn yellow."""
    ss = 3
    atlas = Image.new("RGBA", (CELL * 2, CELL), (0, 0, 0, 0))
    pals = (((140, 204, 56), (86, 160, 36), (44, 106, 30)), ((218, 182, 64), (180, 136, 40), (118, 84, 32)))
    for col, (light, mid, dark) in enumerate(pals):
        rnd = random.Random(90 + col)
        canvas = Image.new("RGBA", (CELL * ss, CELL * ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)
        outline = tuple(int(v * .45) for v in dark)
        blades = [(rnd.uniform(-1, 1), rnd.uniform(130, 215)) for _ in range(9)]
        blades.sort(key=lambda b: abs(b[0]))                                     # outer blades first, central ones in front
        blades.reverse()
        for lean, H in blades:
            bx, by = CELL / 2 + lean * 30, CELL - 8
            tip = (bx + lean * 62, by - H)
            mid_pt = (bx + lean * 24, by - H * .5)

            def edge(w, sign):
                return [(bx + sign * w * (1 - t) + (mid_pt[0] - bx) * 2 * t * (1 - t) + (tip[0] - bx) * t * t,
                         by + (mid_pt[1] - by) * 2 * t * (1 - t) + (tip[1] - by) * t * t) for t in np.linspace(0, 1, 9)]

            def blade(w):
                return [(px * ss, py * ss) for px, py in edge(w, -1) + list(reversed(edge(w, 1)))[1:]]

            d.polygon(blade(10), fill=outline + (255,))
            d.polygon(blade(8), fill=mid + (255,))
            d.polygon([(px * ss, py * ss) for px, py in edge(8, 1) + list(reversed(edge(2, 1)))], fill=light + (255,))   # light right-hand edge
        atlas.paste(canvas.resize((CELL, CELL), Image.LANCZOS), (col * CELL, 0))
    return _tex(atlas)


def make_toon_leaf_shader():
    sh = Shader(language=Shader.GLSL, name="toon_leaf_shader", vertex=toon_shader.vertex,
                fragment=toon_shader.fragment.replace("fragment_color = cast_shadows(fragment_color);",
                                                      "if (fragment_color.a < 0.5) discard;\n    fragment_color = cast_shadows(fragment_color);"),
                default_input=dict(toon_shader.default_input))
    sh.continuous_input = dict(toon_shader.continuous_input)
    return sh


LEAF_CELL = {"green": 0, "yellow": 1, "orange": 2, "red": 3}


def make_leaf_atlas(CELL=256):
    """Vector-style leaf clusters drawn in code: lobed leaves with a dark outline, light top, dark rim and veins.
    4 colour themes x 2 variants, transparent between the leaves (used by the foliage cards)."""
    ss = 3
    atlas = Image.new("RGBA", (CELL * 4, CELL * 2), (0, 0, 0, 0))

    def lobed(cx, cy, R, rot, k=1.0):
        return [((cx + math.cos(t) * R * k * (1 + .2 * math.cos(5 * (t - rot)))) * ss, (cy + math.sin(t) * R * k * (1 + .2 * math.cos(5 * (t - rot)))) * ss)
                for t in np.linspace(0, 2 * math.pi, 60, endpoint=False)]

    for name, col in LEAF_CELL.items():
        light, mid, dark = THEMES[name]
        outline = tuple(int(v * .42) for v in dark)
        for row in range(2):
            rnd = random.Random(col * 10 + row + 3)
            canvas = Image.new("RGBA", (CELL * ss, CELL * ss), (0, 0, 0, 0))
            d = ImageDraw.Draw(canvas)
            leaves = []
            for _ in range(11):
                a, rr = rnd.uniform(0, 2 * math.pi), 66 * math.sqrt(rnd.random())
                leaves.append((CELL / 2 + math.cos(a) * rr, CELL / 2 + math.sin(a) * rr * .92, rnd.uniform(30, 42), rnd.uniform(0, 6.28)))
            leaves.sort(key=lambda l: l[1])                                   # top first: lower leaves overlap the ones above
            for cx, cy, R, rot in leaves:
                d.polygon(lobed(cx, cy, R * 1.1, rot), fill=outline + (255,))                       # dark outline
                d.polygon(lobed(cx, cy, R, rot), fill=dark + (255,))                                # dark rim
                d.polygon(lobed(cx - R * .07, cy - R * .09, R * .93, rot), fill=mid + (255,))        # body
                d.polygon(lobed(cx - R * .17, cy - R * .2, R * .58, rot), fill=light + (255,))       # sunlit top-left
                for t in range(5):                                                                  # veins towards the lobes
                    ang = rot + t * 2 * math.pi / 5
                    d.line([(cx * ss, cy * ss), ((cx + math.cos(ang) * R * .8) * ss, (cy + math.sin(ang) * R * .8) * ss)],
                           fill=dark + (255,), width=ss * 2)
            cell = canvas.resize((CELL, CELL), Image.LANCZOS)
            atlas.paste(cell, (col * CELL, row * CELL))
    return _tex(atlas)


def _leaf_cards(G, c, radii, n, size, name, rnd, min_y, reach=(.8, 1.05), scale=(1.0, 1.4)):
    """n leaf cards all round a clump, built in one go with numpy (a world has ~90k of them, so one at a time was the slow start)."""
    rs = np.random.RandomState(rnd.getrandbits(32))
    d = rs.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:, 1] = np.maximum(d[:, 1], min_y)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    centers = c + d * radii * rs.uniform(*reach, (n, 1))
    sizes = size * rs.uniform(*scale, n)
    roll = np.radians(rs.uniform(0, 360, n))
    up = np.where((np.abs(d[:, 1]) < .95)[:, None], (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    x = np.cross(up, d)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(d, x)
    co, si = np.cos(roll)[:, None], np.sin(roll)[:, None]
    cols = np.stack((x * co + y * si, -x * si + y * co, d), axis=2)                 # (n, 3, 3) rotation matrices
    cv = CARD.V[None] * np.stack((sizes, sizes, np.ones(n)), 1)[:, None, :]          # (n, 6, 3)
    V = np.einsum("nij,nkj->nki", cols, cv) + centers[:, None, :]
    row = rs.randint(0, 2, n)
    uv = CARD.UV[None] * (.25, .5) + np.stack((np.full(n, LEAF_CELL[name] * .25), np.where(row == 0, .5, 0.0)), 1)[:, None, :]
    shade = rs.randint(235, 256, n) / 255.0
    C = np.ones((n, 6, 4))
    C[:, :, :3] = shade[:, None, None]
    G.leaves.add_raw(V.reshape(-1, 3), np.repeat(d, 6, axis=0), uv.reshape(-1, 2), C.reshape(-1, 4))


def make_needle_atlas(CELL=256):
    """Vector-style conifer branches drawn in code: drooping fans with jagged needle tips, a dark outline, a light top band and
    needle strokes. Two greens (deep teal, olive)."""
    ss = 3
    atlas = Image.new("RGBA", (CELL * 2, CELL), (0, 0, 0, 0))
    pals = (((104, 176, 130), (44, 122, 92), (22, 76, 64)), ((156, 200, 104), (86, 142, 70), (44, 96, 52)))
    for col, (light, mid, dark) in enumerate(pals):
        rnd = random.Random(50 + col)
        canvas = Image.new("RGBA", (CELL * ss, CELL * ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)
        outline = tuple(int(v * .4) for v in dark)
        for y0, width, drop in ((52, 220, 150), (98, 192, 124), (142, 158, 96)):                # top tuft first, lower ones overlap
            cx, teeth = CELL / 2 + rnd.uniform(-5, 5), 12
            tips = [(rnd.uniform(-3, 3), rnd.uniform(.85, 1.15)) for _ in range(teeth + 1)]
            vals = [rnd.uniform(.36, .5) for _ in range(teeth)]

            def poly(shrink=1.0, dy=0.0, depth=1.0):
                top = [(cx - width * .28 * shrink + width * .56 * shrink * t / 12, y0 + dy - 12 * math.sin(math.pi * t / 12)) for t in range(13)]
                bottom = []
                for i in range(teeth * 2 + 1):
                    fx = cx + width / 2 * shrink - width * shrink * i / (teeth * 2)
                    if i % 2 == 0:
                        jx, jl = tips[i // 2]
                        bottom.append((fx + jx, y0 + dy + drop * depth * jl))
                    else:
                        bottom.append((fx, y0 + dy + drop * depth * vals[i // 2]))
                return [(px * ss, py * ss) for px, py in top + bottom]

            d.polygon(poly(1.05, -2, 1.05), fill=outline + (255,))                              # dark outline
            d.polygon(poly(1.0, 0, 1.0), fill=dark + (255,))                                    # dark underside / needle tips
            d.polygon(poly(.95, -5, .86), fill=mid + (255,))                                    # body
            d.polygon(poly(.7, -9, .42), fill=light + (255,))                                   # sunlit top band
            for i in range(9):                                                                  # needle strokes fanning out
                ang = math.radians(100 + i * 10)
                x1 = cx + math.cos(ang) * 0
                x2 = cx + math.cos(ang) * drop * .78 * -1 * (1 if i < 5 else 1)
                d.line([(cx * ss, (y0 - 6) * ss), ((cx + (i - 4) * width * .105) * ss, (y0 + drop * .74) * ss)], fill=dark + (255,), width=ss * 2)
        atlas.paste(canvas.resize((CELL, CELL), Image.LANCZOS), (col * CELL, 0))
    return _tex(atlas)


def tube(path, radii, seg=8, u_repeat=2.0, v_per_unit=1 / 2.4):
    """Sweep a circle along a path (trunks and branches). Returns flat V, N, UV arrays. Fully vectorised."""
    path, radii = np.asarray(path, float), np.asarray(radii, float)
    n = len(path)
    tang = np.gradient(path, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    ref = np.where((np.abs(tang[:, 2]) < .9)[:, None], np.array((0.0, 0.0, 1.0)), np.array((1.0, 0.0, 0.0)))
    a = np.cross(tang, ref)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.cross(tang, a)
    ang = np.linspace(0, 2 * np.pi, seg + 1)
    dirs = np.cos(ang)[None, :, None] * a[:, None, :] + np.sin(ang)[None, :, None] * b[:, None, :]       # (ring, seg+1, 3)
    rings = path[:, None, :] + dirs * radii[:, None, None]
    length = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
    ii, jj = np.meshgrid(np.arange(n - 1), np.arange(seg), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    corners = ((ii, jj), (ii, jj + 1), (ii + 1, jj + 1), (ii + 1, jj))
    order = (0, 1, 2, 0, 2, 3)
    V = np.stack([rings[corners[k][0], corners[k][1]] for k in order], axis=1).reshape(-1, 3)
    N = np.stack([dirs[corners[k][0], corners[k][1]] for k in order], axis=1).reshape(-1, 3)
    UV = np.stack([np.stack([corners[k][1] / seg * u_repeat, length[corners[k][0]] * v_per_unit], axis=1) for k in order], axis=1).reshape(-1, 2)
    return V, N, UV


def _spiky_cone(res=10, dip=.3):
    """A conifer tier: cone with a zig-zag bottom edge, flat-shaded facets."""
    V, N = [], []
    pts = []
    for k in range(res * 2):
        a = 2 * math.pi * k / (res * 2)
        r = .5 if k % 2 == 0 else .5 * (1 - dip)
        pts.append((r * math.cos(a), 0.0, r * math.sin(a)))
    apex = (0.0, 1.0, 0.0)
    for k in range(len(pts)):
        a, b = np.array(pts[k]), np.array(pts[(k + 1) % len(pts)])
        n = np.cross(b - a, np.array(apex) - a)
        n = n / (np.linalg.norm(n) + 1e-9)
        if n[1] < 0:
            n = -n
        V += [pts[k], pts[(k + 1) % len(pts)], apex]
        N += [n, n, n]
    return Tpl(V, N, np.zeros((len(V), 2)))


SPH8 = _sphere(8, 5)
SPH10 = _sphere(10, 6)
ICOS = Tpl(ICO.V, ICO.V / (np.linalg.norm(ICO.V, axis=1, keepdims=True) + 1e-9), ICO.UV)      # smooth-shaded low-poly puff
SPIKY = _spiky_cone()


def _pick_theme_name(rnd):
    r, acc = rnd.random(), 0
    for name, w in THEME_WEIGHTS:
        acc += w
        if r < acc:
            return name
    return "green"


def _clump(G, c, r, name, rnd, flat=.7, ncards=22):
    """A foliage clump: a dark core (so you can't see through), many vector leaf-cluster cards facing outward, and an invisible
    shadow blob. Flattened a little, like the layered crowns in the references."""
    theme = THEMES[name]
    c = np.asarray(c, float)
    radii = np.array((1.0, flat, 1.0)) * r
    G.trees.add(SPH8, tuple(c), radii * 1.3, c3(*theme[2]))
    G.shadow.add(SPH8, tuple(c), radii * 2.3, c3(255, 255, 255))
    _leaf_cards(G, c, radii, ncards, r, name, rnd, -.85)             # all round, including the underside


def tree_broadleaf(G, x, z, h, rnd):
    """Tall curved trunk, root flare, 3-4 spreading branches, and a separate leafy clump on each branch tip (gaps show the branches)."""
    name = _pick_theme_name(rnd)
    th = h * rnd.uniform(.46, .56)
    ts = np.linspace(0, 1, 9)
    lx1, lz1, lx2 = rnd.uniform(-.1, .1) * h, rnd.uniform(-.07, .07) * h, rnd.uniform(-.05, .05) * h
    path = np.stack([x + lx1 * ts ** 2 + lx2 * np.sin(ts * 5), th * ts, z + lz1 * ts ** 2], axis=1)
    r0 = h * .075
    radii = r0 * (1 - .55 * ts) + r0 * .75 * np.exp(-ts * 12)
    V, N, UV = tube(path, radii)
    G.bark.add_raw(V, N, UV, np.ones((len(V), 4)))
    top = path[-1]
    n_branch, a0, tips = rnd.choice((3, 4)), rnd.uniform(0, 6.28), []
    for k in range(n_branch):
        a = a0 + k * 2 * math.pi / n_branch + rnd.uniform(-.3, .3)
        L = h * rnd.uniform(.3, .45)
        d = np.array((math.cos(a), rnd.uniform(.55, 1.0), math.sin(a)))
        d /= np.linalg.norm(d)
        start = path[4 + k % 4]
        pts = np.array([start + d * L * t + np.array((0, .12 * L * math.sin(math.pi * t), 0)) for t in np.linspace(0, 1, 6)])
        V, N, UV = tube(pts, np.linspace(r0 * .5, r0 * .16, 6), seg=6, u_repeat=1.0, v_per_unit=1 / 3)
        G.bark.add_raw(V, N, UV, np.ones((len(V), 4)))
        tips.append(pts[-1])
    cr = h * rnd.uniform(.15, .19)
    _clump(G, top + np.array((0, h * .2, 0)), cr * 1.2, name, rnd)
    for tip in tips:
        _clump(G, tip + np.array((0, h * .07, 0)), cr * rnd.uniform(.95, 1.15), name, rnd)
    _clump(G, (top + tips[0]) / 2 + np.array((0, h * .05, 0)), cr * .8, name, rnd, ncards=14)


def bush(G, x, z, size, rnd):
    """A round leafy bush: dark core covered in leaf cards, with a grass tuft at its foot."""
    name = rnd.choice(("green", "green", "green", "yellow", "orange"))
    theme = THEMES[name]
    c = np.array((x, size * .55, z))
    radii = np.array((1.0, .75, 1.0)) * size * .85
    G.trees.add(SPH8, tuple(c), radii * 1.15, c3(*theme[1]))
    _leaf_cards(G, c, radii, 22, size, name, rnd, -.3, reach=(.8, 1.0), scale=(.95, 1.4))
    tuft(G.trees, x + rnd.uniform(-.5, .5) * size, z + size * .7, rnd)


def tree_pine(G, x, z, h, rnd):
    """Slim spruce: visible trunk, a dark spiky core, and many tiers of drooping needle fans that shrink towards the tip."""
    var = rnd.randrange(2)
    ts = np.linspace(0, 1, 4)
    path = np.stack([np.full(4, x), h * .28 * ts, np.full(4, z)], axis=1)
    V, N, UV = tube(path, h * .06 * (1 - .2 * ts) + h * .03 * np.exp(-ts * 10), seg=7)
    G.bark.add_raw(V, N, UV, np.ones((len(V), 4)))
    core = (32, 98, 76) if var == 0 else (58, 118, 62)
    tiers = 8
    for k in range(tiers):                                       # dark inner cones fill the gaps behind the needle fans
        y, w, tall = h * (.15 + .105 * k), h * (.46 - .045 * k), h * .2
        G.trees.add(SPIKY, (x, y, z), (w, tall, w), c3(*core), ry=rnd.uniform(0, 36))
    G.trees.add(SPIKY, (x, h * .9, z), (h * .1, h * .14, h * .1), c3(*core))
    centre = np.array((x, 0.0, z))
    for k in range(tiers):
        y, r, size, n = h * (.17 + .105 * k), h * (.27 - .03 * k), h * (.37 - .032 * k), max(5, 11 - k)
        a0 = rnd.uniform(0, 6.28)
        for i in range(n):
            a = a0 + i / n * 2 * math.pi + rnd.uniform(-.15, .15)
            out = np.array((math.cos(a), 0.0, math.sin(a)))
            d = out * .85 + np.array((0.0, .5, 0.0))               # leans outward-up, so the fringe droops outward-down
            pos = centre + out * r * .55 + np.array((0.0, y, 0.0))
            v = rnd.randint(238, 255) if k % 2 == 0 else rnd.randint(188, 206)           # alternate light / dark tiers so the layers read
            G.needles.add(CARD, tuple(pos), (size, size, 1), c3(v, v, v), uv=(.5, 1), uv_off=(.5 * var, 0),
                          rot=_orient(d, rnd.uniform(-10, 10)), normal=d)
    for _ in range(3):                                           # the leader at the very top
        d = np.array((rnd.uniform(-.5, .5), 1.0, rnd.uniform(-.5, .5)))
        G.needles.add(CARD, (x, h * .92, z), (h * .16, h * .16, 1), c3(245, 245, 245), uv=(.5, 1), uv_off=(.5 * var, 0),
                      rot=_orient(d, rnd.uniform(0, 360)), normal=d)


def wildflower(b, x, z, rnd):
    col = c3(*rnd.choice(WILD))
    b.add(CYL, (x, 0, z), (.025, .26, .025), c3(80, 130, 54))
    for k in range(4):
        a = k / 4 * 2 * math.pi
        b.add(ICO, (x + math.cos(a) * .07, .28, z + math.sin(a) * .07), (.1, .05, .1), col)


def leaf_litter(b, x, z, rnd):
    b.add(ICO, (x, .02, z), (rnd.uniform(.14, .24), .015, rnd.uniform(.1, .18)), c3(*rnd.choice(LEAVES)), ry=rnd.uniform(0, 360))


def tuft(b, x, z, rnd):
    if GRASS_B[0] is not None:
        size, var, a = rnd.uniform(.55, 1.05), rnd.randrange(2), rnd.uniform(0, 180)
        v = rnd.randint(176, 204)                                 # the toon lighting brightens cards, so tint them down a little
        for k in (0, 90):
            GRASS_B[0].add(CARD, (x, size * .5, z), (size, size, 1), c3(v, v, v), ry=a + k, uv=(.5, 1), uv_off=(.5 * var, 0),
                           normal=(0, 1, 0))
        return
    for _ in range(5):
        b.add(CONE, (x + rnd.uniform(-.08, .08), 0, z + rnd.uniform(-.08, .08)), (.09, rnd.uniform(.4, .8), .09),
              c3(*rnd.choice(BLADES)), ry=rnd.uniform(0, 360), rz=rnd.uniform(-35, 35))


def prop(b, name, x, z, rnd, scale, y=0.0, ry=None):
    t = load_obj(name)
    if t is None:
        return False
    v = rnd.randint(228, 255)
    b.add(t, (x, y, z), scale, c3(v, v, v), ry=rnd.uniform(0, 360) if ry is None else ry)
    return True


FLOWER_MODELS = ("flower_purpleA", "flower_purpleB", "flower_purpleC", "flower_redA", "flower_redB", "flower_redC", "flower_yellowA", "flower_yellowB", "flower_yellowC")
ROCKS = ("stone_largeA", "stone_largeB", "stone_largeC", "stone_largeD", "stone_largeE", "stone_largeF", "rock_largeA", "rock_largeC", "rock_tallA")
SMALL_STONES = ("stone_smallA", "stone_smallB", "stone_smallC", "stone_smallD", "stone_smallE", "stone_smallF", "stone_smallG", "stone_smallH")


# ---------------------------------------------------------------- natural rocks and logs (procedural)
def _ico_faces(level=2):
    t = (1 + 5 ** .5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(q, float) / np.linalg.norm(q) for q in v]
    for _ in range(level):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(verts), f


_ICO_V, _ICO_F = _ico_faces(2)                    # small stones (seeds 12+): 320 faces
_ICO_V_FINE, _ICO_F_FINE = _ico_faces(3)          # boulders (seeds 0-11): 1280 faces, so the facets are small and the shape looks smooth
_rock_cache = {}


def make_rock(seed, moss=True, tall=1.0):
    """A chunky, faceted boulder: a noisy ellipsoid with a flat base and a few angular cuts (like the reference rocks), flat-shaded,
    painted per face: lighter on top, darker on the sides, mossy green where it faces up. Unit-ish size: x/z about 1, y about 1."""
    key = (seed, moss, round(tall, 2))
    if key in _rock_cache:
        return _rock_cache[key]
    r = random.Random(seed)
    ph = [r.uniform(0, 6.28) for _ in range(6)]
    fine = seed < 12
    V = (_ICO_V_FINE if fine else _ICO_V).copy()
    ln = 1 + .13 * np.sin(V[:, 0] * 3.1 + ph[0]) * np.cos(V[:, 2] * 2.7 + ph[1]) + .08 * np.sin(V[:, 1] * 4.3 + ph[2]) + .05 * np.sin(V[:, 2] * 7 + ph[3])
    V = V * ln[:, None] * np.array((.5, .5 * tall, .5))
    for _ in range(r.choice((3, 4, 5))):                                    # planar cuts = angular facets
        n = np.array((r.gauss(0, 1), r.uniform(-.1, .9), r.gauss(0, 1)))
        n /= np.linalg.norm(n)
        d = r.uniform(.3, .42)
        over = np.maximum(V @ n - d * np.array((.5, .5 * tall, .5)).mean() * 2 * (.9 + .2 * tall), 0)
        V = V - np.outer(over, n)
    V[:, 1] = np.maximum(V[:, 1], -.28 * tall * .5)                         # flat base
    V[:, 1] += .28 * tall * .5                                              # sit on y = 0
    Vs, Ns, Cs = [], [], []
    for a, b, c in (_ICO_F_FINE if fine else _ICO_F):
        p0, p1, p2 = V[a], V[b], V[c]
        n = np.cross(p1 - p0, p2 - p0)
        n = n / (np.linalg.norm(n) + 1e-9)
        if np.dot(n, (p0 + p1 + p2) / 3 - np.array((0, .2 * tall, 0))) < 0:
            n = -n
            p1, p2 = p2, p1
        up = n[1]
        if moss and up > .55 and r.random() < .92:
            col = (112, 178, 60) if up > .8 else (86, 150, 52)
        elif up > .3:
            col = (176, 178, 186)
        elif up > -.2:
            col = (138, 140, 150) if n[0] + n[2] * .6 > 0 else (116, 118, 130)
        else:
            col = (92, 94, 106)
        j = r.uniform(.96, 1.04)
        Vs += [p0, p1, p2]
        Ns += [n, n, n]
        Cs += [[min(1, q * j / 255) for q in col]] * 3
    t = Tpl(Vs, Ns, np.zeros((len(Vs), 2)), Cs)
    t.size = np.array(Vs).max(0) - np.array(Vs).min(0)
    _rock_cache[key] = t
    return t


def _cap(seg=16):                       # flat disc in the xy plane facing +z, radius .5
    V, N = [], []
    for k in range(seg):
        a0, a1 = 2 * math.pi * k / seg, 2 * math.pi * (k + 1) / seg
        V += [(0, 0, 0), (.5 * math.cos(a0), .5 * math.sin(a0), 0), (.5 * math.cos(a1), .5 * math.sin(a1), 0)]
        N += [(0, 0, 1)] * 3
    return Tpl(V, N, np.zeros((len(V), 2)))


CAP = _cap()


def _end_cap(b, c, axis, radius, rnd):
    """The sawn end of a log: dark bark rim, pale wood face, growth rings and a dark heart."""
    R = _orient(axis)
    for scale, off, col in ((1.0, 0.0, (96, 62, 40)), (.86, .01, (240, 206, 150)), (.6, .02, (216, 178, 122)), (.58, .03, (240, 206, 150)),
                            (.3, .04, (204, 160, 108)), (.28, .05, (232, 194, 140)), (.07, .06, (150, 100, 64))):
        b.add(CAP, tuple(np.asarray(c, float) + np.asarray(axis, float) / np.linalg.norm(axis) * (off * radius)), (radius * 2 * scale,) * 2 + (1,),
              c3(*col), rot=R, normal=axis)


def fallen_log(G, x, z, rnd, size=1.0):
    """A gnarled log lying on the ground: slightly curved, tapering, with sawn ends, stubby branches, moss on top, and a
    little cluster of mushrooms."""
    L, r0 = 2.6 * size * rnd.uniform(.85, 1.25), .3 * size * rnd.uniform(.85, 1.15)
    a = rnd.uniform(0, math.pi)
    ax = np.array((math.cos(a), 0.0, math.sin(a)))
    side = np.array((-ax[2], 0.0, ax[0]))
    ts = np.linspace(0, 1, 9)
    bend = rnd.uniform(-.25, .25) * size
    pts = np.array([np.array((x, r0 * .85, z)) + ax * (t - .5) * L + side * (bend * math.sin(math.pi * t)) for t in ts])
    radii = r0 * (1.05 - .3 * ts) * (1 + .06 * np.sin(ts * 14))
    V, N, UV = tube(pts, radii, seg=9, u_repeat=2.0, v_per_unit=1 / 1.6)
    G.bark.add_raw(V, N, UV, np.ones((len(V), 4)))
    _end_cap(G.trees, pts[0], -ax, radii[0], rnd)
    _end_cap(G.trees, pts[-1], ax, radii[-1], rnd)
    for _ in range(rnd.choice((1, 2))):                                     # broken branch stubs
        t = rnd.uniform(.25, .75)
        i = int(t * 8)
        d = np.array((rnd.uniform(-.3, .3), 1.0, rnd.uniform(-.3, .3))) + side * rnd.choice((-1, 1)) * .8
        d /= np.linalg.norm(d)
        q = pts[i] + d * radii[i] * .7
        stub = np.array([q + d * u * .35 * size for u in np.linspace(0, 1, 3)])
        V, N, UV = tube(stub, np.array((.07, .06, .045)) * size, seg=6, u_repeat=1.0, v_per_unit=1 / 1.2)
        G.bark.add_raw(V, N, UV, np.ones((len(V), 4)))
        _end_cap(G.trees, stub[-1], d, .045 * size, rnd)
    for k in range(rnd.randint(3, 5)):                                      # moss along the top
        t = rnd.uniform(.1, .9)
        i = min(8, int(t * 8))
        G.trees.add(SPH8, tuple(pts[i] + np.array((rnd.uniform(-.08, .08), radii[i] * .82, rnd.uniform(-.08, .08)))),
                    (rnd.uniform(.4, .7) * size, .14 * size, rnd.uniform(.3, .45) * size), c3(*rnd.choice(((112, 178, 60), (86, 150, 52), (146, 196, 74)))),
                    ry=math.degrees(-a) + rnd.uniform(-15, 15))
    if rnd.random() < .6:
        mpos = pts[rnd.randrange(2, 7)] + side * radii[4] * 1.1
        for _ in range(rnd.randint(1, 3)):
            prop(G.trees, rnd.choice(("mushroom_red", "mushroom_redGroup", "mushroom_tan", "mushroom_tanGroup")),
                 mpos[0] + rnd.uniform(-.25, .25), mpos[2] + rnd.uniform(-.25, .25), rnd, rnd.uniform(1.4, 2.0))


def tree_stump(G, x, z, rnd, size=1.0):
    """A stump with a flared root base, bark texture, a sawn top with rings, moss and a few mushrooms."""
    h, r0 = .55 * size * rnd.uniform(.8, 1.3), .36 * size * rnd.uniform(.85, 1.15)
    ts = np.linspace(0, 1, 6)
    path = np.stack([np.full(6, x), h * ts, np.full(6, z)], axis=1)
    radii = r0 * (.8 + .5 * np.exp(-ts * 7))
    V, N, UV = tube(path, radii, seg=10, u_repeat=2.0, v_per_unit=1 / 1.6)
    G.bark.add_raw(V, N, UV, np.ones((len(V), 4)))
    _end_cap(G.trees, path[-1], np.array((0.0, 1.0, 0.0)), radii[-1], rnd)
    for _ in range(3):                                                      # roots creeping into the ground
        a = rnd.uniform(0, 6.28)
        d = np.array((math.cos(a), -.25, math.sin(a)))
        pts = np.array([np.array((x, .12 * size, z)) + d * u * r0 * 2.0 for u in np.linspace(.4, 1, 3)])
        V, N, UV = tube(pts, np.array((.12, .09, .05)) * size, seg=6, u_repeat=1.0, v_per_unit=1 / 1.2)
        G.bark.add_raw(V, N, UV, np.ones((len(V), 4)))
    for _ in range(rnd.randint(1, 2)):
        G.trees.add(SPH8, (x + rnd.uniform(-.15, .15) * size, h * .97, z + rnd.uniform(-.15, .15) * size),
                    (.3 * size, .06 * size, .22 * size), c3(*rnd.choice(((112, 178, 60), (86, 150, 52)))), ry=rnd.uniform(0, 180))
    for _ in range(rnd.randint(0, 2)):
        prop(G.trees, rnd.choice(("mushroom_tan", "mushroom_tanGroup", "mushroom_red")),
             x + rnd.uniform(-1, 1) * r0 * 1.4, z + rnd.uniform(-1, 1) * r0 * 1.4, rnd, rnd.uniform(1.3, 1.9))


def rock(b, x, z, rnd):
    """The 1-high obstacle: a big mossy faceted boulder fitted to its 2x2 cell (never higher than the collision's 1.0), with a smaller
    companion rock and pebbles at its foot."""
    t = make_rock(rnd.randrange(6), moss=True, tall=1.0)
    b.add(t, (x, 0.0, z), ((TILE - .12) / t.size[0], 1.02 / t.size[1], (TILE - .12) / t.size[2]), c3(255, 255, 255), ry=rnd.uniform(0, 360))
    if rnd.random() < .8:
        a, d = rnd.uniform(0, 6.28), rnd.uniform(1.0, 1.25)
        t2 = make_rock(rnd.randrange(6, 12), moss=rnd.random() < .6, tall=.8)
        sz = rnd.uniform(.5, .8)
        b.add(t2, (x + math.cos(a) * d, 0.0, z + math.sin(a) * d), (sz / t2.size[0], sz * .75 / t2.size[1], sz / t2.size[2]), c3(255, 255, 255),
              ry=rnd.uniform(0, 360))
    for _ in range(3):
        stone(b, x + rnd.uniform(-1.3, 1.3), z + rnd.uniform(-1.3, 1.3), rnd)


def stone(b, x, z, rnd):
    t = make_rock(rnd.randrange(12, 20), moss=rnd.random() < .35, tall=rnd.choice((.6, .75, .9)))
    w = rnd.uniform(.25, .6)
    b.add(t, (x, 0.0, z), (w / t.size[0], w * rnd.uniform(.5, .75) / t.size[1], w * rnd.uniform(.8, 1.1) / t.size[2]), c3(255, 255, 255), ry=rnd.uniform(0, 360))


def _rock_boxes(b, x, z, rnd):
    b.add(CUBE, (x, .5, z), (TILE - .1, 1, TILE - .1), c3(126, 126, 134))
    b.add(CUBE, (x + rnd.uniform(-.2, .2), 1.02, z + rnd.uniform(-.2, .2)), (1.3, .12, 1.3), c3(150, 150, 158), ry=rnd.uniform(-20, 20))
    b.add(CUBE, (x - .3, 1.08, z + .2), (.9, .1, .7), c3(104, 150, 66), ry=15)       # moss


MUD = ((104, 74, 54), (88, 62, 46), (126, 104, 90), (112, 84, 64))


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def ground_patch(dirt_b, grass_b, zlo, zhi, xmax, step=.4):
    """Fine ground mesh around the pits: smooth muddy hollows with a raised rim, dirt tinted dark and wet toward the water,
    and a second, alpha-faded grass layer so the trail edge is soft and irregular."""
    xs, zs = np.arange(-xmax, xmax + 1e-6, step), np.arange(zlo, zhi + 1e-6, step)
    X, Z = np.meshgrid(xs, zs)
    S = np.full(X.shape, -1.0)
    for q in PITS:
        S = np.maximum(S, q.s(X, Z))
    H = -.62 * smoothstep(0, .30, S) + .09 * np.exp(-((S + .07) / .07) ** 2)           # bowl + muddy lip
    hx, hz = np.gradient(H, axis=1) / step, np.gradient(H, axis=0) / step
    N = np.stack([-hx, np.ones_like(H), -hz], axis=-1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    UV = np.stack([X / TILE, Z / TILE], axis=-1)
    P = np.stack([X, H, Z], axis=-1)

    mud = smoothstep(-.30, .12, S)
    shade = 1 + .03 * np.sin(X * 1.7 + Z * .9)
    dirt_c = np.stack([(1.0 - .40 * mud) * shade, (.98 - .52 * mud) * shade, (.957 - .58 * mud) * shade, np.ones_like(H)], axis=-1)

    wob = .30 * np.sin(Z * .9) + .18 * np.sin(Z * 2.3 + 1.0) + .10 * np.sin(Z * 5.1)
    edge_x = HALF - .15 + wob
    alpha = smoothstep(edge_x - .3, edge_x + .3, np.abs(X)) * (1 - smoothstep(-.25, .0, S))
    dry = smoothstep(edge_x + .2, edge_x + .9, np.abs(X))                               # dry, trampled grass right at the edge
    grass_c = np.stack([.78 + .22 * dry, .76 + .24 * dry, .40 + .60 * dry, alpha], axis=-1)

    def cells(A):
        a, b, c, d = A[:-1, :-1], A[:-1, 1:], A[1:, 1:], A[1:, :-1]
        return np.stack([a, b, c, a, c, d], axis=2).reshape(-1, A.shape[-1])

    V, Nn, UVc = cells(P), cells(N), cells(UV)
    dirt_b.add_raw(V, Nn, UVc, cells(dirt_c))
    grass_b.add_raw(V + np.array((0, .006, 0)), Nn, UVc, cells(grass_c))


def dress_pit(misc, trees, q, rnd):
    """Water, sky glints, mud lumps, smears and reeds around one pit."""
    misc.add(DISCS[0], (q.cx, -.5, q.cz), (q.rx * 2.5, 1, q.rz * 2.5), c3(100, 130, 158))              # water level
    for _ in range(3):
        misc.add(rnd.choice(DISCS), (q.cx + rnd.uniform(-.4, .4) * q.rx, -.49, q.cz + rnd.uniform(-.4, .4) * q.rz),
                 (q.rx * rnd.uniform(.25, .5), 1, q.rz * rnd.uniform(.1, .25)), c3(176, 206, 228), ry=rnd.uniform(-8, 8))   # sky glints
    for k in range(int(5 * (q.rx + q.rz))):                                                              # lumps of mud on the lip
        th = 2 * math.pi * k / int(5 * (q.rx + q.rz)) + rnd.uniform(-.1, .1)
        x, z = q.point(th, rnd.uniform(.94, 1.08))
        misc.add(SPH, (x, 0.0, z), (rnd.uniform(.22, .45), rnd.uniform(.09, .17), rnd.uniform(.2, .4)), c3(*rnd.choice(MUD)), ry=rnd.uniform(0, 180))
    for _ in range(7):                                                                                   # wet mud smeared outward
        x, z = q.point(rnd.uniform(0, 6.28), rnd.uniform(1.02, 1.3))
        misc.add(rnd.choice(DISCS), (x, .022, z), (rnd.uniform(.9, 1.9), 1, rnd.uniform(.5, 1.1)), c3(160, 106, 66), ry=rnd.uniform(0, 180))
    for _ in range(6):                                                                                   # reeds and stones round the rim
        x, z = q.point(rnd.uniform(0, 6.28), rnd.uniform(1.12, 1.4))
        if abs(x) < 8:
            (tuft if rnd.random() < .6 else stone)(trees, x, z, rnd)


def finish_gate(b, z):
    for px in (-HALF, HALF):
        b.add(CYL, (px, 0, z), (.4, 4.2, .4), c3(112, 80, 52))
    b.add(CUBE, (0, 3.6, z), (HALF * 2, .8, .2), c3(206, 66, 48))
    b.add(CUBE, (0, 3.15, z), (HALF * 2, .08, .22), c3(250, 200, 70))
    for k in range(7):                                                              # bunting in autumn colours
        bx = -HALF + (k + .5) * HALF * 2 / 7
        b.add(CONE, (bx, 2.75, z), (.5, .5, .06), c3(*LEAVES[k % len(LEAVES)]), rz=180)


# ---------------------------------------------------------------- the panda (replaces the kid)
WHITE, BLACK, PINK, CREAM = (250, 250, 246), (34, 32, 40), (255, 160, 176), (255, 240, 222)
CURL_T = 1.15            # seconds the panda takes to curl into a ball and start rolling
UNCURL_T = .8            # ...and to pop back out when it reaches the gate


def ease(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


class Panda(Entity):
    """A cute panda built from soft spheres. Standing form (waves hello) and ball form (rolls); set_form() blends between them.
    The root sits on the ground, facing +z; the rig's origin is the body centre (feet at y = -.65)."""

    def __init__(self):
        super().__init__()

        def part(parent, col, pos, scale, **kw):
            return Entity(parent=parent, model="sphere", color=c3(*col), position=pos, scale=scale, **kw)

        r = self.rig = Entity(parent=self)
        part(r, WHITE, (0, 0, 0), (.7, .74, .6))
        part(r, CREAM, (0, -.04, .2), (.5, .56, .34))                                  # belly
        part(r, BLACK, (0, .2, .01), (.76, .2, .63))                                   # the black shoulder band
        part(r, WHITE, (0, -.2, -.32), .2)                                             # tail
        self.legs, self.arms = [], []
        for side in (-1, 1):
            hip = Entity(parent=r, position=(side * .2, -.4, 0))
            part(hip, BLACK, (0, -.12, .04), (.28, .34, .32))
            part(hip, PINK, (0, -.255, .17), (.17, .05, .12))                          # foot pads
            self.legs.append(hip)
            sh = Entity(parent=r, position=(side * .36, .22, 0))
            part(sh, BLACK, (side * .04, -.18, .04), (.2, .46, .22))
            self.arms.append(sh)
        self.head = h = Entity(parent=r, position=(0, .6, .03))
        part(h, WHITE, (0, 0, 0), (.68, .6, .6))
        self.eyes = []
        for side in (-1, 1):
            part(h, BLACK, (side * .28, .3, -.02), (.22, .22, .13))                    # ears
            part(h, BLACK, (side * .15, .03, .26), (.17, .22, .09), rotation_z=side * -22)   # eye patches
            part(h, PINK, (side * .26, -.1, .24), (.11, .075, .05))                    # rosy cheeks
            e = Entity(parent=h, position=(side * .15, .05, .3))
            part(e, (255, 255, 255), (0, 0, 0), (.085, .095, .05))
            part(e, BLACK, (0, 0, .03), (.055, .065, .03))
            part(e, (255, 255, 255), (side * -.012, .02, .05), .02)                    # sparkle
            self.eyes.append(e)
        part(h, BLACK, (0, -.08, .29), (.1, .072, .075))                               # nose
        part(h, BLACK, (-.03, -.15, .29), .025)
        part(h, BLACK, (.03, -.15, .29), .025)                                         # smile dots

        bg = self.ballg = Entity(parent=self)                                          # curled-up ball form (radius .5)
        part(bg, WHITE, (0, 0, 0), 1.0)
        for side in (-1, 1):
            part(bg, BLACK, (side * .27, .36, .2), .22)                                # ears
            part(bg, BLACK, (side * .15, .2, .44), (.16, .2, .09))                     # eye patches
            part(bg, (255, 255, 255), (side * .15, .22, .47), .08)
            part(bg, BLACK, (side * .15, .22, .505), .045)
            part(bg, PINK, (side * .28, .08, .4), (.1, .07, .05))
            part(bg, BLACK, (side * .46, 0, 0), (.12, .34, .34))                       # arm caps on the sides
            part(bg, BLACK, (side * .2, -.3, .36), .22)                                # leg bumps
        for side in (-1, 1):
            part(bg, BLACK, (side * .22, .22, -.42), (.24, .24, .14))                  # back / shoulder patches
            part(bg, BLACK, (side * .2, -.3, -.36), .22)
        part(bg, BLACK, (0, .1, .49), (.09, .065, .07))                                # nose
        part(bg, WHITE, (0, -.12, -.49), .16)                                          # tail
        self.set_form(0)

    def set_form(self, b):
        """0 = standing, 1 = ball."""
        self.rig.enabled = b < .98
        self.ballg.enabled = b > .02
        self.rig.scale = max(.001, 1 - b)
        self.rig.y = .65 - .15 * b
        self.ballg.scale = max(.001, b)
        self.ballg.y = .5

    def place(self, s, yaw):
        self.position = (s.x + float(curve(s.z)), 0, s.z)
        self.rotation_y = yaw

    def pose_idle(self, t, s):
        """Standing, facing the camera and waving."""
        self.set_form(0)
        self.place(s, 180)
        k = math.sin(t * 3)
        self.rig.y = .65 + .012 * k
        self.rig.scale = (1 + .012 * k, 1 - .014 * k, 1 + .012 * k)
        self.arms[1].rotation_z = -(158 + 16 * math.sin(t * 9))          # (negative = outward for the +x arm)
        self.arms[0].rotation_z = 8 - 3 * math.sin(t * 2.1)
        self.head.rotation = (0, 6 * math.sin(t * .9), 4 * math.sin(t * 1.7))
        blink = (t % 3.8) < .14
        for e in self.eyes:
            e.scale_y = .12 if blink else 1

    def pose_curl(self, u, s):
        """Crouch, hop, spin round and shrink into the ball (u runs 0..1)."""
        a = ease(u / .28)
        v = min(1.0, max(0.0, (u - .28) / .55))
        b = ease(v)
        hop = math.sin(math.pi * v) * .55
        self.set_form(b)
        k = max(.001, 1 - b)
        self.rig.scale = (k * (1 + .14 * a), k * (1 - .22 * a), k * (1 + .14 * a))
        self.rig.y = .65 - .15 * b - .09 * a * (1 - b) + hop
        self.ballg.y = .5 + hop
        w = min(1.0, max(0.0, (u - .8) / .2))
        sq = math.sin(math.pi * w)                                                   # squash and stretch as it settles
        self.ballg.scale = (b * (1 + .12 * sq), b * (1 - .18 * sq), b * (1 + .12 * sq))
        for side, arm in ((-1, self.arms[0]), (1, self.arms[1])):
            arm.rotation_z = -((158 + 16 * math.sin(u * 20) if side == 1 else -8) * (1 - a) + (-side * 25) * a)
        for leg in self.legs:
            leg.rotation_x = -25 * a
        self.head.rotation = (0, 0, 0)
        for e in self.eyes:
            e.scale_y = 1
        self.place(s, 180 + 180 * b)

    def pose_play(self, s, t, win_u=None):
        """Rolling (or squashed / sinking / popping back out when it wins)."""
        heading = math.degrees(math.atan(float(curve_slope(s.z))))
        self.place(s, heading)
        if s.state == "win" and win_u is not None:
            u = ease(win_u)                                                           # pop back out facing the camera and cheer
            self.set_form(1 - u)
            self.rotation_y = heading + 180 * u
            self.rig.y = .65 - .15 * (1 - u) + abs(math.sin(t * 7)) * .12 * u
            self.ballg.y = .5
            self.ballg.rotation_x = math.degrees(s.roll)
            wave = math.sin(t * 12) * 14
            self.arms[0].rotation_z, self.arms[1].rotation_z = 150 + wave, -(150 - wave)
            for leg in self.legs:
                leg.rotation_x = 0
            self.head.rotation = (0, 0, 4 * math.sin(t * 5))
            return
        self.set_form(1)
        self.ballg.y = s.y + R
        self.ballg.rotation_x = math.degrees(s.roll)
        sq = (1.18, .8, .62) if s.state == "crashed" else (1, 1, 1)                  # bonk!
        self.ballg.scale = sq



class Game(Entity):
    def __init__(self, test_mode=False):
        super().__init__()
        self.test_mode = test_mode
        self.paused = False
        self.t = 0.0
        self.build_world()
        self.build_ball()
        self.build_ui()
        self.parts, self.pool = [], []
        self.reset(intro=not test_mode)          # (tests skip the intro and go straight to rolling)

    # ---- world
    def build_world(self):
        self.dirt_tex, self.grass_tex = make_dirt_texture(), make_grass_texture()
        rnd = random.Random(7)
        groups = {}

        base.camNode.set_camera_mask(BitMask32.bit(1))          # main camera sees bit 1, the sun's shadow camera sees bit 0
        specs = {"dirt": (self.dirt_tex, None, None), "grass": (self.grass_tex, None, None), "shadow": (None, None, "shadow")}
        specs["bark"] = (make_bark_texture(), toon_shader, None)
        specs["trees"] = (None, toon_shader, None)
        specs["leaves"] = (make_leaf_atlas(), toon_leaf_shader, "leaves")
        specs["needles"] = (make_needle_atlas(), toon_leaf_shader, "leaves")
        specs["tufts"] = (make_grass_atlas(), toon_leaf_shader, "leaves")
        specs["plants"] = (make_plant_atlas(), toon_leaf_shader, "leaves")

        def grp(chunk, kind):
            if (chunk, kind) not in groups:
                tex, shd, role = specs.get(kind, (None, None, None))
                groups[(chunk, kind)] = Builder(tex, shd, role)
            return groups[(chunk, kind)]

        specs["gover"] = (self.grass_tex, None, "overlay")
        XMAX = 6.4                                              # half-width of the fine ground meshes around pits
        patch_rows = set(range(ROW_MIN, ROW_MAX))               # one fine ground mesh for the whole trail, in chunks of 8 rows
        rows_sorted = sorted(patch_rows)
        runs = [rows_sorted[i:i + 8] for i in range(0, len(rows_sorted), 8)]

        for i in range(ROW_MIN, ROW_MAX):
            ch = i // 8
            z0 = i * TILE
            zc = z0 + TILE / 2
            in_patch = i in patch_rows
            row = LEVEL[i] if 0 <= i < len(LEVEL) else "###"
            misc, dirt, trees = grp(ch, "misc"), grp(ch, "dirt"), grp(ch, "trees")
            GRASS_B[0] = grp(ch, "tufts")
            PLANTS_B[0] = grp(ch, "plants")
            G = SimpleNamespace(trees=trees, bark=grp(ch, "bark"), leaves=grp(ch, "leaves"), shadow=grp(ch, "shadow"),
                                needles=grp(ch, "needles"))
            for c in range(COLS):
                t = row[c]
                x = -HALF + c * TILE + TILE / 2
                v = rnd.randint(-10, 0)
                col = c3(255 + v, 250 + v, 244 + v)
                if t == "F":
                    col = c3(250, 238, 212) if (i + c) % 2 == 0 else c3(178, 86, 54)
                if not in_patch:                         # (rows around pits use the fine heightfield instead)
                    dirt.add(CUBE, (x, -DEPTH / 2, zc), (TILE, DEPTH, TILE), col)
                if t == "F":
                    misc.add(PLANE, (x, .016, zc), (TILE, 1, TILE), col)                    # chequered finish line
                if t == "B":
                    rock(trees, x, zc, rnd)
                elif t == "#" and not near_pit(x, z0, z0 + TILE, .6):
                    if c == 1:                           # grass strip down the middle
                        misc.add(PLANE, (0, .015, zc), (.45, 1, TILE), c3(128, 166, 60))
                    else:                                # wheel rut
                        misc.add(PLANE, (x + (.7 if c == 0 else -.7), .012, zc), (.4, 1, TILE), c3(192, 130, 80))
            solid_cols = [c for c in range(COLS) if row[c] in "#B"]
            if solid_cols and row[0] != "F":
                for _ in range(rnd.choice((1, 2, 2, 3))):        # dappled shade and sunlit patches painted on the dirt
                    px = -HALF + rnd.choice(solid_cols) * TILE + rnd.uniform(.2, TILE - .2)
                    pz = z0 + rnd.random() * TILE
                    lit = rnd.random() < .35
                    if not near_pit(px, pz, None, .8):
                        misc.add(rnd.choice(DISCS), (px, .02, pz), (rnd.uniform(.9, 1.9), 1, rnd.uniform(.5, 1.2)),
                                 c3(232, 174, 108) if lit else c3(186, 126, 76), ry=rnd.uniform(0, 180))
                for _ in range(3):                               # pebbles lying on the trail
                    px = -HALF + rnd.choice(solid_cols) * TILE + rnd.uniform(.2, TILE - .2)
                    pz = z0 + rnd.random() * TILE
                    if not near_pit(px, pz, None, .6):
                        misc.add(SPH, (px, .03, pz), (rnd.uniform(.14, .3), .06, rnd.uniform(.12, .22)),
                                 c3(*rnd.choice(STONE + ((150, 104, 72),))), ry=rnd.uniform(0, 180))
            if i == FINISH_ROW:
                finish_gate(misc, z0)

            # grass either side (a dirt block underneath it, except where the heightfield takes over)
            for side in (-1, 1):
                x_in, width = (XMAX, 144) if in_patch else (HALF, 147)       # wide enough that the edge of the ground is lost in the haze
                xc = side * (x_in + width / 2)
                if not in_patch:
                    dirt.add(CUBE, (xc, -DEPTH / 2 - .02, zc), (width, DEPTH, TILE), c3(214, 160, 110), uv=(width / 2, 1))
                grp(ch, "grass").add(PLANE, (xc, 0, zc), (width, 1, TILE), c3(255, 255, 255) if i % 2 else c3(242, 250, 228), uv=(width / 2, 1))
                if not in_patch:
                    misc.add(PLANE, (side * (HALF + .3), .01, zc), (.6, 1, TILE), c3(176, 168, 78))   # dry, trampled edge
                    for _ in range(rnd.choice((1, 2, 2))):       # dirt spilling into the grass, grass creeping onto the trail
                        misc.add(rnd.choice(DISCS), (side * (HALF + rnd.uniform(-.1, 1.1)), .018, z0 + rnd.random() * TILE),
                                 (rnd.uniform(.9, 1.8), 1, rnd.uniform(.5, 1.1)), c3(*rnd.choice(((196, 134, 80), (176, 116, 68), (210, 150, 92)))),
                                 ry=rnd.uniform(0, 180))
                    for _ in range(rnd.choice((1, 2))):
                        misc.add(rnd.choice(DISCS), (side * (HALF - rnd.uniform(0, .7)), .026, z0 + rnd.random() * TILE),
                                 (rnd.uniform(.5, 1.1), 1, rnd.uniform(.4, .8)), c3(*rnd.choice(BLADES)), ry=rnd.uniform(0, 180))

            # trees, bushes, wildflowers, fallen leaves, stones, grass tufts
            for side in (-1, 1):
                def pos(lo, span, power=1.0):
                    return side * (HALF + lo + rnd.random() ** power * span), z0 + rnd.random() * TILE

                def free(x, z):
                    return not near_pit(x, z, None, .5)
                for _ in range(rnd.choice((1, 1, 2, 2))):
                    x, z = pos(5.6, 18, 1.3)
                    if rnd.random() < .22:
                        tree_pine(G, x, z, rnd.uniform(6.5, 10), rnd)
                    else:
                        tree_broadleaf(G, x, z, rnd.uniform(6.0, 9.0), rnd)
                for _ in range(rnd.choice((0, 1, 1, 2))):
                    x, z = pos(.9, 2.8)
                    if free(x, z):
                        bush(G, x, z, rnd.uniform(.5, .95), rnd)
                if rnd.random() < .6:
                    x, z = pos(.5, 5)
                    if free(x, z):
                        wildflower(trees, x, z, rnd)
                for _ in range(5):
                    x, z = pos(.2, 7)
                    if free(x, z):
                        leaf_litter(trees, x, z, rnd)
                if rnd.random() < .45:
                    x, z = pos(.7, 5)
                    if free(x, z):
                        stone(trees, x, z, rnd)
                for _ in range(3):
                    x, z = pos(.3, 6)
                    if free(x, z):
                        tuft(trees, x, z, rnd)
                for maker, prob, lo, span, size_lo, size_hi in ((fern_clump, .42, .7, 8, 1.0, 1.6), (broad_clump, .28, .9, 9, .9, 1.4),
                                                                (spiky_clump, .2, .6, 7, 1.2, 2.0), (flower_patch, .36, .4, 6, .5, .8)):
                    if rnd.random() < prob:                      # undergrowth: ferns, broad leaves, sword-leaf plants, flower sprigs
                        x, z = pos(lo, span)
                        if free(x, z):
                            maker(x, z, rnd.uniform(size_lo, size_hi), rnd)
                for _ in range(3):                               # tufts along the very edge of the trail
                    x, z = pos(-.2, 1.6)
                    if free(x, z):
                        tuft(trees, x, z, rnd)
                if HAVE_MODELS:                                  # fallen logs, stumps, mushrooms and extra flowers
                    for maker, prob in ((fallen_log, .16), (tree_stump, .12)):
                        if rnd.random() < prob:
                            x, z = pos(1.6, 9)
                            if free(x, z):
                                maker(G, x, z, rnd, rnd.uniform(.9, 1.4))
                    for names, prob, lo, span, size in ((("mushroom_red", "mushroom_redGroup", "mushroom_tan", "mushroom_tanGroup"), .16, .8, 9, 2.0),
                                                        (FLOWER_MODELS, .5, .4, 6, 1.9)):
                        if rnd.random() < prob:
                            x, z = pos(lo, span)
                            if free(x, z):
                                prop(trees, rnd.choice(names), x, z, rnd, size * rnd.uniform(.85, 1.2))

        for run in runs:                                         # fine ground meshes with the muddy hollows
            ground_patch(grp(("patch", run[0]), "dirt"), grp(("patch", run[0]), "gover"), run[0] * TILE, (run[-1] + 1) * TILE, XMAX)
        for q in PITS:
            ch = int(q.cz // TILE) // 8
            dress_pit(grp(ch, "misc"), grp(ch, "trees"), q, rnd)

        far = grp("far", "grass")                                       # ground beyond the last row, out into the haze (no visible end of the world)
        z_end, L, W = ROW_MAX * TILE, 260.0, 300.0
        far.add(PLANE, (0, 0, z_end + L / 2), (W, 1, L), c3(255, 255, 255), uv=(W / 2, L / 2))
        z_start = ROW_MIN * TILE
        far.add(PLANE, (0, 0, z_start - L / 2), (W, 1, L), c3(255, 255, 255), uv=(W / 2, L / 2))

        for g in groups.values():
            g.finish()

        sun = DirectionalLight(shadows=True, shadow_map_resolution=Vec2(2048, 2048))
        sun.look_at(Vec3(-0.30, -0.95, -0.35))                          # sun sits ahead-right, like the backdrop
        self.sun = sun
        self.focus = Entity(model="cube", scale=(36, 14, 50), visible=False)   # region the shadow map covers
        sun.color = color.rgb32(255, 242, 212)                                  # warm afternoon sun
        AmbientLight(color=color.rgba(0.62, 0.64, 0.74, 0.5))
        scene.fog_color = FOG
        scene.fog_density = (55, 190)
        has_backdrop = build_backdrop()
        Sky(texture=make_sky_texture(sun=not has_backdrop, scenery=not has_backdrop)).shader = unlit_shader
        window.color = FOG
        camera.fov = 62
        camera.clip_plane_far = 180

    # ---- panda
    def build_ball(self):
        self.s = BallState()
        self.panda = Panda()
        self.cam_x = 0.0
        self.phase = "play"                                     # ready -> curl -> play
        self.idle_t = self.curl_u = 0.0
        self.win_t0 = None

    # ---- ui
    def build_ui(self):
        ui = camera.ui
        self.panel = Entity(parent=ui, model="quad", color=color.rgba(0.05, 0.28, 0.36, 0.62), scale=(.36, .085),
                            position=(-.66, .44, .02), shader=unlit_shader)
        hud_text = Text("DISTANCE", parent=ui, position=(-.83, .465), scale=.9, color=color.rgb32(235, 245, 225))
        hud_bg = Entity(parent=ui, model="quad", color=color.rgba(0.08, 0.2, 0.26, 1), scale=(.3, .016), position=(-.66, .42, .01), shader=unlit_shader)
        self.bar = Entity(parent=ui, model="quad", color=color.rgb32(250, 205, 80), scale=(.0001, .016), origin=(-.5, 0),
                          position=(-.81, .42, -.01), shader=unlit_shader)
        hud_flag = Entity(parent=ui, model="diamond", color=color.rgb32(230, 60, 50), scale=(.02, .03), position=(-.49, .42, -.01), shader=unlit_shader)
        self.hud = [self.panel, hud_text, hud_bg, self.bar, hud_flag]
        self.hint = Text("Left / Right  or  A / D  to steer     Space to jump     P: pause", parent=ui,
                         origin=(0, 0), y=-.45, scale=1.1, color=color.white)
        self.msg_bg = Entity(parent=ui, model="quad", color=color.rgba(0.04, 0.22, 0.3, 0.75), scale=(.8, .22), position=(0, .12, .02),
                             shader=unlit_shader, enabled=False)
        self.msg = Text("", parent=ui, origin=(0, 0), y=.15, scale=3, color=color.rgb32(255, 220, 90), enabled=False)
        self.msg2 = Text("Press R to play again", parent=ui, origin=(0, 0), y=.07, scale=1.3, color=color.rgb32(235, 245, 225),
                         enabled=False)

        # start screen: title + a big START button (click it, or press Space / Enter)
        self.start_ui = [
            Text("PANDA ROLL", parent=ui, origin=(0, 0), position=(.007, .418, -.05), scale=4.0, color=color.rgb32(40, 60, 56)),
            Text("PANDA ROLL", parent=ui, origin=(0, 0), position=(0, .425, -.06), scale=4.0, color=color.rgb32(255, 226, 120)),
            Text("a cosy roll through the autumn forest", parent=ui, origin=(0, 0), position=(.003, .342, -.05), scale=1.3,
                 color=color.rgb32(40, 60, 56)),
            Text("a cosy roll through the autumn forest", parent=ui, origin=(0, 0), position=(0, .346, -.06), scale=1.3,
                 color=color.rgb32(255, 255, 255)),
            Button(parent=ui, text="KEYBOARD", position=(-.2, -.3, -.05), scale=(.34, .1), color=color.rgb32(52, 168, 124),
                   highlight_color=color.rgb32(88, 208, 156), pressed_color=color.rgb32(36, 138, 100), text_color=color.white,
                   on_click=self.start_keyboard, shader=unlit_shader),
            Button(parent=ui, text="CAMERA", position=(.2, -.3, -.05), scale=(.34, .1), color=color.rgb32(232, 128, 52),
                   highlight_color=color.rgb32(250, 164, 92), pressed_color=color.rgb32(196, 100, 36), text_color=color.white,
                   on_click=self.start_camera, shader=unlit_shader),
            Text("arrow keys + Space", parent=ui, origin=(0, 0), position=(-.2, -.375, -.05), scale=1.0, color=color.rgb32(235, 245, 225)),
            Text("move your body: left / right, jump", parent=ui, origin=(0, 0), position=(.2, -.375, -.05), scale=1.0,
                 color=color.rgb32(235, 245, 225)),
            Text("choose how to play  (1 = keyboard, 2 = camera)", parent=ui, origin=(0, 0), position=(0, -.44, -.05), scale=1.0,
                 color=color.rgb32(255, 255, 255)),
        ]

        # camera calibration screen + the little live view of yourself during play
        self.cam_tex = PandaTexture("camera")
        self.cam_tex.setup2dTexture(320, 240, PandaTexture.TUnsignedByte, PandaTexture.FRgb)
        self.cam_tex.setRamImageAs(bytes(320 * 240 * 3), "RGB")
        self.cam_frame = Entity(parent=ui, model="quad", color=color.rgba(0.02, 0.1, 0.14, 0.85), scale=(.44, .34), position=(.5, .12, .02),
                                shader=unlit_shader, enabled=False)
        self.cam_view = Entity(parent=ui, model="quad", texture=Texture(self.cam_tex), scale=(.4, .3), position=(.5, .12, 0),
                               shader=unlit_shader, enabled=False)
        self.cam_status = Text("", parent=ui, origin=(0, 0), position=(0, -.32, -.05), scale=1.5, color=color.white, enabled=False)
        self.cam_sub = Text("", parent=ui, origin=(0, 0), position=(0, -.38, -.05), scale=1.0, color=color.rgb32(235, 245, 225), enabled=False)
        self.cam_bar_bg = Entity(parent=ui, model="quad", color=color.rgba(0.08, 0.2, 0.26, 1), scale=(.4, .016), position=(0, -.27, -.04),
                                 shader=unlit_shader, enabled=False)
        self.cam_bar = Entity(parent=ui, model="quad", color=color.rgb32(250, 205, 80), scale=(.0001, .016), origin=(-.5, 0),
                              position=(-.2, -.27, -.05), shader=unlit_shader, enabled=False)
        self.cam_title_sh = Text("CAMERA CONTROL", parent=ui, origin=(0, 0), position=(.006, .394, -.04), scale=2.6, color=color.rgb32(40, 60, 56),
                                 enabled=False)
        self.cam_title = Text("CAMERA CONTROL", parent=ui, origin=(0, 0), position=(0, .4, -.05), scale=2.6, color=color.rgb32(255, 226, 120),
                              enabled=False)
        self.cam_help_bg = Entity(parent=ui, model="quad", color=color.rgba(0.04, 0.22, 0.3, 0.8), scale=(.5, .34), position=(-.52, .1, .02),
                                  shader=unlit_shader, enabled=False)
        self.cam_help = Text("Stand back so your upper\nbody and hips are in view.\n\nStep left / right to steer.\nJump to jump.\n\nEsc = back to the menu",
                             parent=ui, origin=(0, 0), position=(-.52, .1, -.05), scale=1.15, color=color.white, enabled=False)
        self.show_pip = True
        self._jump_prev = False
        self.end_t = None
        self.cam_error = None
        self.input_mode = "keyboard"
        self.after_calib = "curl"

        # pause: only a small badge in the top-right corner, so the whole scene stays visible
        self.pause_ui = [
            Entity(parent=ui, model="quad", color=color.rgba(0.04, 0.22, 0.3, 0.8), scale=(.36, .085), position=(.68, .445, .02),
                   shader=unlit_shader, enabled=False),
            Entity(parent=ui, model="quad", color=color.rgb32(255, 226, 120), scale=(.011, .038), position=(.552, .448, -.05),
                   shader=unlit_shader, enabled=False),
            Entity(parent=ui, model="quad", color=color.rgb32(255, 226, 120), scale=(.011, .038), position=(.574, .448, -.05),
                   shader=unlit_shader, enabled=False),
            Text("PAUSED", parent=ui, origin=(-.5, 0), position=(.595, .455, -.05), scale=1.6, color=color.rgb32(255, 226, 120), enabled=False),
            Text("P resume   R restart   Q quit", parent=ui, origin=(-.5, 0), position=(.55, .418, -.05), scale=.8,
                 color=color.rgb32(235, 245, 225), enabled=False),
        ]
        # the layout above is drawn for a 16:9 picture (half-width .889); fit_view slides the left-hand and right-hand items so they
        # keep their distance from the edge on any other shape (16:10 laptops, 4:3, long phones)
        self.lshift = 0.0
        self._left_items = [(e, e.x) for e in self.hud + [self.cam_help_bg, self.cam_help]]
        self._right_items = [(e, e.x) for e in self.pause_ui]

    def set_paused(self, value):
        self.paused = value
        for e in self.pause_ui:
            e.enabled = value

    def _hide_menu(self):
        for e in self.start_ui:
            e.enabled = False

    def _calib_ui(self, on):
        for e in (self.cam_frame, self.cam_view, self.cam_status, self.cam_sub, self.cam_bar_bg, self.cam_bar, self.cam_title, self.cam_title_sh, self.cam_help, self.cam_help_bg):
            e.enabled = on
        if on:                                            # big view in the middle-right
            self.cam_status.scale, self.cam_status.y, self.cam_sub.y = 1.5, -.32, -.38
            self.cam_frame.scale, self.cam_frame.position = (.44, .34), (.5 - self.lshift, .12, .02)
            self.cam_view.scale, self.cam_view.position = (.4, .3), (.5 - self.lshift, .12, 0)

    def go_await(self):
        """Camera mode: stand in place and wait for the first jump, so you have time to step back from the computer."""
        self.phase = "await"
        self._jump_prev = True                              # a jump already in progress doesn't count
        for e in self.hud:
            e.enabled = False
        self.hint.enabled = False
        self.cam_status.enabled = self.cam_sub.enabled = True
        self.cam_status.scale, self.cam_status.y, self.cam_sub.y = 3.0, -.3, -.4
        self.cam_bar_bg.enabled = self.cam_bar.enabled = False

    def start_keyboard(self):
        if self.phase == "ready":
            self.input_mode = "keyboard"
            active_camera[0] = None
            self.hint.text = "Left / Right  or  A / D  to steer     Space to jump     P: pause"
            self.phase, self.curl_u = "curl", 0.0
            self._hide_menu()

    def start_camera(self):
        if self.phase != "ready":
            return
        self.input_mode = "camera"
        self._hide_menu()
        if PoseCamera is None:
            self.cam_error = "Camera control needs OpenCV and MediaPipe (py -m pip install opencv-python mediapipe==0.10.21)"
        else:
            self.cam_error = None
            if active_camera[0] is None:
                active_camera[0] = PoseCamera()
            active_camera[0].start()
        self.hint.text = "Move left / right to steer  -  jump to jump     C: camera view     B: recalibrate     P: pause"
        self.after_calib = "curl"
        self.phase = "calib"
        self._calib_ui(True)

    def start(self):
        self.start_keyboard()

    def to_menu(self):
        cam = active_camera[0]
        if cam is not None:
            cam.stop()
        active_camera[0] = None
        self._calib_ui(False)
        self.reset(intro=True)

    def quit_game(self):
        if active_camera[0] is not None:
            active_camera[0].stop()
        application.quit()

    def update_camera_ui(self):
        """Refresh the live camera picture and the calibration / status texts (every frame, whatever phase)."""
        cam = active_camera[0]
        frame = None if cam is None else cam.preview
        if frame is not None:
            self.cam_tex.setRamImageAs(np.ascontiguousarray(frame[::-1]).tobytes(), "RGB")
        err = (cam.error if cam is not None else None) or self.cam_error
        if cam is None and not err:
            if self.phase != "calib":
                self.cam_view.enabled = self.cam_frame.enabled = False
            return
        if self.phase == "calib":
            m = cam.mapper if cam is not None else None
            if err:
                self.cam_status.text, self.cam_status.color = err, color.rgb32(255, 150, 130)
                self.cam_sub.text = "Press Esc to go back and use the keyboard instead"
                self.cam_bar_bg.enabled = self.cam_bar.enabled = False
                return
            if not cam.opened:
                self.cam_status.text, self.cam_status.color = "Opening the camera ...  (a few seconds)", color.white
                self.cam_sub.text = ""
                return
            self.cam_status.text = STATUS_TEXT.get(m.status, "")
            self.cam_status.color = color.rgb32(140, 240, 160) if m.status in ("ready", "calibrating") else color.rgb32(255, 226, 120)
            self.cam_sub.text = "Hold still for a moment to calibrate" if m.status == "calibrating" else ""
            self.cam_bar.scale_x = max(.0001, .4 * m.progress)
            if m.calibrated:
                self._calib_ui(False)
                self.curl_u = 0.0
                if self.after_calib == "play":
                    self.phase = "play"
                    for e in self.hud:
                        e.enabled = True
                else:
                    self.go_await()
        elif self.phase == "await":
            self.cam_status.text, self.cam_status.color = "JUMP TO START!", color.rgb32(255, 226, 120)
            self.cam_sub.text = "Take your time - get into position, then jump" if cam.mapper.status == "ready" else STATUS_TEXT.get(cam.mapper.status, "")
            self.cam_view.enabled = self.cam_frame.enabled = frame is not None and self.show_pip
            if self.cam_view.enabled:
                self.cam_frame.scale, self.cam_frame.position = (.25, .19), (.72 - self.lshift, -.31, .02)
                self.cam_view.scale, self.cam_view.position = (.23, .1725), (.72 - self.lshift, -.31, 0)
        else:
            show = self.show_pip and self.phase in ("play", "curl") and frame is not None
            self.cam_frame.enabled = self.cam_view.enabled = show
            if show:
                self.cam_frame.scale, self.cam_frame.position = (.25, .19), (.72 - self.lshift, -.31, .02)
                self.cam_view.scale, self.cam_view.position = (.23, .1725), (.72 - self.lshift, -.31, 0)
            m = cam.mapper
            if self.phase == "play" and m.status == "lost":
                self.hint.text = "I lost you - step back into view  (Esc / B to recalibrate)"
                self.hint.alpha = 1

    def reset(self, intro=False):
        self.set_paused(False)
        self.s = BallState()
        self.t = 0.0
        self.win_t0 = None
        self.end_t = None
        self.phase = "ready" if intro else "play"
        self.idle_t = self.curl_u = 0.0
        if intro:
            self._calib_ui(False)
            self.cam_view.enabled = self.cam_frame.enabled = False
        for e in self.start_ui:
            e.enabled = intro
        for e in self.hud:
            e.enabled = not intro
        for p in self.parts:
            p[0].enabled = False
            self.pool.append(p[0])
        self.parts = []
        self.hint.enabled = not intro
        self.hint.alpha = 1
        for e in (self.msg_bg, self.msg, self.msg2):
            e.enabled = False
        self.cam_x = 0.0
        self.sync(1.0, snap=True)

    # ---- particles
    def spawn(self, n, col, speed=.08, up=.12, size=.14, life=30, grav=.006, at=None):
        s = self.s
        r = random.uniform
        bx, by, bz = at or (s.x, s.y + .1, s.z)
        for _ in range(n):
            e = self.pool.pop() if self.pool else Entity(model="icosphere", shader=unlit_shader)
            e.enabled = True
            e.color = col
            sz = size * r(.6, 1.2)
            e.scale = sz
            e.position = (bx + float(curve(bz)) + r(-.3, .3), by, bz + r(-.3, .3))
            self.parts.append([e, r(-speed, speed), r(0, up), r(-speed, speed), life + r(-8, 8), life, sz, grav])

    def effects(self, ev):
        dust = c3(240, 196, 136)
        s = self.s
        for e in ev:
            if e == "trail" and int(self.t * 60) % 3 == 0:
                self.spawn(1, dust, speed=.01, up=.02, size=.12, life=20, grav=-.0006, at=(s.x, s.y + .1, s.z - .4))
            elif e == "land":
                self.spawn(10, dust, speed=.07, up=.05, size=.13, life=24, grav=-.0005, at=(s.x, s.y + .1, s.z))
            elif e == "jump":
                self.spawn(5, dust, speed=.05, up=.03, size=.12, life=20, grav=-.0005)
            elif e == "crash":
                self.spawn(30, c3(130, 130, 135), speed=.12, up=.2, size=.22)
                self.spawn(12, c3(220, 70, 60), speed=.1, up=.2, size=.18)
            elif e == "fell":
                self.spawn(26, c3(176, 206, 230), speed=.09, up=.2, size=.14, life=28)
            elif e == "win":
                for _ in range(90):
                    self.spawn(1, random.choice((c3(255, 90, 90), c3(255, 210, 80), c3(90, 200, 255), c3(140, 230, 120), c3(255, 150, 230))),
                               speed=.06, up=.3, size=.18, life=100, grav=.008, at=(s.x + random.uniform(-1.5, 1.5), 1.0, s.z + random.uniform(-1, 1.5)))

    def step_parts(self, f):
        keep = []
        for p in self.parts:
            e, vx, vy, vz, life, maxlife, sz, grav = p
            e.x += vx * f; e.y += vy * f; e.z += vz * f
            p[2] = vy - grav * f
            p[4] = life - f
            if p[4] > 0:
                e.scale = sz * max(0.05, (p[4] / maxlife) ** .6)
                keep.append(p)
            else:
                e.enabled = False
                self.pool.append(e)
        self.parts = keep

    # ---- per-frame
    def sync(self, f, snap=False):
        s = self.s
        if self.phase in ("ready", "calib", "await"):
            self.panda.pose_idle(self.idle_t, s)
        elif self.phase == "curl":
            self.panda.pose_curl(self.curl_u, s)
        else:
            wu = None if self.win_t0 is None else (self.t - self.win_t0) / UNCURL_T
            self.panda.pose_play(s, self.t, wu)
        k = 1.0 if snap else 1 - 0.85 ** f
        self.cam_x += (s.x * 0.6 - self.cam_x) * k
        c = {"ready": 0.0, "calib": 0.0, "await": 0.0, "curl": ease(self.curl_u)}.get(self.phase, 1.0)                  # 0 = close-up on the panda, 1 = follow cam
        play_pos = (self.cam_x + float(curve(s.z - 6.0)), 3.0 + max(0.0, s.y) * 0.3, s.z - 6.6)
        near_pos = (float(curve(s.z - 4.2)), 1.45, s.z - 4.5)
        camera.position = tuple(n + (p - n) * c for n, p in zip(near_pos, play_pos))
        yaw = .7 * math.degrees(math.atan(float(curve_slope(s.z + 4))))
        camera.rotation = (7 + (CAM_PITCH - 7) * c, yaw * c, 0)                             # look along the bend
        if backdrop_rig is not None:
            backdrop_rig.rotation_y = -camera.rotation_y                                                 # painted sky stays put
            lateral = camera.x - float(curve(s.z))                                                       # sideways offset from the trail's line
            for e, par in backdrop_layers:
                e.x = -lateral * par                                                                      # distant layers slide a little: parallax
            if cloud_layer is not None:
                cloud_layer.texture_offset = (self.idle_t * .004 + self.t * .004, 0)                      # clouds drift slowly
        self.focus.position = (float(curve(s.z + 8)), 0, s.z + 8)
        if snap or int(self.t * 60) % 4 == 0:
            self.sun.update_bounds(self.focus)
        self.bar.scale_x = max(.0001, .3 * min(1.0, s.z / (FINISH_ROW * TILE)))

    def tick(self, act, f=1.0):
        cam = active_camera[0] if self.input_mode == "camera" else None
        jump_now = bool(cam is not None and cam.mapper.status == "ready" and cam.mapper.jump)
        jump_edge = jump_now and not self._jump_prev
        self._jump_prev = jump_now
        if self.phase == "await":
            self.idle_t += f / 60
            if jump_edge:
                self.phase, self.curl_u = "curl", 0.0
                self.cam_status.enabled = self.cam_sub.enabled = False
            self.sync(f)
            return
        if self.phase != "play":
            self.idle_t += f / 60
            if self.phase == "curl":
                self.curl_u += f / 60 / CURL_T
                if self.curl_u >= 1:
                    self.phase = "play"
                    self.panda.set_form(1)
                    for e in self.hud:
                        e.enabled = True
                    self.hint.enabled = True
            self.sync(f)
            return
        self.t += f / 60
        ev = self.s.update(act, f)
        if "win" in ev:
            self.win_t0 = self.t
        self.effects(ev)
        self.step_parts(f)
        self.sync(f)
        if self.t > 6:
            self.hint.alpha = max(0, self.hint.alpha - .02 * f)
        if self.s.state != "play" and self.end_t is None:
            self.end_t = self.t
            self.msg2.text = "Jump to play again" if cam is not None else "Press R to play again"
        if self.s.state != "play" and cam is not None and jump_edge and self.t - self.end_t > 1.5:
            self.reset()                                    # a jump restarts straight away
            self.phase, self.curl_u = "curl", 0.0
            return
        if self.s.state != "play":
            title = {"win": "You made it!", "fell": "Splash! Into the puddle", "crashed": "Hit a rock!"}[self.s.state]
            self.msg.text = title
            self.msg.color = color.rgb32(255, 220, 90) if self.s.state == "win" else color.rgb32(255, 140, 120)
            for e in (self.msg_bg, self.msg, self.msg2):
                e.enabled = True

    def fit_view(self):
        """Keep the picture between MIN_ASPECT and MAX_ASPECT (black bars outside that), however the window is shaped or resized."""
        w, h = base.win.getXSize(), base.win.getYSize()
        if h <= 0 or (w, h) == getattr(self, "_view_size", None) and abs(camera.perspective_lens.get_aspect_ratio() - self._view_aspect) < 1e-4:
            return
        asp = w / h
        t = min(MAX_ASPECT, max(MIN_ASPECT, asp))
        fx, fy = min(1.0, t / asp), min(1.0, asp / t)
        dims = ((1 - fx) / 2, (1 + fx) / 2, (1 - fy) / 2, (1 + fy) / 2)
        for region in (camera.display_region, camera.ui_display_region):
            region.set_dimensions(*dims)
        camera.display_region.set_clear_color_active(True)
        camera.display_region.set_clear_color(FOG)                        # the picture area keeps the haze colour, the bars are black
        base.win.set_clear_color((0, 0, 0, 1))
        camera.perspective_lens.set_aspect_ratio(t)
        camera.ui_lens.set_film_size(camera._ui_size * .5 * t, camera._ui_size * .5)
        self._view_size, self._view_aspect = (w, h), t
        self.lshift = .889 - t / 2
        for e, x0 in getattr(self, "_left_items", []):
            e.x = x0 + self.lshift
        for e, x0 in getattr(self, "_right_items", []):
            e.x = x0 - self.lshift

    def update(self):
        self.fit_view()
        if not self.test_mode:
            self.update_camera_ui()
        if not self.test_mode and not self.paused:
            self.tick(get_actions(), min(time.dt * 60, 3.0))

    def input(self, key):
        if self.phase == "ready":
            if key in ("space", "enter", "1", "k"):
                self.start_keyboard()
            elif key in ("2", "c"):
                self.start_camera()
            elif key == "escape":
                self.quit_game()
        elif self.phase in ("calib", "await"):
            if key == "escape":
                self.to_menu()
        elif self.phase == "curl":
            if key == "escape":
                self.quit_game()
        elif key == "r":
            self.reset()                                   # restart straight away, already rolling
            if self.input_mode == "camera":
                self.go_await()                            # ...except with the camera: wait for your jump
        elif key == "c" and self.input_mode == "camera":
            self.show_pip = not self.show_pip
        elif key == "b" and self.input_mode == "camera" and active_camera[0] is not None:
            active_camera[0].recalibrate()                 # stand in the middle again, then carry on where you were
            self.set_paused(False)
            self.after_calib = "play"
            self.phase = "calib"
            self._calib_ui(True)
        elif key in ("p", "escape"):
            if self.s.state == "play":
                self.set_paused(not self.paused)           # pause / resume
            elif key == "escape":
                self.quit_game()                           # on the game-over screen Esc still quits
        elif key == "q" and self.paused:
            self.quit_game()


def create_app():
    offscreen = os.environ.get("WHEEL_TEST")
    app = Ursina(title="Forest Trail Runner", borderless=False, size=Vec2(1280, 720),
                 window_type="offscreen" if offscreen else "onscreen", development_mode=False)
    if offscreen:
        window.update_aspect_ratio()
    window.exit_button.visible = False
    window.fps_counter.enabled = False
    Entity.default_shader = lit_with_shadows_shader
    lit_with_shadows_shader.default_input["shadow_color"] = color.rgba(0.30, 0.26, 0.56, 0.32)
    global foliage_shader, toon_shader, toon_leaf_shader
    foliage_shader = make_foliage_shader()
    toon_shader = make_toon_shader()
    toon_leaf_shader = make_toon_leaf_shader()
    return app


def main():
    app = create_app()
    window.color = color.rgb32(40, 90, 80)
    loading = Text("Loading the forest ...", origin=(0, 0), scale=2.2, color=color.rgb32(255, 226, 120))
    app.step()                                      # draw it once before the (several seconds long) world build
    Game()
    destroy(loading)
    app.run()


if __name__ == "__main__":
    main()
