"""Full-body camera control for the game.

MediaPipe Pose (the lite BlazePose model, 33 body landmarks, runs on the CPU) finds the player's skeleton in each webcam frame.
MotionMapper turns the skeleton into the game's three actions:

    stand in front of the camera, with your upper body and hips in view  ->  calibration (hold still for about a second)
    step / lean left or right of where you calibrated                    ->  steer left / right (smooth, proportional)
    jump (shoulders rise)                                                ->  jump

The camera image is mirrored, so moving to *your* left moves the panda left.

Run `py pose_control.py` on its own to see the camera view with the skeleton and the live readings (press Q to close it).
"""
import threading
import time

import numpy as np

# MediaPipe pose landmark indices
L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 23, 24, 25, 26, 27, 28

CALIB_FRAMES = 30          # frames of a steady body needed to calibrate
DEAD_ZONE = .05            # sideways offset (in body-scale units) that is ignored, so tiny sways don't move the ball
FULL_STEER = .5            # offset (in body-scale units, about one shoulder width) that puts the ball at the edge of the trail
JUMP_ON, JUMP_OFF = .14, .06   # how far the shoulders must rise above their recent height (body-scale units) to start / end a jump
JUMP_MAX_T = .9            # a jump flag never lasts longer than this (seconds)
JUMP_COOLDOWN = .55        # seconds after a jump before another can register
LOST_AFTER = .6            # seconds without a body before steering stops
HIST_NEAR, HIST_FAR = .35, 1.6   # the "recent height" is the median shoulder height from this long ago (seconds): slow leans, slouching
                                 # and drift are absorbed into it, only a quick rise stands out as a jump


class MotionMapper:
    """Pure logic (no camera): feed it landmarks, read left / right / jump back. Coordinates are normalised image coordinates
    (0..1, y down), already mirrored. Only the SHOULDERS are used (the hips are often off the bottom of the picture and their
    guessed positions jitter), and body scale is the shoulder width, so it also works when you sit close to the camera."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.calibrated = False
        self.progress = 0.0
        self._buf = []
        self._hist = []                     # (time, shoulder y)
        self._raw = []
        self.x0 = self.y0 = self.scale = None
        self.sx = self.sy = None            # smoothed shoulder-line centre
        self.rise = 0.0                     # shoulders above their recent height, in body-scale units (for the test window)
        self.steer = 0.0
        self.jump = False
        self._jump_t = -9.0
        self.status = "searching"
        self.last_seen = 0.0

    @staticmethod
    def _body(lm):
        """(shoulder centre x, y, body scale, visible) from 33 landmarks of (x, y, visibility)."""
        l, r = lm[L_SH], lm[R_SH]
        width = abs(l[0] - r[0])
        return (l[0] + r[0]) / 2, (l[1] + r[1]) / 2, max(1e-3, width * 1.7), min(l[2], r[2]) > .6 and width > .05

    def update(self, lm, now=None):
        """lm: list of 33 (x, y, visibility) or None when nobody was found. Returns the current status string."""
        now = time.time() if now is None else now
        if lm is None:
            if now - self.last_seen > LOST_AFTER:
                self.steer, self.jump = 0.0, False
                self.status = "lost" if self.calibrated else "searching"
            return self.status
        cx, cy, scale, ok = self._body(lm)
        if not ok:
            if now - self.last_seen > LOST_AFTER:
                self.steer, self.jump = 0.0, False
                self.status = "adjust"
            return self.status
        self.last_seen = now
        self._raw = (self._raw + [(cx, cy)])[-3:]                       # median of 3 frames kills single-frame spikes
        mx = float(np.median([q[0] for q in self._raw]))
        my = float(np.median([q[1] for q in self._raw]))
        a = .5
        self.sx = mx if self.sx is None else self.sx + (mx - self.sx) * a
        self.sy = my if self.sy is None else self.sy + (my - self.sy) * a

        if not self.calibrated:
            self._buf.append((mx, my, scale))
            self._buf = self._buf[-CALIB_FRAMES:]
            arr = np.array(self._buf)
            steady = len(arr) < 6 or (arr[:, 0].std() < .012 and arr[:, 1].std() < .012)      # standing still
            if not steady:
                self._buf = self._buf[-2:]                                  # moved: start collecting again
            self.progress = min(1.0, len(self._buf) / CALIB_FRAMES) if steady else 0.0
            self.status = "calibrating"
            if len(self._buf) >= CALIB_FRAMES and steady:
                self.x0, self.y0, self.scale = float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())
                self.calibrated, self.status, self.progress = True, "ready", 1.0
                self._hist = [(now, self.y0)]
            return self.status

        self.status = "ready"
        dx = (self.sx - self.x0) / self.scale
        mag = max(0.0, abs(dx) - DEAD_ZONE) / (FULL_STEER - DEAD_ZONE)
        self.steer = float(np.sign(dx) * min(1.0, mag))

        # jump = the shoulders quickly rise above where they have been for the last second or so
        old = [y for t, y in self._hist if HIST_NEAR <= now - t <= HIST_FAR]
        self._hist = [(t, y) for t, y in self._hist if now - t <= HIST_FAR + .2] + [(now, self.sy)]
        if len(old) >= 5:
            self.rise = (float(np.median(old)) - self.sy) / max(scale, self.scale * .6)
        else:
            self.rise = 0.0
        if not self.jump:
            if self.rise > JUMP_ON and now - self._jump_t > JUMP_COOLDOWN:
                self.jump, self._jump_t = True, now
        elif self.rise < JUMP_OFF or now - self._jump_t > JUMP_MAX_T:
            self.jump, self._jump_t = False, now
        return self.status

    def actions(self):
        s = self.steer if self.status == "ready" else 0.0
        # "target" = where the ball should be across the trail (-1 left edge .. +1 right edge): the ball follows your body position
        return {"left": max(0.0, -s), "right": max(0.0, s), "jump": 1 if (self.jump and self.status == "ready") else 0,
                "target": s if self.status == "ready" else None}


def _worker():
    """Child process: webcam + MediaPipe. Writes one message per frame to stdout: [json length][image length][json][RGB image].
    It lives in its own process because MediaPipe's native libraries fail to load once the 3D engine is running in the same process."""
    import json
    import struct
    import sys
    out = sys.stdout.buffer

    def send(obj, img=None):
        data = json.dumps(obj).encode()
        blob = b"" if img is None else img.tobytes()
        out.write(struct.pack("<ii", len(data), len(blob)) + data + blob)
        out.flush()

    try:
        import cv2
        import mediapipe as mp
        cap = None
        for backend in (getattr(cv2, "CAP_DSHOW", 0), 0):              # DirectShow opens faster on Windows
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                break
        if cap is None or not cap.isOpened():
            send({"error": "No camera found (check that it is connected and not used by another app)"})
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        pose = mp.solutions.pose.Pose(model_complexity=0, smooth_landmarks=True, min_detection_confidence=.5, min_tracking_confidence=.5)
        drawing, styles = mp.solutions.drawing_utils, mp.solutions.drawing_styles
        send({"opened": True})
        n, t0, fps = 0, time.time(), 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(.01)
                continue
            rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)   # mirrored: the player's left is the screen's left
            res = pose.process(rgb)
            small = cv2.resize(rgb, (320, 240))
            lm = None
            if res.pose_landmarks:
                lm = [(round(q.x, 4), round(q.y, 4), round(q.visibility, 3)) for q in res.pose_landmarks.landmark]
                drawing.draw_landmarks(small, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                       landmark_drawing_spec=styles.get_default_pose_landmarks_style())
            n += 1
            if n % 20 == 0:
                fps, t0 = 20 / max(1e-3, time.time() - t0), time.time()
            send({"lm": lm, "fps": fps}, small)
    except (BrokenPipeError, OSError):
        pass                                                           # the game closed
    except Exception as e:
        try:
            send({"error": f"Camera control failed: {e}"})
        except Exception:
            pass


class PoseCamera:
    """Starts the camera worker process and reads its results in a background thread, so the game never waits for the camera.
    Read .mapper for actions / status, .preview (small RGB picture with the skeleton drawn on) for on-screen feedback."""

    def __init__(self):
        self.mapper = MotionMapper()
        self.preview = None
        self.error = None
        self.opened = False
        self.fps = 0.0
        self._proc = None
        self._thread = None

    def start(self):
        import os
        import subprocess
        import sys
        if self._proc is not None and self._proc.poll() is None:
            return
        self.error, self.opened, self.preview = None, False, None
        self.mapper.reset()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._proc = subprocess.Popen([sys.executable] + ([] if getattr(sys, "frozen", False) else [os.path.abspath(__file__)]) + ["--worker"], stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, creationflags=flags)
        except Exception as e:
            self.error = f"Could not start the camera process ({e})"
            return
        self._thread = threading.Thread(target=self._read, args=(self._proc,), daemon=True)
        self._thread.start()

    def _read(self, proc):
        import json
        import struct

        def take(n):
            buf = b""
            while len(buf) < n:
                chunk = proc.stdout.read(n - len(buf))
                if not chunk:
                    raise EOFError
                buf += chunk
            return buf
        try:
            while True:
                jl, il = struct.unpack("<ii", take(8))
                msg = json.loads(take(jl))
                img = take(il) if il else None
                if "error" in msg:
                    self.error = msg["error"]
                if msg.get("opened"):
                    self.opened = True
                if "lm" in msg:
                    self.mapper.update(msg["lm"])
                    self.fps = msg.get("fps", 0.0)
                if img is not None:
                    self.preview = np.frombuffer(img, np.uint8).reshape(240, 320, 3)
        except Exception:
            if self.error is None and self._proc is proc and not self.opened:
                self.error = "The camera process stopped before it started (is mediapipe installed? py -m pip install mediapipe==0.10.21)"
            self.opened = False

    def stop(self):
        proc, self._proc = self._proc, None
        self.opened = False
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def recalibrate(self):
        self.mapper.reset()


STATUS_TEXT = {
    "searching": "Step into view - I need to see your upper body and hips",
    "adjust": "Move back a little so your shoulders and hips are in view",
    "calibrating": "Stand still in the middle ...",
    "ready": "Ready!  Step left / right to steer, jump to jump",
    "lost": "I lost you - step back into view",
}


if __name__ == "__main__":
    import sys
    if "--worker" in sys.argv:
        _worker()
        raise SystemExit
    import cv2                                                         # stand-alone camera test
    cam = PoseCamera()
    cam.start()
    print("Camera test: stand back so your upper body is visible, hold still to calibrate, then step left / right and jump. Q closes.")
    while True:
        if cam.error:
            print(cam.error)
            break
        if cam.preview is not None:
            img = cv2.cvtColor(cam.preview, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (640, 480))
            m = cam.mapper
            a = m.actions()
            cv2.putText(img, f"{m.status}  steer {m.steer:+.2f}  rise {m.rise:+.2f}  jump {a['jump']}  fps {cam.fps:.0f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            if m.status == "calibrating":
                cv2.rectangle(img, (10, 40), (10 + int(300 * m.progress), 55), (0, 255, 0), -1)
            cv2.imshow("pose test", img)
        if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
            break
    cam.stop()
    cv2.destroyAllWindows()
