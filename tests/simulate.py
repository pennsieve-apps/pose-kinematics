"""
A simulated mouse in a circular open field, seen from above, with three planted behaviours:

  rest  - still, compact body (head tucked, ~6.5 cm nose to tail base), small postural jitter
  walk  - moving forward at ~16 cm/s along its heading, elongated body (~9 cm), gentle turns
  rear  - nearly still, body foreshortened (~4 cm: the head goes up, so the nose projects close to
          the body centre), ears pulled in

Behaviours follow a semi-Markov chain (gamma-distributed durations); the animal turns away from
the wall. `center_bias` pulls walks towards the arena centre (more centre time), `walk_bias`
makes walking more frequent. Six keypoints: nose, leftear, rightear, headcenter, bodycenter,
tailbase; positions in cm, arena centred at (0, 0).

Only numpy; used by tests/make_fixture.py (and, in behavior-syllables, by its own copy).
"""
from __future__ import annotations

import numpy as np

KEYPOINTS = ["nose", "leftear", "rightear", "headcenter", "bodycenter", "tailbase"]
STATES = ["rest", "walk", "rear"]
# Body shape per state, in the animal's frame (x forward, y left), cm, relative to the body centre.
SHAPES = {
    "rest": {"nose": (3.6, 0.0), "leftear": (2.2, 1.0), "rightear": (2.2, -1.0), "headcenter": (2.4, 0.0),
             "bodycenter": (0.0, 0.0), "tailbase": (-2.9, 0.0)},
    "walk": {"nose": (5.2, 0.0), "leftear": (3.4, 0.8), "rightear": (3.4, -0.8), "headcenter": (3.7, 0.0),
             "bodycenter": (0.0, 0.0), "tailbase": (-3.8, 0.0)},
    "rear": {"nose": (1.4, 0.0), "leftear": (0.9, 0.55), "rightear": (0.9, -0.55), "headcenter": (0.9, 0.0),
             "bodycenter": (0.0, 0.0), "tailbase": (-2.6, 0.0)},
}
MEAN_DURATION = {"rest": 1.6, "walk": 2.0, "rear": 1.2}
SPEED = {"rest": 0.0, "walk": 16.0, "rear": 0.6}


def simulate(seconds: float, fps: float, radius: float, seed: int, center_bias: float = 0.0,
             walk_bias: float = 1.0):
    rng = np.random.default_rng(seed)
    n = int(round(seconds * fps))
    dt = 1.0 / fps
    # State sequence.
    state = np.empty(n, dtype=object)
    i, s = 0, "walk"
    weights = {"rest": 1.0, "walk": walk_bias, "rear": 0.7}
    while i < n:
        d = max(0.3, rng.gamma(4.0, MEAN_DURATION[s] / 4.0))
        k = int(round(d * fps))
        state[i:i + k] = s
        i += k
        choices = [x for x in STATES if x != s]
        p = np.array([weights[x] for x in choices])
        s = choices[rng.choice(len(choices), p=p / p.sum())]
    # Trajectory.
    pos = np.zeros((n, 2))
    heading = np.zeros(n)
    x = rng.uniform(-0.5, 0.5, 2) * radius
    h = rng.uniform(-np.pi, np.pi)
    body = {k: np.array(v, float) for k, v in SHAPES["walk"].items()}
    kp = np.zeros((n, len(KEYPOINTS), 2))
    turn = 0.0
    for t in range(n):
        st = state[t]
        if st == "walk":
            turn = 0.9 * turn + rng.normal(0, 0.35)
            # Turn away from the wall when close; pulled towards the centre by center_bias.
            r = np.linalg.norm(x)
            to_center = np.arctan2(-x[1], -x[0])
            diff = (to_center - h + np.pi) % (2 * np.pi) - np.pi
            pull = center_bias * 0.6 + (2.5 if r > radius - 8 else 0.0)
            h += (turn + pull * diff * (0.3 if r > radius - 8 else 0.08)) * dt * 3
            x = x + SPEED[st] * dt * np.array([np.cos(h), np.sin(h)])
        else:
            x = x + rng.normal(0, SPEED[st] * dt, 2)
            h += rng.normal(0, 0.02)
        # Keep the body inside the arena.
        r = np.linalg.norm(x)
        if r > radius - 4:
            x = x * (radius - 4) / r
        pos[t] = x
        heading[t] = h
        # Smoothly morph the body shape towards the current state's.
        target = SHAPES[st]
        for k in KEYPOINTS:
            body[k] += 0.25 * (np.array(target[k]) - body[k])
        c, s_ = np.cos(h), np.sin(h)
        rot = np.array([[c, -s_], [s_, c]])
        for j, k in enumerate(KEYPOINTS):
            kp[t, j] = x + rot @ (body[k] + rng.normal(0, 0.05, 2))
    return {"t": np.arange(n) * dt, "state": state.astype(str), "keypoints": kp, "centroid": pos,
            "heading": heading}


def to_pixels(kp_cm, px_per_cm: float, center_px):
    """cm (arena centre at 0, y up) -> pixels (top-left origin, y down)."""
    out = np.empty_like(kp_cm)
    out[..., 0] = center_px[0] + kp_cm[..., 0] * px_per_cm
    out[..., 1] = center_px[1] - kp_cm[..., 1] * px_per_cm
    return out
