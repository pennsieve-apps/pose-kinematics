"""PNG figures, 150 dpi, session id in the title."""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 150
GROUP_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#008300", "#eda100", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.titlesize": 10, "figure.facecolor": "white"})


# Sequential, one hue light -> dark; the lightest quarter of Blues is dropped so early/rare points stay visible.
_TIME_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blues_trunc", plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256)))


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", s)


def _arena(ax, arena, rois):
    if arena is None:
        return
    if arena.outline is not None:
        ax.plot(arena.outline[:, 0], arena.outline[:, 1], color=MUTED, lw=1)
    t = np.linspace(0, 2 * np.pi, 200)
    if arena.shape == "circle":
        if arena.outline is None:
            ax.plot(arena.cx + arena.rx * np.cos(t), arena.cy + arena.ry * np.sin(t), color=MUTED, lw=1)
    elif arena.outline is None:
        ax.add_patch(plt.Rectangle((arena.cx - arena.rx, arena.cy - arena.ry), 2 * arena.rx, 2 * arena.ry,
                                   fill=False, ec=MUTED, lw=1))
    for name, outline in rois:
        ax.plot(outline[:, 0], outline[:, 1], color="#6250d6", lw=1.2)
        ax.text(outline[:, 0].mean(), outline[:, 1].mean(), name, color="#6250d6", fontsize=7, ha="center")


def _center(ax, arena, frac):
    if arena is None:
        return
    t = np.linspace(0, 2 * np.pi, 200)
    if arena.shape == "circle":
        ax.plot(arena.cx + frac * arena.rx * np.cos(t), arena.cy + frac * arena.ry * np.sin(t), color=MUTED,
                lw=0.8, ls="--")
    else:
        ax.add_patch(plt.Rectangle((arena.cx - frac * arena.rx, arena.cy - frac * arena.ry), 2 * frac * arena.rx,
                                   2 * frac * arena.ry, fill=False, ec=MUTED, lw=0.8, ls="--"))


def _limits(ax, arena, x, y):
    if arena is not None:
        pad = 0.05 * max(arena.rx, arena.ry)
        ax.set_xlim(arena.cx - arena.rx - pad, arena.cx + arena.rx + pad)
        ax.set_ylim(arena.cy + arena.ry + pad, arena.cy - arena.ry - pad)   # y down, as in the video
    else:
        ax.invert_yaxis()
    ax.set_aspect("equal")


def trajectory(sid, ts, x, y, speed, arena, center_frac, rois, units, out):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ok = np.isfinite(x) & np.isfinite(y)
    ax.plot(x, y, color=GRID, lw=0.5, zorder=1)
    sc = ax.scatter(x[ok], y[ok], c=ts[ok] - ts[ok][0] if ok.any() else None, s=1.2, cmap=_TIME_CMAP, zorder=2)
    _arena(ax, arena, rois)
    _center(ax, arena, center_frac)
    _limits(ax, arena, x, y)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("time since first frame (s)")
    ax.set_xlabel(f"x ({units})")
    ax.set_ylabel(f"y ({units})")
    ax.set_title(f"{sid}: trajectory")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def occupancy(sid, x, y, dur, arena, center_frac, rois, units, out):
    ok = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    if arena is not None:
        xr = (arena.cx - arena.rx, arena.cx + arena.rx)
        yr = (arena.cy - arena.ry, arena.cy + arena.ry)
    else:
        xr, yr = (np.nanmin(x), np.nanmax(x)), (np.nanmin(y), np.nanmax(y))
    nb = 40
    h, xe, ye = np.histogram2d(x[ok], y[ok], bins=nb, range=[xr, yr], weights=dur[ok])
    h = np.where(h > 0, h, np.nan)
    from matplotlib.colors import LogNorm

    vmax = np.nanmax(h) if np.isfinite(h).any() else 1.0
    im = ax.imshow(h.T, origin="upper", extent=(xe[0], xe[-1], ye[-1], ye[0]), cmap=_TIME_CMAP, aspect="equal",
                   interpolation="nearest", norm=LogNorm(vmin=max(vmax / 1000, np.nanmin(h)), vmax=vmax))
    _arena(ax, arena, rois)
    _center(ax, arena, center_frac)
    _limits(ax, arena, x, y)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(f"seconds per bin, log scale ({(xe[1] - xe[0]):.2g} x {(ye[1] - ye[0]):.2g} {units})")
    ax.set_xlabel(f"x ({units})")
    ax.set_ylabel(f"y ({units})")
    ax.set_title(f"{sid}: occupancy")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def speed_distribution(per_session, groups, units, out):
    """per_session: {sid: speed array}; groups: {sid: group or None}."""
    names = sorted({g for g in groups.values() if g}) or [None]
    colour = {g: GROUP_COLOURS[i % len(GROUP_COLOURS)] for i, g in enumerate(names)}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    allv = np.concatenate([v[np.isfinite(v)] for v in per_session.values()]) if per_session else np.array([1.0])
    hi = float(np.percentile(allv, 99.5)) if allv.size else 1.0
    bins = np.linspace(0, max(hi, 1e-6), 60)
    for sid, v in per_session.items():
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        h, _ = np.histogram(np.clip(v, 0, hi), bins=bins, density=True)
        ax.plot((bins[:-1] + bins[1:]) / 2, h, color=colour.get(groups.get(sid), GROUP_COLOURS[0]), lw=1.2,
                alpha=0.85)
    if names != [None]:
        for g in names:
            ax.plot([], [], color=colour[g], lw=2, label=g)
        ax.legend(frameon=False, title="condition")
    ax.set_xlabel(f"centroid speed ({units}/s)")
    ax.set_ylabel("density")
    ax.set_title(f"Speed distribution, {len(per_session)} sessions")
    ax.grid(axis="y", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def group_comparison(rows, stats_rows, group_col, out):
    """Strip plots of the per-session metrics, one panel per metric, with the test in the title."""
    stats_rows = [s for s in stats_rows if any(s.get(f"n_{g}") for g in s["groups"])]
    if not stats_rows:
        return None
    groups = stats_rows[0]["groups"]
    colour = {g: GROUP_COLOURS[i % len(GROUP_COLOURS)] for i, g in enumerate(groups)}
    n = len(stats_rows)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n + 0.6, 3.4), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, s in zip(axes[0], stats_rows):
        for i, g in enumerate(groups):
            v = np.array([float(r[s["metric"]]) for r in rows if r.get(group_col) == g and r.get(s["metric"]) is not None])
            v = v[np.isfinite(v)]
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=28, color=colour[g], edgecolor="white", lw=1,
                       zorder=3)
            if len(v):
                ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=INK, lw=1.5, zorder=2)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g}\nn={s.get(f'n_{g}', 0)}" for g in groups])
        ax.set_xlim(-0.6, len(groups) - 0.4)
        p = s.get("p_value")
        g = s.get("hedges_g")
        ptxt = (f"{s['test'].split()[0]} p = {p:.2g}" + (f", g = {g:.2f}" if g is not None else "")) if p is not None else "no test (n < 2)"
        ax.set_title(f"{s['label']}\n{ptxt}", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.5)
    fig.suptitle(f"Per-session metrics by {group_col} (each dot one session, bar = mean)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def confidence_qc(sid, keypoints, below, missing_after, out):
    fig, ax = plt.subplots(figsize=(6.4, 0.28 * len(keypoints) + 1.2))
    y = np.arange(len(keypoints))
    ax.barh(y - 0.2, [100 * below[k] for k in keypoints], height=0.38, color="#eb6834", label="below confidence threshold")
    ax.barh(y + 0.2, [100 * missing_after[k] for k in keypoints], height=0.38, color="#2a78d6",
            label="still missing after interpolation")
    ax.set_yticks(y)
    ax.set_yticklabels(keypoints)
    ax.invert_yaxis()
    ax.set_xlabel("% of frames")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title(f"{sid}: pose quality per keypoint")
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out
