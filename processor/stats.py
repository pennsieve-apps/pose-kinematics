"""Between-group comparison of per-session metrics (sessions.csv `condition`, or `group`)."""
from __future__ import annotations

import numpy as np


def hedges_g(a, b):
    """Standardised mean difference (a - b) with the small-sample correction (Hedges & Olkin)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return None
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)))


def compare_groups(rows: list[dict], metrics: list[tuple[str, str]], group_col: str) -> list[dict]:
    """One row per metric: n, mean and SD per group, and a test.

    Two groups: Welch's t-test (unequal variances), with the Mann-Whitney U test (two-sided, exact for
    small n) alongside; with 3 sessions per group the smallest possible Mann-Whitney p is 0.1, which is
    why Welch's t is the headline test. Three or more groups: one-way ANOVA and Kruskal-Wallis.
    Unit of observation: the session. No correction across metrics (exploratory).
    """
    from scipy import stats

    groups = sorted({r.get(group_col) for r in rows if r.get(group_col) not in (None, "")})
    out = []
    if len(groups) < 2:
        return out
    for key, label in metrics:
        vals = {g: np.array([float(r[key]) for r in rows if r.get(group_col) == g and r.get(key) is not None
                             and np.isfinite(float(r[key]))]) for g in groups}
        row = {"metric": key, "label": label, "group_column": group_col, "groups": groups}
        for g in groups:
            v = vals[g]
            row[f"n_{g}"] = int(len(v))
            row[f"mean_{g}"] = float(v.mean()) if len(v) else None
            row[f"sd_{g}"] = float(v.std(ddof=1)) if len(v) > 1 else None
        usable = [vals[g] for g in groups if len(vals[g]) >= 2]
        row["test"], row["statistic"], row["p_value"], row["test_secondary"], row["p_value_secondary"] = \
            None, None, None, None, None
        if len(groups) == 2 and len(usable) == 2:
            a, b = vals[groups[0]], vals[groups[1]]
            t = stats.ttest_ind(a, b, equal_var=False)
            u = stats.mannwhitneyu(a, b, alternative="two-sided")
            row.update(test="Welch t-test", statistic=float(t.statistic), p_value=float(t.pvalue),
                       test_secondary="Mann-Whitney U", p_value_secondary=float(u.pvalue),
                       difference=float(a.mean() - b.mean()), hedges_g=hedges_g(a, b))
        elif len(usable) >= 3 and len(usable) == len(groups):
            f = stats.f_oneway(*usable)
            k = stats.kruskal(*usable)
            row.update(test="one-way ANOVA", statistic=float(f.statistic), p_value=float(f.pvalue),
                       test_secondary="Kruskal-Wallis", p_value_secondary=float(k.pvalue))
        out.append(row)
    return out
