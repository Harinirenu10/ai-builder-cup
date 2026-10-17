"""A population of bodies for the return simulation.

`Population` holds body measurements as arrays (cm) with a weight each. Two sources are meant to fill it:
  - "smpl":     betas sampled from the SMPL shape space and measured by the geometry service (NOT BUILT YET);
  - "stand-in": a correlated normal over the zone measurements with hand-chosen parameters (config/sim.yaml). INVENTED, not data.
The banner the studio shows comes from `banner()`, so it can never claim SMPL for a stand-in.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fitcore.fit.config import get_config
from fitcore.schemas import Measurement, Measurements, MeasurementZone as Z

BODY_ZONES: list[Z] = [Z.chest, Z.waist, Z.hip, Z.shoulder_width, Z.sleeve_length, Z.torso_length, Z.thigh, Z.inseam, Z.neck, Z.back_rise, Z.upper_back]
ORDER = ["height"] + [z.value for z in BODY_ZONES]
GENDERS = ["female", "male"]

BANNER_SMPL = "Synthetic population from the SMPL shape space. Calibrate with consented, anonymized twins before relying on it."
BANNER_STANDIN = ("Synthetic population from a statistical stand-in with invented body-measurement distributions, not the SMPL shape space. "
                  "Calibrate with consented, anonymized twins before relying on it.")


def banner(source: str) -> str:
    return BANNER_SMPL if source == "smpl" else BANNER_STANDIN


@dataclass
class Population:
    source: str                                   # "smpl" or "stand-in"
    seed: int
    gender: np.ndarray                            # "female" / "male" per body
    height_cm: np.ndarray
    zones: dict[Z, np.ndarray]
    weights: np.ndarray = field(default=None)  # type: ignore[assignment]   # sum to 1
    reweighted_to: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = np.full(len(self.gender), 1.0 / len(self.gender))

    def __len__(self) -> int:
        return len(self.gender)

    def body(self, i: int) -> Measurements:
        """One body as the fit engine's input (confidence 1: these are model bodies, not photos)."""
        return Measurements(zones={z: Measurement(value_cm=float(self.zones[z][i]), confidence=1.0, source="population") for z in BODY_ZONES})

    def banner(self) -> str:
        return banner(self.source)

    # ------------------------------------------------------------------ reweighting

    def reweighted(self, height_cm: dict[str, list[float]] | None = None, gender_mix: dict[str, float] | None = None,
                   bin_cm: float | None = None, max_weight: float | None = None) -> "Population":
        """Weights that make the height distribution (per gender) and the gender mix match a target. Works for any source:
        it compares the share of bodies in each height bin with the share the target puts there."""
        t = get_config().target
        height_cm = height_cm or t["height_cm"]
        gender_mix = gender_mix or t["gender_mix"]
        bin_cm = bin_cm or t["bin_cm"]
        max_weight = max_weight or t["max_weight"]
        w = np.zeros(len(self))
        mix_total = sum(gender_mix.values())
        for g in GENDERS:
            idx = np.nonzero(self.gender == g)[0]
            if len(idx) == 0:
                continue
            mean, sd = height_cm[g]
            h = self.height_cm[idx]
            edges = np.arange(math.floor(h.min() / bin_cm) * bin_cm, h.max() + bin_cm, bin_cm)
            which = np.clip(np.digitize(h, edges) - 1, 0, len(edges) - 2)
            have = np.bincount(which, minlength=len(edges) - 1) / len(idx)
            centres = (edges[:-1] + edges[1:]) / 2
            want = np.exp(-0.5 * ((centres - mean) / sd) ** 2)
            want = want / want.sum()
            ratio = np.where(have > 0, want / np.where(have > 0, have, 1), 0.0)
            ww = np.minimum(ratio[which], max_weight)
            ww = ww / ww.sum() if ww.sum() > 0 else np.full(len(idx), 1 / len(idx))
            w[idx] = ww * gender_mix.get(g, 0.0) / mix_total
        w = w / w.sum()
        return Population(self.source, self.seed, self.gender, self.height_cm, self.zones, w, {"height_cm": height_cm, "gender_mix": gender_mix})

    def weighted_mean(self, values: np.ndarray) -> float:
        return float(np.sum(values * self.weights))

    def weighted_sd(self, values: np.ndarray) -> float:
        m = self.weighted_mean(values)
        return float(math.sqrt(np.sum(self.weights * (values - m) ** 2)))

    # ------------------------------------------------------------------ parquet (local path, or gs://bucket/path)

    def to_table(self) -> Any:
        import pyarrow as pa

        cols: dict[str, Any] = {"gender": self.gender.astype(str), "height": self.height_cm}
        cols.update({z.value: self.zones[z] for z in BODY_ZONES})
        meta = {b"source": self.source.encode(), b"seed": str(self.seed).encode()}
        return pa.table(cols).replace_schema_metadata(meta)

    def save(self, path: str | Path) -> None:
        import pyarrow.parquet as pq

        table = self.to_table()
        if str(path).startswith("gs://"):
            import io

            from google.cloud import storage  # lazy: only in the cloud

            buf = io.BytesIO()
            pq.write_table(table, buf)
            bucket, _, name = str(path)[5:].partition("/")
            storage.Client().bucket(bucket).blob(name).upload_from_string(buf.getvalue(), content_type="application/octet-stream")
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "Population":
        import pyarrow.parquet as pq

        if str(path).startswith("gs://"):
            import io

            from google.cloud import storage

            bucket, _, name = str(path)[5:].partition("/")
            table = pq.read_table(io.BytesIO(storage.Client().bucket(bucket).blob(name).download_as_bytes()))
        else:
            table = pq.read_table(str(path))
        meta = table.schema.metadata or {}
        return cls(source=meta.get(b"source", b"stand-in").decode(), seed=int(meta.get(b"seed", b"0")), gender=np.array(table["gender"].to_pylist()),
                   height_cm=table["height"].to_numpy(), zones={z: table[z.value].to_numpy() for z in BODY_ZONES})


# ----------------------------------------------------------------------------- the stand-in generator


def _correlation(cfg: dict[str, Any]) -> np.ndarray:
    c = cfg["correlation"]
    groups = {"girth": set(c["girth"]), "length": set(c["length"]), "width": set(c["width"])}
    names = ORDER

    def grp(n: str) -> str:
        return next(g for g, members in groups.items() if n in members)

    n = len(names)
    r = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = names[i], names[j]
            ga, gb = grp(a), grp(b)
            if ga == gb:
                v = c[f"within_{ga}"]
                if "height" in (a, b) and ga == "length":
                    v = c["height_with_length"]
            else:
                pair = frozenset((ga, gb))
                v = c["girth_length"] if pair == {"girth", "length"} else c["girth_width"] if pair == {"girth", "width"} else c["length_width"]
                if "height" in (a, b) and "width" in (ga, gb):
                    v = c["height_width"]
            r[i, j] = r[j, i] = v
    vals, vecs = np.linalg.eigh(r)                     # repair: a correlation matrix must have no negative eigenvalues
    if vals.min() < 1e-3:
        r = vecs @ np.diag(np.maximum(vals, 1e-3)) @ vecs.T
        d = np.sqrt(np.diag(r))
        r = r / np.outer(d, d)
    return r


def generate_standin(seed: int | None = None, per_gender: int | None = None) -> Population:
    """The stand-in population: per gender, a correlated normal over the measurements with the parameters in config/sim.yaml. Fixed seed."""
    cfg = get_config().population
    seed = cfg["seed"] if seed is None else seed
    per_gender = cfg["per_gender"] if per_gender is None else per_gender
    chol = np.linalg.cholesky(_correlation(cfg))
    genders, cols = [], {name: [] for name in ORDER}
    for k, g in enumerate(GENDERS):
        rng = np.random.default_rng([seed, k])
        z = np.clip(rng.standard_normal((per_gender, len(ORDER))) @ chol.T, -cfg["clip_sd"], cfg["clip_sd"])
        for j, name in enumerate(ORDER):
            mean, sd = cfg["standin"][g][name]
            cols[name].append(mean + sd * z[:, j])
        genders += [g] * per_gender
    return Population(source="stand-in", seed=seed, gender=np.array(genders), height_cm=np.concatenate(cols["height"]),
                      zones={z: np.concatenate(cols[z.value]) for z in BODY_ZONES})


def load_or_create(path: str | Path | None = None) -> Population:
    """The cached population, from parquet if it is there, otherwise generated and cached (when a path is given)."""
    if path:
        try:
            return Population.load(path)
        except Exception:
            pass
    pop = generate_standin()
    if path:
        try:
            pop.save(path)
        except Exception:
            pass                                       # an unwritable cache must not stop the simulation
    return pop
