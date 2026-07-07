#!/usr/bin/env python3
"""
coverage_diff.py — Telstra Coverage Map Comparator  (v2, reviewed & corrected)
==============================================================================
Compares two Telstra mobile coverage map screenshots (before/after the ACMA
Telecommunications (Mobile Network Coverage Maps) Industry Standard 2026) and
produces:

  1. Re-aligned side-by-side comparison (PNG)
  2. Pixel-level diff map  — red = coverage lost, blue = coverage gained (PNG)
  3. Three-panel state choropleth — before (green) / after (blue) / removed (red) (PNG)
  4. Per-state coverage statistics table (CSV + console)

Usage
-----
    python coverage_diff.py before.png after.png [options]

    python coverage_diff.py before.png after.png \\
        --output-dir results/ \\
        --control-points control_points.json \\
        --state-data ne_states.geojson

Changes in v2
-------------
* FIX: choropleth state labels were off by 1000x ("500,000k km²") — now shown
  correctly in thousands of km².
* FIX: UI-chrome exclusion is now applied to BOTH images. New optional config
  key "ui_exclusions_after" (defaults to []).
* FIX: per-state km² are kept at full precision internally and rounded only
  for display/CSV, so small territories (ACT, Jervis Bay) no longer quantize
  to 0/1000/2000 km² and drop % is computed from unrounded values.
* FIX: purple classifier now requires a minimum on the blue channel too.
* NEW: Mercator area correction — pixel counts are weighted by cos²(latitude)
  so northern (tropical) pixels are no longer under-weighted relative to
  southern ones. Australia spans ~9°S–44°S, a ~1.9x per-pixel area ratio.
* NEW: registration quality report — per-point and RMS residuals are printed;
  a warning is emitted above 25 px RMS and the run aborts above 80 px unless
  --force is given.
* NEW: robust registration — RANSAC (cv2.estimateAffine2D) with least-squares
  fallback, so one badly digitized control point cannot skew the fit.
* NEW: area-weighted majority warping — class masks are blurred to the target
  scale and warped bilinearly, then combined by majority vote, instead of
  nearest-neighbour point sampling (reduces boundary aliasing when the
  before image is higher resolution than the after image).
* NEW: boundary-noise sensitivity band — coverage masks are eroded/dilated by
  2 px and the "lost" area recomputed, giving an honest error range for the
  headline number.
* NEW: cropped-state detection — warns when a state's rasterized mask is
  substantially smaller than its expected on-screen footprint (i.e. the
  screenshot cuts the state off), since its km² would then be biased.
* Cleanup: visualization reuses classify_pixels() with the SAME thresholds as
  the statistics (custom --control-points thresholds now affect plots too);
  Natural Earth download is streamed to disk instead of loaded into memory.

Notes
-----
* Default control points are calibrated for Telstra's map tool at "Country"
  zoom level (full Australia view). If your screenshots use a different zoom
  or crop, provide a custom --control-points JSON file (see README).
* State km² estimates carry ±15–25% uncertainty — they are derived from pixel
  fractions multiplied by official ABS land areas. Telstra's stated national
  totals (~3.0M km² before, ~2.14M km² after) are more authoritative for the
  overall figure.

References
----------
* Telstra pre-standard claim:
    https://www.telstra.com.au/exchange/telstra-coverage-maps--the-facts-about-how-it-s-measured
* Telstra post-standard disclosure (1 Jul 2026):
    https://www.telstra.com.au/exchange/australia-s-mobile-coverage-maps-are-changing--what-the-new-nati
* ACMA standard:
    https://www.acma.gov.au/telecommunications-mobile-network-coverage-maps-industry-standard-2026
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import MultiPolygon, Polygon

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Official ABS land areas (km²) — used to scale pixel fractions to km²
ABS_STATE_AREAS = {
    "Western Australia":            2_645_615,
    "Northern Territory":           1_419_630,
    "South Australia":              1_043_514,
    "Queensland":                   1_853_188,
    "New South Wales":                809_444,
    "Victoria":                       237_659,
    "Tasmania":                        90_758,
    "Australian Capital Territory":     2_358,
    "Jervis Bay Territory":                70,
}

STATE_ABBREV = {
    "Western Australia": "WA", "Northern Territory": "NT",
    "South Australia": "SA", "Queensland": "QLD",
    "New South Wales": "NSW", "Victoria": "VIC",
    "Tasmania": "TAS", "Australian Capital Territory": "ACT",
}

# Label offsets (lon, lat degrees) for choropleth annotation — tweak if needed
STATE_LABEL_OFFSETS = {
    "Western Australia": (-2.2, -1.8), "Northern Territory": (1.2, 1.8),
    "South Australia": (0.2, 0.8),     "Queensland": (2.0, -1.5),
    "New South Wales": (1.5, 0.8),     "Victoria": (0.8, -0.5),
    "Tasmania": (0.5, -0.6),           "Australian Capital Territory": (1.8, 0.2),
}

# Natural Earth admin-1 GeoJSON URL (states / provinces, 10m resolution).
# NOTE: points at `master` — for strict reproducibility, download once, commit
# the file (or pin a release tag), and pass it via --state-data.
NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_admin_1_states_provinces.geojson"
)

# Web Mercator earth radius
WGS84_R = 6_378_137.0

# Registration quality thresholds (px RMS in the AFTER image frame)
REG_RMS_WARN  = 25.0
REG_RMS_ABORT = 80.0

# ---------------------------------------------------------------------------
# Default configuration (overridable via --control-points JSON)
# ---------------------------------------------------------------------------

# Control points mapping image2 (after) pixel coordinates → WGS84 lon/lat.
# These are calibrated for Telstra's map at "Country" zoom, 1370×988 px.
# Format: [px_x, px_y, longitude, latitude]
DEFAULT_GEO_CONTROL_POINTS = [
    [280,  570,  115.86, -31.95],   # Perth
    [775,  645,  138.60, -34.93],   # Adelaide
    [900,  715,  144.96, -37.81],   # Melbourne
    [1040, 615,  151.21, -33.87],   # Sydney
    [1110, 450,  153.03, -27.47],   # Brisbane
    [893,   60,  147.15,  -9.44],   # Port Moresby
    [1099, 465,  153.54, -28.16],   # Gold Coast
    [979,  268,  146.82, -19.25],   # Townsville
]

# Control points mapping image1 (before) pixel → image2 (after) pixel.
# These allow the before image to be warped into the after image's coordinate frame.
# Format: [x_before, y_before, x_after, y_after]
DEFAULT_REGISTRATION_POINTS = [
    [715,   668,  280,  570],   # Perth
    [1245,  775,  775,  645],   # Adelaide
    [1320,  868,  900,  715],   # Melbourne
    [1545,  745, 1040,  615],   # Sydney
    [1570,  610, 1110,  450],   # Brisbane
    [1355,  190,  893,   60],   # Port Moresby
]

# UI chrome exclusion zones in the BEFORE image — list of [y0, y1, x0, x1]
# Set to [] if your before image has no overlaid UI elements.
DEFAULT_UI_EXCLUSIONS = [
    [0,   200,    0,  510],   # "Explore our full coverage" button
    [0,   120,  480,  930],   # Search / locate icons
    [895, 1362,    0, 1140],  # Legend box (Type of coverage)
    [1035, 1362, 1590, 2260],  # Opacity / Zoom panel
    [1345, 1362,    0, 2260],  # Attribution bar
]

# UI chrome exclusion zones in the AFTER image — same format. The default
# after-screenshot was cropped to the map canvas, hence empty; override via
# the "ui_exclusions_after" key in --control-points JSON if yours has chrome.
DEFAULT_UI_EXCLUSIONS_AFTER = []

# Pixel colour thresholds for coverage classification
COLOR_THRESHOLDS = {
    "ocean_rgb":   (208, 208, 208),   # Grey ocean background
    "ocean_tol":   14,                # Tolerance ± around ocean colour
    "green_lead":  30,                # Green channel must exceed R and B by this much
    "green_min":   60,                # Green channel minimum absolute value
    "purple_lead": 20,                # R and B must exceed G by this much
    "purple_min":  60,                # R and B minimum absolute value
}

# ---------------------------------------------------------------------------
# 1. Image loading & pixel classification
# ---------------------------------------------------------------------------

def load_rgb(path: str) -> np.ndarray:
    """Load an image and return an int32 H×W×3 RGB array."""
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.int32)


def classify_pixels(arr: np.ndarray, t: dict = None) -> np.ndarray:
    """
    Classify each pixel as:
      0 = ocean / background
      1 = land, no coverage
      2 = covered (4G green or 5G purple)

    Returns uint8 label array of same H×W shape.
    """
    if t is None:
        t = COLOR_THRESHOLDS
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    or_, og, ob = t["ocean_rgb"]
    tol = t["ocean_tol"]

    ocean  = (np.abs(r - or_) < tol) & (np.abs(g - og) < tol) & (np.abs(b - ob) < tol)
    green  = (g > r + t["green_lead"])  & (g > b + t["green_lead"])  & (g > t["green_min"])
    # v2 FIX: require the minimum on BOTH red and blue channels for purple
    purple = ((r > g + t["purple_lead"]) & (b > g + t["purple_lead"])
              & (r > t["purple_min"])    & (b > t["purple_min"]))

    label = np.ones(arr.shape[:2], dtype=np.uint8)  # default: land, no coverage
    label[ocean]           = 0
    label[green | purple]  = 2
    return label


def apply_ui_mask(label: np.ndarray, exclusions: list) -> np.ndarray:
    """
    Set UI-chrome pixels to 255 (invalid) so they are ignored in analysis.
    exclusions: list of [y0, y1, x0, x1] rectangles.
    """
    out = label.copy()
    for y0, y1, x0, x1 in exclusions:
        out[y0:y1, x0:x1] = 255
    return out


# ---------------------------------------------------------------------------
# 2. Image registration (before → after coordinate frame)
# ---------------------------------------------------------------------------

def fit_affine(pts_src: np.ndarray, pts_dst: np.ndarray) -> np.ndarray:
    """
    Fit a 2×3 affine matrix mapping pts_src → pts_dst.

    v2: uses RANSAC (cv2.estimateAffine2D) when ≥3 points are available so a
    single badly digitized control point cannot skew the fit; falls back to
    plain least squares if RANSAC fails (e.g. too few inliers).

    pts_src, pts_dst: N×2 float arrays of (x, y) pixel coordinates.
    Returns a 2×3 matrix M such that M @ [x, y, 1]^T ≈ [x', y']^T.
    """
    if pts_src.shape[0] >= 3:
        M, inliers = cv2.estimateAffine2D(
            pts_src.reshape(-1, 1, 2).astype(np.float32),
            pts_dst.reshape(-1, 1, 2).astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=40.0,
            refineIters=50,
        )
        if M is not None:
            n_in = int(inliers.sum()) if inliers is not None else pts_src.shape[0]
            if n_in < pts_src.shape[0]:
                print(f"  Registration: RANSAC kept {n_in}/{pts_src.shape[0]} "
                      f"control points (outliers down-weighted).")
            return M

    # Fallback: ordinary least squares on all points
    N = pts_src.shape[0]
    X = np.hstack([pts_src, np.ones((N, 1))])          # N×3
    sol_x, *_ = np.linalg.lstsq(X, pts_dst[:, 0], rcond=None)
    sol_y, *_ = np.linalg.lstsq(X, pts_dst[:, 1], rcond=None)
    return np.vstack([sol_x, sol_y])                    # 2×3


def registration_residuals(M: np.ndarray,
                           pts_src: np.ndarray,
                           pts_dst: np.ndarray) -> tuple:
    """
    Compute per-point and RMS registration residuals (px, in the AFTER frame).
    Returns (per_point_residuals: N array, rms: float).
    """
    N = pts_src.shape[0]
    X = np.hstack([pts_src, np.ones((N, 1))])          # N×3
    pred = X @ M.T                                      # N×2
    res  = np.linalg.norm(pred - pts_dst, axis=1)
    rms  = float(np.sqrt(np.mean(res ** 2)))
    return res, rms


def report_registration_quality(M: np.ndarray,
                                reg_pts: list,
                                force: bool = False) -> float:
    """
    Print a per-point residual table and enforce quality thresholds.
    Warns above REG_RMS_WARN px RMS; aborts above REG_RMS_ABORT unless force.
    """
    pts_src = np.array([[p[0], p[1]] for p in reg_pts], dtype=float)
    pts_dst = np.array([[p[2], p[3]] for p in reg_pts], dtype=float)
    res, rms = registration_residuals(M, pts_src, pts_dst)

    print("  Registration residuals (px, in AFTER frame):")
    for i, r in enumerate(res):
        print(f"    point {i + 1}: {r:6.1f} px")
    print(f"    RMS      : {rms:6.1f} px")

    if rms > REG_RMS_ABORT and not force:
        sys.exit(
            f"ERROR: registration RMS residual is {rms:.1f} px "
            f"(> {REG_RMS_ABORT:.0f} px). Control points are likely wrong for "
            f"these screenshots — fix --control-points, or rerun with --force "
            f"to proceed anyway."
        )
    if rms > REG_RMS_WARN:
        print(f"  WARNING: RMS residual {rms:.1f} px exceeds {REG_RMS_WARN:.0f} px — "
              f"state-level results remain usable but treat boundary detail "
              f"with caution.")
    return rms


def warp_label(label: np.ndarray, M: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Warp a uint8 label map using affine matrix M into target_shape (H, W).

    v2: area-weighted majority vote instead of nearest-neighbour sampling.
    Each class {0, 1, 2} is warped as a soft (float) mask — pre-blurred to the
    target scale when downsampling — and the class with the largest warped
    weight wins. Pixels whose warped validity weight < 0.5 (out-of-frame or
    dominated by UI-masked source pixels) are set to 255 (invalid).
    """
    H, W = target_shape

    # Estimate the linear scale factor of the transform; if we are shrinking
    # (scale < 1), low-pass filter the masks so bilinear sampling approximates
    # an area average rather than point sampling.
    A = M[:, :2]
    scale = math.sqrt(abs(float(np.linalg.det(A))))
    sigma = 0.0
    if scale < 0.99:
        sigma = 0.5 / max(scale, 1e-6)   # heuristic anti-alias sigma in src px

    def soft_warp(mask_f32: np.ndarray) -> np.ndarray:
        if sigma > 0.3:
            mask_f32 = cv2.GaussianBlur(mask_f32, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return cv2.warpAffine(
            mask_f32, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    valid_w = soft_warp((label != 255).astype(np.float32))
    class_w = [soft_warp((label == c).astype(np.float32)) for c in (0, 1, 2)]

    out = np.argmax(np.stack(class_w, axis=0), axis=0).astype(np.uint8)
    out[valid_w < 0.5] = 255
    return out


# ---------------------------------------------------------------------------
# 3. Geographic projection (pixel ↔ WGS84 via Web Mercator)
# ---------------------------------------------------------------------------

def lonlat_to_merc(lon: float, lat: float) -> tuple:
    """Convert (lon, lat) degrees to Web Mercator (x, y) metres."""
    x = lon * math.pi / 180.0 * WGS84_R
    y = math.log(math.tan(math.pi / 4.0 + lat * math.pi / 180.0 / 2.0)) * WGS84_R
    return x, y


def fit_geo_transform(control_points: list) -> np.ndarray:
    """
    Fit a Web Mercator scale+translate transform from control points.
    control_points: list of [px_x, px_y, lon, lat]
    Returns [ax, bx, ay, by] such that:
        pixel_x = ax * merc_x + bx
        pixel_y = ay * merc_y + by
    """
    px  = np.array([c[0] for c in control_points], dtype=float)
    py  = np.array([c[1] for c in control_points], dtype=float)
    mx  = np.array([lonlat_to_merc(c[2], c[3])[0] for c in control_points])
    my  = np.array([lonlat_to_merc(c[2], c[3])[1] for c in control_points])

    sol_x, *_ = np.linalg.lstsq(np.vstack([mx, np.ones_like(mx)]).T, px, rcond=None)
    sol_y, *_ = np.linalg.lstsq(np.vstack([my, np.ones_like(my)]).T, py, rcond=None)
    return np.array([sol_x[0], sol_x[1], sol_y[0], sol_y[1]])


def lonlat_to_pixel(lon: float, lat: float, geo_t: np.ndarray) -> tuple:
    """Apply geo transform to convert (lon, lat) to pixel (x, y)."""
    ax, bx, ay, by = geo_t
    mx, my = lonlat_to_merc(lon, lat)
    return ax * mx + bx, ay * my + by


def mercator_row_weights(geo_t: np.ndarray, shape: tuple) -> np.ndarray:
    """
    v2: per-row true-area weights correcting Web Mercator area inflation.

    In Web Mercator, the ground area represented by one pixel is proportional
    to cos²(latitude): high-latitude pixels cover less real ground. Weighting
    each pixel row by cos²(lat) makes weighted pixel sums proportional to true
    ground area. Over Australia (~9°S–44°S) the per-pixel area ratio is ~1.9x,
    so this materially changes fractions for tall states (WA/QLD/NT).

    Returns an (H,) float64 array of weights (cos²(lat) per pixel row).
    """
    H, _ = shape
    ax, bx, ay, by = geo_t
    rows = np.arange(H, dtype=np.float64) + 0.5          # pixel-centre rows
    my   = (rows - by) / ay                              # invert pixel_y = ay*my + by
    lat  = 2.0 * np.arctan(np.exp(my / WGS84_R)) - math.pi / 2.0
    return np.cos(lat) ** 2


def wsum(mask: np.ndarray, row_w: np.ndarray) -> float:
    """Area-weighted pixel count: sum of mask pixels weighted per row."""
    return float(mask.sum(axis=1, dtype=np.float64) @ row_w)


def rasterize_state(geom, geo_t: np.ndarray, shape: tuple) -> tuple:
    """
    Rasterize a Shapely geometry (lon/lat) into an H×W binary uint8 mask
    using the geo_t pixel transform.

    v2: also returns the UNCLIPPED expected pixel area of the geometry in the
    image coordinate frame (float), so callers can detect states that fall
    partly outside the screenshot (cv2.fillPoly silently clips).
    Returns (mask, expected_px_area).
    """
    H, W = shape

    def transform_ring(coords):
        return [(lonlat_to_pixel(x, y, geo_t)[0],
                 lonlat_to_pixel(x, y, geo_t)[1]) for x, y in coords]

    def to_px_poly(poly):
        ext  = transform_ring(poly.exterior.coords)
        ints = [transform_ring(r.coords) for r in poly.interiors]
        return Polygon(ext, ints)

    if geom.geom_type == "Polygon":
        px_geoms = [to_px_poly(geom)]
    elif geom.geom_type == "MultiPolygon":
        px_geoms = [to_px_poly(p) for p in geom.geoms]
    else:
        return np.zeros((H, W), dtype=np.uint8), 0.0

    mask = np.zeros((H, W), dtype=np.uint8)
    expected_area = 0.0
    for pg in px_geoms:
        if not pg.is_valid:
            pg = pg.buffer(0)              # attempt repair
            if pg.is_empty or not pg.is_valid:
                continue
        expected_area += pg.area           # unclipped px² area (Shapely)
        polys = pg.geoms if pg.geom_type == "MultiPolygon" else [pg]
        for p in polys:
            pts = np.array(p.exterior.coords, dtype=np.float32)[:, :2].reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
            for interior in p.interiors:
                ipts = np.array(interior.coords, dtype=np.float32)[:, :2].reshape(-1, 1, 2)
                cv2.fillPoly(mask, [ipts.astype(np.int32)], 0)
    return mask, expected_area


# ---------------------------------------------------------------------------
# 4. State boundary data
# ---------------------------------------------------------------------------

def load_state_boundaries(geojson_path: str = None) -> gpd.GeoDataFrame:
    """
    Load Australian state boundaries from a local GeoJSON file or, if not
    provided / not found, download from Natural Earth and cache locally.
    """
    cache = geojson_path or "ne_admin1_states.geojson"
    if not os.path.exists(cache):
        print(f"Downloading state boundaries from Natural Earth → {cache}")
        # v2: stream to disk instead of loading ~40 MB into memory
        with requests.get(NATURAL_EARTH_URL, timeout=60, stream=True) as r:
            r.raise_for_status()
            with open(cache, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

    gdf = gpd.read_file(cache)
    aus = gdf[gdf["admin"] == "Australia"].copy()
    print(f"Loaded {len(aus)} Australian state/territory features.")
    return aus


def build_state_masks(aus_gdf: gpd.GeoDataFrame,
                      geo_t: np.ndarray,
                      image_shape: tuple) -> dict:
    """
    Build a dict of {state_name: binary_mask (H×W uint8)} for all Australian
    states/territories, projected into the image coordinate frame.

    v2: warns when a state's rasterized (clipped) mask is < 90% of its
    expected unclipped footprint — i.e. the screenshot cuts the state off —
    because its km² estimate would then be biased (the coverage fraction of
    the VISIBLE part is extrapolated to the FULL ABS area).
    """
    masks = {}
    for _, row in aus_gdf.iterrows():
        name = row["name"]
        mask, expected_px = rasterize_state(row.geometry, geo_t, image_shape)
        mask_px = float((mask > 0).sum())
        if mask_px <= 50:                # skip near-empty (offshore islands etc.)
            continue
        if expected_px > 0 and mask_px < 0.90 * expected_px:
            visible = 100.0 * mask_px / expected_px
            print(f"  WARNING: '{name}' is only ~{visible:.0f}% visible in the "
                  f"image frame — its km² estimate extrapolates the visible "
                  f"portion to the full state and may be biased.")
        masks[name] = mask
    return masks


# ---------------------------------------------------------------------------
# 5. Statistics
# ---------------------------------------------------------------------------

def compute_national_diff(warped1: np.ndarray,
                          label2:  np.ndarray,
                          row_w:   np.ndarray) -> dict:
    """
    Compute national-level coverage diff between the warped before label map
    and the after label map.

    v2: fractions are Mercator-area-weighted (cos²(lat) per pixel row), and a
    boundary-noise sensitivity band is computed by eroding the before-coverage
    and dilating the after-coverage by 2 px and recomputing the lost fraction.

    Returns a dict with pixel counts, weighted fractions, and binary maps.
    """
    valid       = (warped1 != 255) & (label2 != 255)
    land1       = ((warped1 == 1) | (warped1 == 2)) & valid
    land2       = ((label2  == 1) | (label2  == 2)) & valid
    land_both   = land1 & land2

    cov1   = (warped1 == 2) & land_both
    cov2   = (label2  == 2) & land_both
    lost   = cov1 & ~cov2
    gained = ~cov1 & cov2
    both   = cov1 & cov2

    total_w = wsum(land_both, row_w)
    frac_before = wsum(cov1, row_w) / total_w if total_w else 0.0
    frac_after  = wsum(cov2, row_w) / total_w if total_w else 0.0
    frac_lost   = wsum(lost, row_w) / total_w if total_w else 0.0

    # --- boundary-noise sensitivity: erode "before", dilate "after" by 2 px ---
    k = np.ones((3, 3), np.uint8)
    cov1_e = cv2.erode(cov1.astype(np.uint8),  k, iterations=2).astype(bool)
    cov2_d = cv2.dilate(cov2.astype(np.uint8), k, iterations=2).astype(bool) & land_both
    lost_conservative = cov1_e & ~cov2_d & land_both
    frac_lost_lo = wsum(lost_conservative, row_w) / total_w if total_w else 0.0

    return {
        "total_land_px": int(land_both.sum()),
        "covered_before_px": int(cov1.sum()),
        "covered_after_px":  int(cov2.sum()),
        "lost_px":   int(lost.sum()),
        "gained_px": int(gained.sum()),
        "frac_before": float(frac_before),
        "frac_after":  float(frac_after),
        "frac_lost":   float(frac_lost),
        "frac_lost_conservative": float(frac_lost_lo),
        # binary maps for visualisation
        "_cov1": cov1, "_cov2": cov2, "_lost": lost,
        "_gained": gained, "_both": both, "_land_both": land_both,
    }


def compute_state_stats(warped1: np.ndarray,
                        label2:  np.ndarray,
                        state_masks: dict,
                        row_w:   np.ndarray) -> list:
    """
    Compute per-state coverage stats.

    v2: pixel fractions are Mercator-area-weighted; km² values are kept at
    FULL precision here (floats) and only rounded at display/CSV time, so
    small territories (ACT, Jervis Bay) are no longer quantized to the
    nearest 1000 km² before the drop % is computed.

    Returns a list of dicts, one per state, sorted by coverage drop %.
    """
    rows = []
    for name, smask in state_masks.items():
        area_km2 = ABS_STATE_AREAS.get(name)
        if area_km2 is None:
            continue

        m     = smask > 0
        valid = (warped1 != 255) & (label2 != 255) & m
        l1    = ((warped1 == 1) | (warped1 == 2)) & valid
        l2    = ((label2  == 1) | (label2  == 2)) & valid
        both  = l1 & l2

        if both.sum() < 100:
            continue

        c1      = (warped1 == 2) & both
        c2      = (label2  == 2) & both
        denom_w = wsum(both, row_w)
        frac1   = wsum(c1, row_w) / denom_w if denom_w else 0.0
        frac2   = wsum(c2, row_w) / denom_w if denom_w else 0.0

        km2_before = frac1 * area_km2          # full precision — no rounding
        km2_after  = frac2 * area_km2
        km2_lost   = km2_before - km2_after
        drop_pct   = (km2_lost / km2_before * 100.0) if km2_before else 0.0

        rows.append({
            "state":        name,
            "abbrev":       STATE_ABBREV.get(name, name[:3]),
            "area_km2":     area_km2,
            "before_km2":   km2_before,
            "after_km2":    km2_after,
            "lost_km2":     km2_lost,
            "before_pct":   frac1 * 100.0,
            "after_pct":    frac2 * 100.0,
            "drop_pct":     drop_pct,
        })

    rows.sort(key=lambda r: -r["drop_pct"])
    return rows


def print_stats_table(national: dict, state_rows: list):
    """Print a formatted summary to stdout. (Rounding happens here only.)"""
    frac_b = national["frac_before"]
    frac_a = national["frac_after"]
    print()
    print("=" * 75)
    print("  Telstra Coverage Diff — ACMA Industry Standard 2026")
    print("=" * 75)
    print(f"  National (pixel-derived, Mercator-area-weighted estimate):")
    print(f"    Before  : {frac_b*100:5.1f}% of matched landmass covered")
    print(f"    After   : {frac_a*100:5.1f}% of matched landmass covered")
    drop = frac_b - frac_a
    print(f"    Drop    : {drop*100:5.1f} percentage points")
    print(f"    Removed : {national['frac_lost']*100:5.1f}% of landmass "
          f"(boundary-noise floor: {national['frac_lost_conservative']*100:.1f}%)")
    print()
    print(f"  Telstra's own stated national totals:")
    print(f"    Before  : ~3,000,000 km²  (pre-ACMA standard)")
    print(f"    After   : ~2,140,000 km²  (post-ACMA, 1 Jul 2026)")
    print(f"    Removed :   ~860,000 km²  (-28.7%)")
    print()
    print(f"  Per-state estimates (area-weighted pixel fraction × ABS official area):")
    hdr = f"  {'State':<28} {'Area km²':>10} {'Before km²':>11} {'After km²':>10} {'Lost km²':>10} {'Drop':>7}"
    print(hdr)
    print("  " + "-" * 73)
    for r in state_rows:
        print(
            f"  {r['state']:<28} {r['area_km2']:>10,} "
            f"{round(r['before_km2']):>11,} {round(r['after_km2']):>10,} "
            f"{round(r['lost_km2']):>10,} {r['drop_pct']:>6.1f}%"
        )
    print("  " + "-" * 73)
    total_b = sum(r["before_km2"] for r in state_rows)
    total_a = sum(r["after_km2"]  for r in state_rows)
    total_l = total_b - total_a
    drop_t  = total_l / total_b * 100 if total_b else 0
    print(
        f"  {'TOTAL':<28} {'':>10} "
        f"{round(total_b):>11,} {round(total_a):>10,} "
        f"{round(total_l):>10,} {drop_t:>6.1f}%"
    )
    print()
    print("  * State km² are derived estimates (±15–25% uncertainty).")
    print("    State-level % drops are more reliable than absolute km².")
    print("=" * 75)
    print()


def save_csv(state_rows: list, path: str):
    """Save per-state stats to a CSV file. (Rounding happens here only.)"""
    fields = ["state", "abbrev", "area_km2", "before_km2", "after_km2",
              "lost_km2", "before_pct", "after_pct", "drop_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in state_rows:
            out = dict(r)
            for k in ("before_km2", "after_km2", "lost_km2"):
                out[k] = round(out[k])
            for k in ("before_pct", "after_pct", "drop_pct"):
                out[k] = round(out[k], 1)
            w.writerow({k: out[k] for k in fields})
    print(f"Saved statistics → {path}")


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------

def _clean_colorize(arr: np.ndarray, t: dict = None) -> np.ndarray:
    """
    Return an H×W×3 uint8 image with clean coverage colours.
    v2: reuses classify_pixels() so plots always agree with the statistics,
    including when custom colour thresholds are supplied via config.
    """
    if t is None:
        t = COLOR_THRESHOLDS
    label = classify_pixels(arr, t)

    out = np.full(arr.shape, 255, dtype=np.uint8)   # default white (no-coverage land)
    out[label == 0] = (208, 208, 208)               # ocean
    out[label == 2] = (48, 128, 32)                 # coverage → green by default
    # Re-split coverage into green/purple purely for display fidelity
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    purple = ((r > g + t["purple_lead"]) & (b > g + t["purple_lead"])
              & (r > t["purple_min"])    & (b > t["purple_min"]))
    out[(label == 2) & purple] = (155, 40, 170)
    return out


def plot_side_by_side(before_arr: np.ndarray,
                      after_arr:  np.ndarray,
                      reg_M:      np.ndarray,
                      ui_excl:    list,
                      ui_excl_after: list,
                      output_path: str,
                      color_t:    dict = None):
    """
    Produce a re-aligned side-by-side comparison image.
    before_arr / after_arr: original int32 RGB arrays.
    reg_M: 2×3 affine matrix (before → after pixel space).
    """
    H2, W2 = after_arr.shape[:2]

    # Clean-colour before image, zero-out UI chrome, warp into after frame
    c1 = _clean_colorize(before_arr, color_t)
    for y0, y1, x0, x1 in ui_excl:
        c1[y0:y1, x0:x1] = (208, 208, 208)
    warped_c1 = cv2.warpAffine(
        c1, reg_M, (W2, H2),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 235, 210),
    )

    c2 = _clean_colorize(after_arr, color_t)
    for y0, y1, x0, x1 in ui_excl_after:      # v2: mask after-image chrome too
        c2[y0:y1, x0:x1] = (208, 208, 208)

    # Stitch side-by-side with a label bar
    pad_top = 54
    gap     = 6
    canvas  = Image.new("RGB", (W2 * 2 + gap, H2 + pad_top), (255, 255, 255))
    canvas.paste(Image.fromarray(warped_c1), (0, pad_top))
    canvas.paste(Image.fromarray(c2),        (W2 + gap, pad_top))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    draw.text((10, 10), "BEFORE — April 2026 (re-aligned, pre-ACMA)", fill=(0, 0, 0), font=font)
    draw.text((W2 + gap + 10, 10), "AFTER — 1 July 2026 (ACMA standard)", fill=(0, 0, 0), font=font)
    draw.line([(W2 + gap // 2, 0), (W2 + gap // 2, H2 + pad_top)], fill=(100, 100, 100), width=2)

    canvas.save(output_path)
    print(f"Saved side-by-side comparison → {output_path}")


def plot_diff_map(warped1: np.ndarray,
                  label2:  np.ndarray,
                  diff:    dict,
                  after_arr: np.ndarray,
                  output_path: str):
    """
    Produce a pixel-level diff overlay:
      red  = covered before, no longer covered (regulatory removal)
      blue = newly covered (registration noise / minor change)
      grey = covered in both
      white = no coverage in both
    """
    H, W = label2.shape
    vis  = np.zeros((H, W, 3), dtype=np.uint8)

    ocean2 = (label2 == 0)
    vis[:]                    = (235, 235, 235)      # invalid / out-of-frame
    vis[ocean2]               = (208, 208, 208)      # ocean
    vis[diff["_land_both"]]   = (255, 255, 255)      # no coverage either time
    vis[diff["_both"]]        = (190, 190, 190)      # covered in both → neutral grey
    vis[diff["_lost"]]        = (220,  40,  40)      # RED: removed
    vis[diff["_gained"]]      = ( 40,  90, 220)      # BLUE: gained

    out = Image.fromarray(vis)
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    # Legend
    legend_items = [
        ((220, 40,  40), "Covered in April — removed by ACMA threshold"),
        (( 40, 90, 220), "Newly shown as covered"),
        ((190, 190, 190), "Covered in both"),
        ((255, 255, 255), "No coverage, both periods"),
    ]
    lx, ly = 10, H - 30
    for color, label in legend_items:
        draw.rectangle([lx, ly - 18, lx + 22, ly], fill=color, outline=(80, 80, 80))
        draw.text((lx + 28, ly - 18), label, fill=(0, 0, 0), font=font)
        lx += 28 + int(draw.textlength(label, font=font)) + 30

    out.save(output_path)
    print(f"Saved diff map → {output_path}")


def plot_choropleth(aus_gdf:     gpd.GeoDataFrame,
                    state_rows:  list,
                    output_path: str):
    """
    Three-panel state choropleth:
      Panel 1: Before coverage % — green
      Panel 2: After  coverage % — blue
      Panel 3: Coverage removed  — red
    """
    # Build lookup from state rows
    stats = {r["state"]: r for r in state_rows}

    aus = aus_gdf.copy()
    aus["before_pct"] = aus["name"].map(lambda n: stats.get(n, {}).get("before_pct", 0))
    aus["after_pct"]  = aus["name"].map(lambda n: stats.get(n, {}).get("after_pct",  0))
    aus["drop_pct"]   = aus["name"].map(lambda n: stats.get(n, {}).get("drop_pct",   0))
    aus["before_v"]   = aus["name"].map(lambda n: stats.get(n, {}).get("before_km2", 0))
    aus["after_v"]    = aus["name"].map(lambda n: stats.get(n, {}).get("after_km2",  0))
    aus["lost_v"]     = aus["name"].map(lambda n: stats.get(n, {}).get("lost_km2",   0))

    cmap_g = LinearSegmentedColormap.from_list("g", ["#edf2ec", "#1d7a45"], N=256)
    cmap_b = LinearSegmentedColormap.from_list("b", ["#e8eef7", "#1a4a8a"], N=256)
    cmap_r = LinearSegmentedColormap.from_list("r", ["#f7eaea", "#b71c1c"], N=256)
    ocean  = "#b8cdd8"

    panels = [
        ("before_pct", "BEFORE  —  April 2026",                   cmap_g, 0, 85, "before_v",
         "Estimated coverage (% of state land area)"),
        ("after_pct",  "AFTER  —  1 July 2026  (ACMA standard)",  cmap_b, 0, 85, "after_v",
         "Estimated coverage (% of state land area)"),
        ("drop_pct",   "COVERAGE REMOVED  —  % of April footprint", cmap_r, 0, 60, "lost_v",
         "Coverage removed (% of April footprint)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8.5), facecolor="#f4f3ee")
    fig.patch.set_facecolor("#f4f3ee")

    for ax, (col, title, cmap, vmin, vmax, km2_col, cbar_label) in zip(axes, panels):
        ax.set_facecolor(ocean)
        aus.plot(column=col, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                 edgecolor="#ffffff", linewidth=1.2, zorder=2)
        ax.set_xlim(112.5, 154); ax.set_ylim(-44.5, -9.5)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color="#111")
        ax.axis("off")

        for _, row in aus.iterrows():
            name = row["name"]
            if name == "Australian Capital Territory":
                continue
            s = stats.get(name)
            if not s:
                continue
            try:
                cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
                ox, oy = STATE_LABEL_OFFSETS.get(name, (0, 0))
                km2    = row[km2_col]
                pct    = row[col]
                abbr   = STATE_ABBREV.get(name, name[:3])
                # v2 FIX: value is in km² — convert to thousands for the "k" label
                txt    = ax.text(
                    cx + ox, cy + oy,
                    f"{abbr}\n{round(km2 / 1000):,}k km²\n({pct:.0f}%)",
                    fontsize=7.0, ha="center", va="center",
                    color="#111", fontweight="bold", zorder=6, linespacing=1.35,
                )
                txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])
            except Exception:
                pass

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin, vmax))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, orientation="horizontal",
                          fraction=0.035, pad=0.015, aspect=30, shrink=0.70)
        cb.set_label(cbar_label, fontsize=8, color="#444")
        cb.ax.tick_params(labelsize=7, colors="#444")

    fig.suptitle(
        "Telstra Mobile Coverage — State-Level Impact of ACMA\n"
        "Telecommunications (Mobile Network Coverage Maps) Industry Standard 2026",
        fontsize=12.5, fontweight="bold", y=1.01, color="#111",
    )
    fig.text(
        0.5, -0.02,
        "* Derived estimates from pixel-level analysis of Telstra's published coverage screenshots "
        "(April vs 1 July 2026).\n"
        "  Coverage km² = Mercator-area-weighted fraction of covered pixels per state × official ABS land area.  "
        "State % drops are directionally reliable; absolute km² carry ±15–25% uncertainty.\n"
        "  Telstra's stated national totals: ~3.0M km² (pre-standard) → ~2.14M km² (post-standard).",
        ha="center", fontsize=7, color="#666", linespacing=1.5,
    )

    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.06, wspace=0.06)
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="#f4f3ee")
    plt.close()
    print(f"Saved choropleth → {output_path}")


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load optional JSON config (control points, UI exclusions, thresholds)."""
    with open(config_path) as f:
        return json.load(f)


def run(before_path: str,
        after_path:  str,
        output_dir:  str,
        state_data:  str = None,
        config:      dict = None,
        force:       bool = False):
    """
    Full pipeline: load → classify → register → geo-project → stats → plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    cfg           = config or {}
    reg_pts       = cfg.get("registration_points",  DEFAULT_REGISTRATION_POINTS)
    geo_pts       = cfg.get("geo_control_points",   DEFAULT_GEO_CONTROL_POINTS)
    ui_excl       = cfg.get("ui_exclusions",        DEFAULT_UI_EXCLUSIONS)
    ui_excl_after = cfg.get("ui_exclusions_after",  DEFAULT_UI_EXCLUSIONS_AFTER)
    color_t       = cfg.get("color_thresholds",     COLOR_THRESHOLDS)

    # ---- Load images ----
    print(f"\nLoading images …")
    a1 = load_rgb(before_path)
    a2 = load_rgb(after_path)
    H2, W2 = a2.shape[:2]
    print(f"  Before : {a1.shape[1]}×{a1.shape[0]} px  ({before_path})")
    print(f"  After  : {a2.shape[1]}×{a2.shape[0]} px  ({after_path})")

    # ---- Classify pixels ----
    print("Classifying pixels …")
    label1 = classify_pixels(a1, color_t)
    label2 = classify_pixels(a2, color_t)
    label1 = apply_ui_mask(label1, ui_excl)
    label2 = apply_ui_mask(label2, ui_excl_after)   # v2: mask AFTER image too

    # ---- Register before → after ----
    print("Registering images …")
    pts_src = np.array([[p[0], p[1]] for p in reg_pts], dtype=float)
    pts_dst = np.array([[p[2], p[3]] for p in reg_pts], dtype=float)
    M       = fit_affine(pts_src, pts_dst)
    report_registration_quality(M, reg_pts, force=force)   # v2: residual QA
    warped1 = warp_label(label1, M, (H2, W2))

    # ---- Geographic transform ----
    print("Fitting geographic transform …")
    geo_t = fit_geo_transform(geo_pts)
    row_w = mercator_row_weights(geo_t, (H2, W2))   # v2: Mercator area weights

    # ---- State boundaries ----
    print("Loading state boundaries …")
    aus_gdf     = load_state_boundaries(state_data)
    state_masks = build_state_masks(aus_gdf, geo_t, (H2, W2))
    print(f"  Rasterized {len(state_masks)} state masks.")

    # ---- Statistics ----
    print("Computing statistics …")
    national    = compute_national_diff(warped1, label2, row_w)
    state_rows  = compute_state_stats(warped1, label2, state_masks, row_w)
    print_stats_table(national, state_rows)
    save_csv(state_rows, os.path.join(output_dir, "state_stats.csv"))

    # ---- Plots ----
    print("Generating plots …")
    plot_side_by_side(
        a1, a2, M, ui_excl, ui_excl_after,
        os.path.join(output_dir, "01_side_by_side.png"),
        color_t,
    )
    plot_diff_map(
        warped1, label2, national, a2,
        os.path.join(output_dir, "02_diff_map.png"),
    )
    plot_choropleth(
        aus_gdf, state_rows,
        os.path.join(output_dir, "03_choropleth.png"),
    )

    print(f"\nAll outputs written to: {output_dir}/")


# ---------------------------------------------------------------------------
# 8. CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare Telstra coverage map screenshots before/after ACMA 2026 standard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("before", help="Path to the BEFORE screenshot (pre-ACMA)")
    parser.add_argument("after",  help="Path to the AFTER  screenshot (post-ACMA)")
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory for output files (default: output/)",
    )
    parser.add_argument(
        "--control-points",
        help="JSON file with custom control points / exclusions (see README)",
    )
    parser.add_argument(
        "--state-data",
        help="Path to a local Natural Earth admin-1 GeoJSON (downloaded automatically if omitted)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Proceed even if registration RMS residual exceeds the abort threshold",
    )

    args = parser.parse_args()

    config = None
    if args.control_points:
        config = load_config(args.control_points)
        print(f"Loaded config from {args.control_points}")

    run(
        before_path=args.before,
        after_path=args.after,
        output_dir=args.output_dir,
        state_data=args.state_data,
        config=config,
        force=args.force,
    )


if __name__ == "__main__":
    main()
