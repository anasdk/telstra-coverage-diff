#!/usr/bin/env python3
"""
coverage_diff.py — Telstra Coverage Map Comparator
====================================================
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

Notes
-----
* Default control points are calibrated for Telstra's map tool at "Country"
  zoom level (full Australia view). If your screenshots use a different zoom
  or crop, provide a custom --control-points JSON file (see README).
* The before image is assumed to contain UI chrome (legend box, zoom controls
  etc.). Default exclusion zones are set for a 2260×1362 px screenshot; adjust
  --ui-exclusions if your before image differs.
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

# Natural Earth admin-1 GeoJSON URL (states / provinces, 10m resolution)
NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_admin_1_states_provinces.geojson"
)

# Web Mercator earth radius
WGS84_R = 6_378_137.0

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
    purple = (r > g + t["purple_lead"]) & (b > g + t["purple_lead"]) & (r > t["purple_min"])

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
    Fit a 2×3 least-squares affine matrix mapping pts_src → pts_dst.
    pts_src, pts_dst: N×2 float arrays of (x, y) pixel coordinates.
    Returns a 2×3 matrix M such that M @ [x, y, 1]^T ≈ [x', y']^T.
    """
    N = pts_src.shape[0]
    X = np.hstack([pts_src, np.ones((N, 1))])          # N×3
    sol_x, *_ = np.linalg.lstsq(X, pts_dst[:, 0], rcond=None)
    sol_y, *_ = np.linalg.lstsq(X, pts_dst[:, 1], rcond=None)
    return np.vstack([sol_x, sol_y])                    # 2×3


def warp_label(label: np.ndarray, M: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Warp a uint8 label map using affine matrix M into target_shape (H, W).
    Out-of-bounds pixels are set to 255 (invalid).
    """
    H, W = target_shape
    return cv2.warpAffine(
        label, M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


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


def rasterize_state(geom, geo_t: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Rasterize a Shapely geometry (lon/lat) into an H×W binary uint8 mask
    using the geo_t pixel transform.
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
        return np.zeros((H, W), dtype=np.uint8)

    mask = np.zeros((H, W), dtype=np.uint8)
    for pg in px_geoms:
        if not pg.is_valid:
            continue
        pts = np.array(pg.exterior.coords, dtype=np.float32)[:, :2].reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
        for interior in pg.interiors:
            ipts = np.array(interior.coords, dtype=np.float32)[:, :2].reshape(-1, 1, 2)
            cv2.fillPoly(mask, [ipts.astype(np.int32)], 0)
    return mask


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
        r = requests.get(NATURAL_EARTH_URL, timeout=60)
        r.raise_for_status()
        with open(cache, "wb") as f:
            f.write(r.content)

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
    """
    masks = {}
    for _, row in aus_gdf.iterrows():
        name = row["name"]
        mask = rasterize_state(row.geometry, geo_t, image_shape)
        if mask.sum() > 50:          # skip near-empty (offshore islands etc.)
            masks[name] = mask
    return masks


# ---------------------------------------------------------------------------
# 5. Statistics
# ---------------------------------------------------------------------------

def compute_national_diff(warped1: np.ndarray,
                          label2: np.ndarray) -> dict:
    """
    Compute national-level coverage diff between the warped before label map
    and the after label map.
    Returns a dict with pixel counts and fractions.
    """
    valid       = warped1 != 255
    land1       = ((warped1 == 1) | (warped1 == 2)) & valid
    land2       = ((label2  == 1) | (label2  == 2)) & valid
    land_both   = land1 & land2

    cov1  = (warped1 == 2) & land_both
    cov2  = (label2  == 2) & land_both
    lost  = cov1 & ~cov2
    gained = ~cov1 & cov2
    both  = cov1 & cov2

    total = land_both.sum()
    return {
        "total_land_px": int(total),
        "covered_before_px": int(cov1.sum()),
        "covered_after_px":  int(cov2.sum()),
        "lost_px":   int(lost.sum()),
        "gained_px": int(gained.sum()),
        "frac_before": float(cov1.sum() / total) if total else 0.0,
        "frac_after":  float(cov2.sum() / total) if total else 0.0,
        # binary maps for visualisation
        "_cov1": cov1, "_cov2": cov2, "_lost": lost,
        "_gained": gained, "_both": both, "_land_both": land_both,
    }


def compute_state_stats(warped1: np.ndarray,
                        label2: np.ndarray,
                        state_masks: dict) -> list:
    """
    Compute per-state coverage stats.
    Returns a list of dicts, one per state, sorted by coverage drop %.
    """
    rows = []
    for name, smask in state_masks.items():
        area_km2 = ABS_STATE_AREAS.get(name)
        if area_km2 is None:
            continue

        m     = smask > 0
        valid = (warped1 != 255) & m
        l1    = ((warped1 == 1) | (warped1 == 2)) & m & valid
        l2    = ((label2  == 1) | (label2  == 2)) & m & valid
        both  = l1 & l2

        if both.sum() < 100:
            continue

        c1    = (warped1 == 2) & both
        c2    = (label2  == 2) & both
        frac1 = float(c1.sum() / both.sum())
        frac2 = float(c2.sum() / both.sum())

        km2_before = round(frac1 * area_km2, -3)
        km2_after  = round(frac2 * area_km2, -3)
        km2_lost   = km2_before - km2_after
        drop_pct   = (km2_lost / km2_before * 100) if km2_before else 0.0

        rows.append({
            "state":        name,
            "abbrev":       STATE_ABBREV.get(name, name[:3]),
            "area_km2":     area_km2,
            "before_km2":   int(km2_before),
            "after_km2":    int(km2_after),
            "lost_km2":     int(km2_lost),
            "before_pct":   round(frac1 * 100, 1),
            "after_pct":    round(frac2 * 100, 1),
            "drop_pct":     round(drop_pct, 1),
        })

    rows.sort(key=lambda r: -r["drop_pct"])
    return rows


def print_stats_table(national: dict, state_rows: list):
    """Print a formatted summary to stdout."""
    frac_b = national["frac_before"]
    frac_a = national["frac_after"]
    # Use IBSAAR methodology note
    print()
    print("=" * 75)
    print("  Telstra Coverage Diff — ACMA Industry Standard 2026")
    print("=" * 75)
    print(f"  National (pixel-derived estimate):")
    print(f"    Before  : {frac_b*100:5.1f}% of matched landmass covered")
    print(f"    After   : {frac_a*100:5.1f}% of matched landmass covered")
    drop = frac_b - frac_a
    print(f"    Drop    : {drop*100:5.1f} percentage points")
    print()
    print(f"  Telstra's own stated national totals:")
    print(f"    Before  : ~3,000,000 km²  (pre-ACMA standard)")
    print(f"    After   : ~2,140,000 km²  (post-ACMA, 1 Jul 2026)")
    print(f"    Removed :   ~860,000 km²  (-28.7%)")
    print()
    print(f"  Per-state estimates (pixel-fraction × ABS official area):")
    hdr = f"  {'State':<28} {'Area km²':>10} {'Before km²':>11} {'After km²':>10} {'Lost km²':>10} {'Drop':>7}"
    print(hdr)
    print("  " + "-" * 73)
    for r in state_rows:
        print(
            f"  {r['state']:<28} {r['area_km2']:>10,} "
            f"{r['before_km2']:>11,} {r['after_km2']:>10,} "
            f"{r['lost_km2']:>10,} {r['drop_pct']:>6.1f}%"
        )
    print("  " + "-" * 73)
    total_b = sum(r["before_km2"] for r in state_rows)
    total_a = sum(r["after_km2"]  for r in state_rows)
    total_l = total_b - total_a
    drop_t  = total_l / total_b * 100 if total_b else 0
    print(
        f"  {'TOTAL':<28} {'':>10} "
        f"{total_b:>11,} {total_a:>10,} "
        f"{total_l:>10,} {drop_t:>6.1f}%"
    )
    print()
    print("  * State km² are derived estimates (±15–25% uncertainty).")
    print("    State-level % drops are more reliable than absolute km².")
    print("=" * 75)
    print()


def save_csv(state_rows: list, path: str):
    """Save per-state stats to a CSV file."""
    fields = ["state", "abbrev", "area_km2", "before_km2", "after_km2",
              "lost_km2", "before_pct", "after_pct", "drop_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in state_rows:
            w.writerow({k: r[k] for k in fields})
    print(f"Saved statistics → {path}")


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------

def _clean_colorize(arr: np.ndarray) -> np.ndarray:
    """Return an H×W×3 uint8 image with clean coverage colours."""
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    t = COLOR_THRESHOLDS
    or_, og, ob = t["ocean_rgb"]
    tol = t["ocean_tol"]
    ocean  = (np.abs(r - or_) < tol) & (np.abs(g - og) < tol) & (np.abs(b - ob) < tol)
    green  = (g > r + t["green_lead"])  & (g > b + t["green_lead"])  & (g > t["green_min"])
    purple = (r > g + t["purple_lead"]) & (b > g + t["purple_lead"]) & (r > t["purple_min"])

    out = np.full(arr.shape, 255, dtype=np.uint8)   # default white (no coverage land)
    out[ocean]  = (208, 208, 208)
    out[green]  = (48, 128, 32)
    out[purple] = (155, 40, 170)
    return out


def plot_side_by_side(before_arr: np.ndarray,
                      after_arr:  np.ndarray,
                      reg_M:      np.ndarray,
                      ui_excl:    list,
                      output_path: str):
    """
    Produce a re-aligned side-by-side comparison image.
    before_arr / after_arr: original int32 RGB arrays.
    reg_M: 2×3 affine matrix (before → after pixel space).
    """
    H2, W2 = after_arr.shape[:2]

    # Clean-colour before image, zero-out UI chrome, warp into after frame
    c1 = _clean_colorize(before_arr)
    for y0, y1, x0, x1 in ui_excl:
        c1[y0:y1, x0:x1] = (208, 208, 208)
    warped_c1 = cv2.warpAffine(
        c1, reg_M, (W2, H2),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 235, 210),
    )

    c2 = _clean_colorize(after_arr)

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
                txt    = ax.text(
                    cx + ox, cy + oy,
                    f"{abbr}\n{km2:,}k km²\n({pct:.0f}%)",
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
        "  Coverage km² = fraction of covered pixels per state × official ABS land area.  "
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
        config:      dict = None):
    """
    Full pipeline: load → classify → register → geo-project → stats → plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    cfg          = config or {}
    reg_pts      = cfg.get("registration_points",  DEFAULT_REGISTRATION_POINTS)
    geo_pts      = cfg.get("geo_control_points",   DEFAULT_GEO_CONTROL_POINTS)
    ui_excl      = cfg.get("ui_exclusions",        DEFAULT_UI_EXCLUSIONS)
    color_t      = cfg.get("color_thresholds",     COLOR_THRESHOLDS)

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

    # ---- Register before → after ----
    print("Registering images …")
    pts_src = np.array([[p[0], p[1]] for p in reg_pts], dtype=float)
    pts_dst = np.array([[p[2], p[3]] for p in reg_pts], dtype=float)
    M       = fit_affine(pts_src, pts_dst)
    warped1 = warp_label(label1, M, (H2, W2))

    # ---- Geographic transform ----
    print("Fitting geographic transform …")
    geo_t = fit_geo_transform(geo_pts)

    # ---- State boundaries ----
    print("Loading state boundaries …")
    aus_gdf     = load_state_boundaries(state_data)
    state_masks = build_state_masks(aus_gdf, geo_t, (H2, W2))
    print(f"  Rasterized {len(state_masks)} state masks.")

    # ---- Statistics ----
    print("Computing statistics …")
    national    = compute_national_diff(warped1, label2)
    state_rows  = compute_state_stats(warped1, label2, state_masks)
    print_stats_table(national, state_rows)
    save_csv(state_rows, os.path.join(output_dir, "state_stats.csv"))

    # ---- Plots ----
    print("Generating plots …")
    plot_side_by_side(
        a1, a2, M, ui_excl,
        os.path.join(output_dir, "01_side_by_side.png"),
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
    )


if __name__ == "__main__":
    main()
