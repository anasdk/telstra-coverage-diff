# telstra-coverage-diff
# Telstra Coverage Map Comparator

A Python tool for comparing Telstra mobile coverage map screenshots before and after the **ACMA Telecommunications (Mobile Network Coverage Maps) Industry Standard 2026**, which took effect on 1 July 2026.

---

## What it does

Given two screenshots of Telstra's published coverage map tool (one taken before the standard, one after), the script produces:

| Output file | Description |
|---|---|
| `01_side_by_side.png` | Re-aligned before/after comparison at the same geographic scale |
| `02_diff_map.png` | Pixel-level diff — **red** = coverage removed, **blue** = coverage gained |
| `03_choropleth.png` | Three-panel state choropleth (before / after / % removed) |
| `state_stats.csv` | Per-state coverage estimates in km² and % |

It also prints a formatted summary table to the console.

---

## Approach

1. **Pixel classification** — coverage pixels are identified by colour: green (4G) and purple (5G) on the Telstra map, with white for no-coverage land and grey for ocean.

2. **Image registration** — the two screenshots (different crops/zoom levels of the same tool) are co-registered using a least-squares affine transform fitted to 6 shared control points (major city coverage clusters).

3. **Geographic projection** — state boundaries are projected onto the after-image's pixel frame using a Web Mercator scale+translate transform calibrated from 8 control points (cities with known lon/lat).

4. **Statistics** — for each state, the fraction of covered pixels (before vs after) is multiplied by the official ABS land area to estimate km².

### Accuracy notes

- National totals from pixel analysis run ~15–25% below Telstra's own stated figures (3.0M km² before, 2.14M km² after), because the colour classifier misses faint or anti-aliased pixels at coverage boundaries. **For the national total, cite Telstra's own figures.**
- State-level **percentage drops** are more reliable than absolute km², and are the primary output of interest.
- The registration has ~10–40 pixel residuals at city-level control points, which is adequate for state-level analysis but not for precise boundary-level measurements.

### References

| Figure | Source |
|---|---|
| ~3,000,000 km² (pre-ACMA) | Telstra Exchange: *"Telstra coverage maps: the facts about how it's measured"* — `telstra.com.au/exchange/telstra-coverage-maps--the-facts-about-how-it-s-measured` |
| ~2,140,000 km² (post-ACMA) | Telstra Exchange: *"Australia's mobile coverage maps are changing"*, 1 Jul 2026 — `telstra.com.au/exchange/australia-s-mobile-coverage-maps-are-changing--what-the-new-nati` |
| ACMA standard | `acma.gov.au/telecommunications-mobile-network-coverage-maps-industry-standard-2026` |

---

## Installation

```bash
git clone https://github.com/anasdk/telstra-coverage-diff
cd telstra-coverage-diff

pip install -r requirements.txt
```

> **Python 3.9+** required.

---

## Usage

### Basic

```bash
python coverage_diff.py before.png after.png
```

Outputs are written to `output/` by default.

### Custom output directory

```bash
python coverage_diff.py before.png after.png --output-dir results/
```

### With a local state boundary file (avoids downloading on each run)

On first run the script downloads `ne_10m_admin_1_states_provinces.geojson` (~40 MB) from Natural Earth and saves it locally. Pass the cached path to skip the download:

```bash
python coverage_diff.py before.png after.png --state-data ne_admin1_states.geojson
```

### Custom control points

If your screenshots are at a different zoom level or crop, provide a JSON config file:

```bash
python coverage_diff.py before.png after.png --control-points control_points_example.json
```

See `control_points_example.json` for the full schema. You only need to include the sections you want to override.

---

## Acquiring screenshots

1. Open [Telstra's coverage map](https://www.telstra.com.au/coverage-checker) in a browser.
2. Select **4G + 5G**, zoom to **Country** level (shows all of Australia).
3. Take a full-window screenshot (on macOS: `⌘ Shift 3`; on Windows: `Win + PrintScreen`).
4. Repeat for the before and after dates (or use archived screenshots).

The default control points assume the Telstra map at **Country zoom, full window, approx 1370×988 px (after) and 2260×1362 px (before)**. Adjust `control_points_example.json` if your screenshots differ.

---

## Output examples

### Side-by-side comparison
Re-aligned April (before) and July (after) maps at the same geographic scale, with coverage coloured green (4G) and purple (5G).

### Diff map
Red pixels show coverage that was shown on the April map but is now shown as "no coverage" under the ACMA standard. The pattern is concentrated along regional highways and inland fringe areas, consistent with a signal-threshold reclassification.

### Choropleth
Three-panel map showing per-state coverage as a percentage of land area:
- **Panel 1 (green):** Before — April 2026
- **Panel 2 (blue):** After — 1 July 2026 (ACMA standard)
- **Panel 3 (red):** % of April coverage footprint removed

The Northern Territory shows the steepest relative loss (~56%), followed by Queensland (~43%) and Western Australia (~40%). Victoria and Tasmania are least affected (~13% and ~10%).

---

## File structure

```
telstra-coverage-diff/
├── coverage_diff.py            # Main script (single file, no sub-packages needed)
├── requirements.txt            # Python dependencies
├── control_points_example.json # Example config for custom screenshots
└── README.md
```

---

## Citation

If you use this tool in research or reporting, please cite:

> Dakkak, M. A. (2026). *Pixel-level analysis of Telstra coverage map changes under the ACMA Telecommunications (Mobile Network Coverage Maps) Industry Standard 2026*. IBSAAR. https://github.com/<your-username>/telstra-coverage-diff

---

## Licence

MIT — see below.

```
MIT License

Copyright (c) 2026 Mohamed Anas Dakkak / IBSAAR

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
