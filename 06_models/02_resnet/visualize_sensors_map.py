"""
Plot sensor values on an interactive map, with a toggle between raw and
distance-weighted-smoothed layers, so smoothing artifacts are visible by eye.

Uses the SAME inverse-distance-weighting smoothing as smooth_labels() in
03_train_resnet.py, reimplemented here standalone so this script has no
dependency on the training repo's internals.

Usage:
  pip install folium branca pandas numpy --break-system-packages
  python3 visualize_sensors_map.py path/to/fold/annual/2024.csv \
      --value PM10_corrected --radius-km 10 --out map.html

Then open map.html in a browser. Use the layer control (top right) to
toggle Raw / Smoothed. Hover a circle for its exact value.
"""
import argparse
import numpy as np
import pandas as pd
import folium
import branca.colormap as cm

LAT0 = 51.0
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.deg2rad(LAT0))


def to_km_xy(lat, lon):
    return lon * KM_PER_DEG_LON, lat * KM_PER_DEG_LAT


def smooth_values(lat, lon, val, radius_km, d0_km=1.0):
    """Inverse-distance-weighted smoothing, matching smooth_labels() in
    03_train_resnet.py: each point's smoothed value is a weighted average of
    itself and every neighbor within radius_km, weight = 1/(distance + d0_km)."""
    x, y = to_km_xy(lat, lon)
    xy = np.stack([x, y], axis=1)
    n = len(xy)
    w0 = 1.0 / d0_km
    wsum = np.full(n, w0)
    ysum = val * w0
    for start in range(0, n, 512):
        stop = min(start + 512, n)
        d = np.sqrt(((xy[start:stop, None, :] - xy[None, :, :]) ** 2).sum(-1))
        rows, cols = np.where(d <= radius_km)
        keep = (rows + start) < cols
        i, j, dd = rows[keep] + start, cols[keep], d[rows[keep], cols[keep]]
        w = 1.0 / (dd + d0_km)
        np.add.at(wsum, i, w); np.add.at(ysum, i, w * val[j])
        np.add.at(wsum, j, w); np.add.at(ysum, j, w * val[i])
    return ysum / wsum


def add_layer(fmap, lat, lon, val, colormap, name, show, radius_px=5):
    fg = folium.FeatureGroup(name=name, show=show)
    for la, lo, v in zip(lat, lon, val):
        folium.CircleMarker(
            location=(la, lo),
            radius=radius_px,
            color=None,
            fill=True,
            fill_color=colormap(v),
            fill_opacity=0.85,
            popup=f"{v:.2f}",
        ).add_to(fg)
    fg.add_to(fmap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV with lat, lon, and a value column (e.g. a fold's annual/2024.csv)")
    ap.add_argument("--value", default="PM10_corrected")
    ap.add_argument("--lat-col", default="lat")
    ap.add_argument("--lon-col", default="lon")
    ap.add_argument("--radius-km", type=float, default=10.0, help="smoothing radius to preview")
    ap.add_argument("--log", action="store_true",
                    help="value column is log-transformed; exponentiate before plotting "
                         "(needed for cv_predictions.csv from training, NOT for the "
                         "calibration script's annual/2024.csv, which is already ug/m3)")
    ap.add_argument("--out", default="sensor_map.html")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=[args.lat_col, args.lon_col, args.value])
    val = df[args.value].values.astype(float)
    if args.log:
        val = np.exp(val)
    lat, lon = df[args.lat_col].values, df[args.lon_col].values
    smoothed = smooth_values(lat, lon, val, args.radius_km)

    print(f"{len(df)} points  |  raw: mean={val.mean():.2f} std={val.std():.2f}  "
          f"|  smoothed (r={args.radius_km:g}km): mean={smoothed.mean():.2f} std={smoothed.std():.2f}")

    vmin, vmax = np.percentile(np.concatenate([val, smoothed]), [2, 98])
    colormap = cm.LinearColormap(["blue", "green", "yellow", "red"], vmin=vmin, vmax=vmax)
    colormap.caption = args.value

    fmap = folium.Map(location=(lat.mean(), lon.mean()), zoom_start=6, tiles="cartodbpositron")
    add_layer(fmap, lat, lon, val, colormap, "Raw", show=True)
    add_layer(fmap, lat, lon, smoothed, colormap, f"Smoothed r={args.radius_km:g}km", show=False)
    colormap.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    fmap.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
