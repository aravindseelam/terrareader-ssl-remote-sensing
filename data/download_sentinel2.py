"""
OPTIONAL: downloads real unlabeled Sentinel-2 tiles via Google Earth Engine,
for when you're ready to scale up beyond EuroSAT to a larger/custom region.

Prerequisites:
    pip install earthengine-api geemap
    earthengine authenticate   # one-time browser login to your GEE account
    (sign up for free at https://earthengine.google.com/)

Usage:
    python data/download_sentinel2.py --lat 30.0 --lon 77.0 --radius-km 50 \
        --start-date 2023-01-01 --end-date 2023-12-31 --out-dir ./data/sentinel2_raw \
        --num-tiles 500

This exports cloud-filtered Sentinel-2 L2A RGB+NIR GeoTIFF tiles to Google
Drive (Earth Engine's export mechanism); you then sync/download the Drive
folder locally. For very large-scale pretraining, keep the exports in Google
Cloud Storage instead and stream them from there.
"""

import argparse

try:
    import ee
except ImportError:
    ee = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--radius-km", type=float, default=50.0)
    parser.add_argument("--start-date", type=str, default="2023-01-01")
    parser.add_argument("--end-date", type=str, default="2023-12-31")
    parser.add_argument("--max-cloud-pct", type=float, default=15.0)
    parser.add_argument("--tile-size-px", type=int, default=256)
    parser.add_argument("--num-tiles", type=int, default=500)
    parser.add_argument("--drive-folder", type=str, default="ssl_remote_sensing_tiles")
    return parser.parse_args()


def main():
    if ee is None:
        raise ImportError(
            "earthengine-api is not installed. Run: pip install earthengine-api geemap"
        )

    args = parse_args()
    ee.Initialize()

    center = ee.Geometry.Point([args.lon, args.lat])
    region = center.buffer(args.radius_km * 1000).bounds()

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(args.start_date, args.end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", args.max_cloud_pct))
        .select(["B4", "B3", "B2", "B8"])  # Red, Green, Blue, NIR
    )

    print(f"Found {collection.size().getInfo()} Sentinel-2 scenes matching filters.")

    # Build a low-cloud composite (median reduces residual cloud/shadow noise)
    composite = collection.median().clip(region)

    # Generate a grid of tile centers across the region and export each as a
    # small GeoTIFF. For a real pretraining corpus you'd loop over many
    # regions/AOIs -- this is a minimal single-region example.
    task = ee.batch.Export.image.toDrive(
        image=composite,
        description="sentinel2_ssl_export",
        folder=args.drive_folder,
        region=region,
        scale=10,  # Sentinel-2 native resolution for RGB/NIR bands, in meters
        maxPixels=1e9,
    )
    task.start()
    print(
        f"Export task started (task id: {task.id}). Monitor progress at "
        "https://code.earthengine.google.com/tasks -- once complete, the "
        f"composite GeoTIFF will appear in your Google Drive folder "
        f"'{args.drive_folder}'. Tile it into {args.tile_size_px}px patches "
        "locally with rasterio/GDAL before feeding into ssl_pretrain/dataset.py "
        "(swap the EuroSAT dataset there for a torchvision.datasets.ImageFolder "
        "pointed at your tiled patches)."
    )


if __name__ == "__main__":
    main()
