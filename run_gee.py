"""
=============================================================================
BUILD DISTRICT GEOMETRIES FOR GEE
=============================================================================
Purpose: Convert Telangana district shapefile/GeoJSON to Earth Engine-ready
         geometry dict saved as JSON for use in rainfall_pipeline.py.

Usage:   python run_once_build_geometries.py

Output:  data/external/district_geometries.json

Windows-compatible version — uses pathlib for cross-platform paths.
=============================================================================
"""

import json
import os
from pathlib import Path

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("⚠️  geopandas not installed. Install with: pip install geopandas")

try:
    import ee
    HAS_EE = True
except ImportError:
    HAS_EE = False
    print("⚠️  earthengine-api not installed. Install with: pip install earthengine-api")

# District name corrections (must match rainfall_pipeline.py)
DISTRICT_NAME_CORRECTIONS = {
    "Jagtial"            : "Jagitial",
    "Jangoan"            : "Jangaon",
    "Kumuram Bheem"      : "Kumuram Bheem Asifabad",
    "Medchal-Malkajgiri" : "Medchal Malkajgiri",
    "Rangareddy"         : "Ranga Reddy",
    "Ranjanna Sircilla"  : "Rajanna Sircilla",
    "Warangal Rural"     : "Warangal (Rural)",
    "Warangal Urban"     : "Warangal (Urban)",
}

def initialize_gee(project_id: str = None):
    """Initialize Google Earth Engine. Prompts authentication if needed."""
    if not HAS_EE:
        print("❌ earthengine-api not installed.")
        return False
    
    # If no project provided, try to use a default
    if not project_id:
        project_id = "ee-default"  # Fallback project name
        print(f"⚠️  No PROJECT_ID specified, using default: {project_id}")
    
    try:
        ee.Initialize(project=project_id)
        print(f"✅ GEE already authenticated (project: {project_id})")
        return True
    except Exception as init_error:
        print("🔐 GEE not authenticated. Starting authentication flow...")
        try:
            ee.Authenticate()
            ee.Initialize(project=project_id)
            print(f"✅ GEE authenticated successfully (project: {project_id})")
            return True
        except Exception as e:
            print(f"❌ GEE authentication failed: {e}")
            print("\n💡 To create a GEE project:")
            print("   1. Visit: https://code.earthengine.google.com/")
            print("   2. Click 'Get Started' or sign in with your Google account")
            print("   3. Register a cloud project (free for non-commercial use)")
            print("   4. After registration, your project ID will look like:")
            print("      - 'ee-yourname' (if using Earth Engine default)")
            print("      - 'your-gcp-project-id' (if using Google Cloud Platform)")
            print("   5. Update PROJECT_ID in this script with your actual project ID")
            print("\n   Example project IDs:")
            print("      PROJECT_ID = 'ee-hasirainfall'")
            print("      PROJECT_ID = 'my-gcp-project-123'")
            return False


def build_geometries(
    shapefile_path: str | Path,
    district_column: str = "DISTRICT_N",
    project_id: str = None
):
    """
    Builds GEE-compatible geometry dict from shapefile/GeoJSON.
    
    Args:
        shapefile_path  : Path to .shp or .geojson file
        district_column : Name of the column containing district names
        project_id      : Google Earth Engine project ID
    
    Returns:
        Dict mapping {district_name: ee.Geometry dict}
    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required. Install with: pip install geopandas")
    
    if not HAS_EE:
        raise ImportError("earthengine-api is required. Install with: pip install earthengine-api")
    
    # Initialize GEE
    if not initialize_gee(project_id):
        raise RuntimeError("GEE initialization failed")
    
    # Load shapefile/GeoJSON
    shapefile_path = Path(shapefile_path)
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    
    print(f"\n📂 Loading: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    
    print(f"   Columns: {gdf.columns.tolist()}")
    print(f"   Districts found: {len(gdf)}")
    
    if district_column not in gdf.columns:
        print(f"\n❌ Column '{district_column}' not found!")
        print(f"   Available columns: {gdf.columns.tolist()}")
        print("\n   Update the district_column parameter and try again.")
        return None
    
    print(f"\n   Sample names from '{district_column}':")
    print(f"   {gdf[district_column].head().tolist()}\n")
    
    # Build geometry dict
    geometries = {}
    skipped    = []
    
    print("🔨 Converting geometries to GEE format...\n")
    
    for idx, row in gdf.iterrows():
        # Clean and correct district name
        raw_name      = str(row[district_column]).strip().title()
        district_name = DISTRICT_NAME_CORRECTIONS.get(raw_name, raw_name)
        
        try:
            # Convert shapely polygon → GeoJSON dict → GEE geometry → dict
            geom_geojson = row["geometry"].__geo_interface__
            ee_geom      = ee.Geometry(geom_geojson)
            geometries[district_name] = ee_geom.getInfo()
            print(f"  ✅ {district_name}")
        except Exception as e:
            print(f"  ❌ {district_name}: {e}")
            skipped.append(district_name)
    
    # Save to JSON
    output_dir  = Path("data/external")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "district_geometries.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geometries, f, indent=2)
    
    print(f"\n✅ Saved {len(geometries)} district geometries → {output_path}")
    
    if skipped:
        print(f"\n⚠️  Skipped districts: {skipped}")
    
    return geometries


if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION — Update these paths for your system
    # ========================================================================
    
    # Google Earth Engine project ID
    # Get this from: https://code.earthengine.google.com/
    # After registering, use your project ID (e.g., "ee-yourname" or "your-gcp-project")
    # 
    # IMPORTANT: You MUST update this with your actual GEE project ID!
    PROJECT_ID = "ee-default"  # ← UPDATE THIS with your GEE project ID
    
    # Option 1: If you have a .shp shapefile
    SHAPEFILE_PATH = Path("data/external/telangana_districts.shp")
    
    # Option 2: If you have a .geojson file (uncomment to use)
    # SHAPEFILE_PATH = Path("data/external/telangana_districts.geojson")
    
    # The column in your shapefile that contains district names
    # Common values: "DISTRICT", "DISTRICT_N", "NAME", "dist_name"
    # Check the "Columns:" output above to find the correct one
    DISTRICT_COLUMN = "DISTRICT_N"
    
    # ========================================================================
    # RUN
    # ========================================================================
    
    print("=" * 65)
    print("  BUILD DISTRICT GEOMETRIES FOR GEE")
    print("=" * 65)
    
    try:
        geometries = build_geometries(
            shapefile_path  = SHAPEFILE_PATH,
            district_column = DISTRICT_COLUMN,
            project_id      = PROJECT_ID,
        )
        
        if geometries:
            print("\n" + "=" * 65)
            print("  SUCCESS — geometries ready for GEE enrichment")
            print("=" * 65)
            print("\nNext steps:")
            print("  1. Run: python rainfall_pipeline.py")
            print("     → This will use cached GEE features if available")
            print("\n  2. To refresh GEE data, update rainfall_pipeline.py:")
            print("     → Change use_gee=False to use_gee=True")
            print("     → Or call: run_pipeline(use_gee=True, district_geometries=geoms)")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check:")
        print(f"  1. Shapefile exists at: {SHAPEFILE_PATH}")
        print("  2. Update SHAPEFILE_PATH in this script if needed")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()