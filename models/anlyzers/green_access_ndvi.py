import os, tempfile, base64
import math
import warnings
from datetime import date
# from memory_profiler import profile

try:
    import ee
    import folium
    import networkx as nx
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import Point, LineString, Polygon, MultiPolygon
    from shapely.ops import unary_union
except Exception as e:
    raise SystemExit(
        f"\nImport error: {e}\n\n"
        "This often happens when GeoPandas/Shapely wheels were built for NumPy 1.x but you're on NumPy 2.x.\n"
        "Quick fix (in a clean venv):\n"
        "  pip install 'numpy<2'\n"
        "  pip install --force-reinstall geopandas shapely pyproj fiona rtree\n"
        "Or use a fresh 'conda create -n aoi python=3.11' env and install the deps.\n"
    )

try:
    from osmnx.features import features_from_polygon as osm_features_from_polygon
except Exception:
    try:
        from osmnx import geometries_from_polygon as osm_features_from_polygon
    except Exception:
        raise SystemExit("Your osmnx version is missing polygon geometries. Please: pip install --upgrade osmnx.")


warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# FEATURE FLAGS (toggle heavy layers quickly)
# ----------------------------
DO_POP   = True   # population in 10-min walkshed
DO_HAND  = True   # DEM low-lying proxy + slope
DO_SMAP  = True   # SMAP soil moisture
DO_HEAT  = True   # MODIS/ECOSTRESS heat
DO_SOIL  = True   # Soil properties
DO_COUNTS= True   # Nearby counts (kept for popups/CSV; still not printed)

# ----------------------------
# SETTINGS
# ----------------------------
# PLACE = "Narayanganj, Dhaka Division, Bangladesh"

# NDVI
NDVI_GREEN_MIN = 0.35

# Walking thresholds
T5 = 5 * 60
T10 = 10 * 60
WALK_MPS = 1.3

# OSM green tags
GREEN_TAGS = {
    "leisure": ["park", "garden"],
    "landuse": ["recreation_ground", "grass"],
    "natural": ["wood"],
}

EDGE_BUFFER_M = 25
TOP_N_CANDIDATES = 20

# GEE composite tries
DATE_TRIES = [
    ("2025-09-01", "2025-09-25", 20),
    ("2025-08-01", "2025-09-25", 40),
    ("2025-06-01", "2025-09-25", 80),
]

# Context radii / scales (tuned for speed)
SITE_BUFFER_M = 800
SITE_BUFFER_M_FALLBACK = 1200
WATER_STATS_RADIUS_M = 150

CURRENT_YEAR = date.today().year
HEAT_START = f"{CURRENT_YEAR}-04-01"
HEAT_END   = f"{CURRENT_YEAR}-06-30"

SMAP_DAYS = 30

# EE scales
POP_SCALE   = 200   # coarser than 100 m → faster population sums
DEM_SCALE   = 30
MODIS_SCALE = 1000
ECOS_SCALE  = 100
SOIL_SCALE  = 250
JRC_SCALE   = 30
SMAP_SCALE  = 9000

# DEM neighborhood for 5th percentile (HAND proxy)
HAND_RADIUS_M = 1500  # was 2000 → faster


# ----------------------------
# EE INIT
# ----------------------------
def ee_init_headless():
    sa = os.environ["EE_SERVICE_ACCOUNT"]       # ee-runner@<project>.iam.gserviceaccount.com
    key_b64 = os.environ["EE_KEY_B64"]          # base64 of the JSON key

    # Write key to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(base64.b64decode(key_b64).decode("utf-8"))
        key_path = f.name

    creds = ee.ServiceAccountCredentials(sa, key_path)
    ee.Initialize(credentials=creds)

# ----------------------------
# NDVI → green polygons
# ----------------------------
def choose_s2_composite(aoi_geom):
    for (start, end, cloud) in DATE_TRIES:
        s2sr = (ee.ImageCollection("COPERNICUS/S2_SR")
                .filterBounds(aoi_geom).filterDate(start, end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)))
        if s2sr.size().getInfo() > 0:
            return s2sr.median(), f"S2_SR {start}..{end} cloud<{cloud}%"
        s2l1c = (ee.ImageCollection("COPERNICUS/S2")
                 .filterBounds(aoi_geom).filterDate(start, end)
                 .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)))
        if s2l1c.size().getInfo() > 0:
            return s2l1c.median(), f"S2_L1C {start}..{end} cloud<{cloud}%"
    raise SystemExit("No recent Sentinel-2 scenes for AOI after fallbacks.")

def gee_green_polygons(aoi_geom, ndvi_min=NDVI_GREEN_MIN, scale=30, max_features=700):
    composite, desc = choose_s2_composite(aoi_geom)
    print("GEE composite picked:", desc)
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    green_mask = ndvi.gte(ndvi_min).selfMask()
    vectors = green_mask.reduceToVectors(
        geometry=aoi_geom, scale=scale, geometryType="polygon",
        bestEffort=True, maxPixels=1e13
    ).limit(max_features)
    fc = vectors.getInfo()
    feats = fc.get("features", [])
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    return gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

# ----------------------------
# EE helpers (batched reducers)
# ----------------------------
def ee_geom_from_shapely(geom):
    return ee.Geometry(geom.__geo_interface__)

def fc_from_buffers(ids, lonlats, radius_m):
    feats = []
    for cid, (lon, lat) in zip(ids, lonlats):
        feats.append(ee.Feature(ee.Geometry.Point([lon, lat]).buffer(radius_m), {"cid": int(cid)}))
    return ee.FeatureCollection(feats)

def fc_from_polys(ids, shapely_polys):
    feats = []
    for cid, shp in zip(ids, shapely_polys):
        if shp is None: 
            continue
        if isinstance(shp, (Polygon, MultiPolygon)):
            feats.append(ee.Feature(ee_geom_from_shapely(shp), {"cid": int(cid)}))
    return ee.FeatureCollection(feats)

def reduce_regions_to_dict(image, fc, reducer, scale):
    """Run reduceRegions and return dict: cid -> properties dict (including band-named keys)."""
    out = {}
    try:
        coll = image.reduceRegions(collection=fc, reducer=reducer, scale=scale)
        data = coll.getInfo().get("features", [])
        for f in data:
            props = f.get("properties", {})
            cid = int(props.get("cid"))
            out[cid] = props
    except Exception:
        pass
    return out

def first_number(props, preferred_keys=None):
    """Get a numeric value from a properties dict, preferring certain keys."""
    if not isinstance(props, dict):
        return None
    if preferred_keys:
        for k in preferred_keys:
            if k in props:
                try:
                    return float(props[k])
                except Exception:
                    pass
    for k, v in props.items():
        if k == "cid": 
            continue
        try:
            return float(v)
        except Exception:
            continue
    return None

# ----------------------------
# DATASETS (images)
# ----------------------------
def jrc_occurrence_img():
    try:
        return ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    except Exception:
        return None

def soil_sources_images():
    out = []
    # SoilGrids 2020
    try:
        img = ee.Image("ISRIC/SoilGrids/2020")
        bands = img.bandNames().getInfo()
        def pick(tokens):
            for b in bands:
                bl = b.lower()
                if all(t in bl for t in tokens): return b
            return None
        bmap = {
            "soil_ph":   pick(("phh2o","0-5","mean")) or pick(("phh2o","0-5cm","mean")),
            "soil_clay": pick(("clay","0-5","mean"))  or pick(("clay","0-5cm","mean")),
            "soil_sand": pick(("sand","0-5","mean"))  or pick(("sand","0-5cm","mean")),
            "soil_soc":  pick(("soc","0-5","mean"))   or pick(("soc","0-5cm","mean")),
        }
        selects = [b for b in bmap.values() if b]
        if selects:
            out.append(("SoilGrids2020", img.select(selects)))
    except Exception:
        pass

    # SoilGrids 250m
    try:
        img = ee.Image("ISRIC/SoilGrids/250m")
        bands = img.bandNames().getInfo()
        def pick(tokens):
            for b in bands:
                bl = b.lower()
                if all(t in bl for t in tokens): return b
            return None
        bmap = {
            "soil_ph":   pick(("phh2o","0-5","mean")) or pick(("phh2o","0-5cm","mean")),
            "soil_clay": pick(("clay","0-5","mean"))  or pick(("clay","0-5cm","mean")),
            "soil_sand": pick(("sand","0-5","mean"))  or pick(("sand","0-5cm","mean")),
            "soil_soc":  pick(("soc","0-5","mean"))   or pick(("soc","0-5cm","mean")),
        }
        selects = [b for b in bmap.values() if b]
        if selects:
            out.append(("SoilGrids250m", img.select(selects)))
    except Exception:
        pass

    # OpenLandMap fallbacks
    try:
        ph   = ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02")
        clay = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLAY_USDA-3A1A1A_M/v02")
        sand = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-SAND_USDA-3A1A1A_M/v02")
        soc  = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02")
        out.append(("OpenLandMap", ph.addBands([clay, sand, soc])))
    except Exception:
        pass

    return out

def population_image(aoi):
    for yr in [2025, 2023, 2022, 2021, 2020, 2019]:
        try:
            col = ee.ImageCollection("WorldPop/GP/100m/pop").filterBounds(aoi).filter(ee.Filter.eq('year', yr))
            if col.size().getInfo() > 0:
                img = col.mosaic()
                bname = img.bandNames().getInfo()[0]
                return img.select(bname, ["pop"])
        except Exception:
            pass
    try:
        img = ee.Image("JRC/GHSL/P2019/POP_GLOBE_R2019A")
        bands = [b for b in img.bandNames().getInfo() if "2020" in b or "2015" in b]
        if bands:
            return img.select(bands[0], ["pop"])
    except Exception:
        pass
    try:
        col = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Count").filter(ee.Filter.eq("year", 2020))
        img = col.first()
        if img:
            b = img.bandNames().getInfo()[0]
            return img.select(b, ["pop"])
    except Exception:
        pass
    return None

def dem_and_slope():
    try:
        dem = ee.Image("COPERNICUS/DEM/GLO30")
    except Exception:
        dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem).rename("slope")
    return dem.rename("elevation"), slope

def heat_images(aoi):
    modis = None
    eco = None
    try:
        col = (ee.ImageCollection("MODIS/061/MOD11A2")
               .filterBounds(aoi).filterDate(HEAT_START, HEAT_END).select("LST_Day_1km"))
        if col.size().getInfo() > 0:
            modisK = col.mean()
            modis = modisK.multiply(0.02).subtract(273.15).rename("LST_modis_C")
    except Exception:
        pass
    try:
        col = (ee.ImageCollection("NASA/JPL/ECOSTRESS/L2_LSTE")
               .filterBounds(aoi).filterDate(HEAT_START, HEAT_END).select("LST"))
        if col.size().getInfo() > 0:
            ecoK = col.mean()
            eco = ecoK.subtract(273.15).rename("LST_eco_C")
    except Exception:
        pass
    return modis, eco

def smap_image(aoi):
    try:
        end = ee.Date(date.today().isoformat())
        start = end.advance(-SMAP_DAYS, 'day')
        col = (ee.ImageCollection("NASA/SMAP/SPL3SMP_E")
               .filterBounds(aoi).filterDate(start, end).select("soil_moisture"))
        if col.size().getInfo() == 0:
            return None
        return col.mean().rename("soil_moisture")
    except Exception:
        return None

# ----------------------------
# INTERPRETERS
# ----------------------------
def fmt_meters_and_walk(min_meters):
    if min_meters is None:
        return "unknown distance"
    minutes = (min_meters / WALK_MPS) / 60.0
    return f"~{round(min_meters):,} m (~{round(minutes,1)} min walk)"

def interpret_water_occurrence(p):
    if p is None:  return "No satellite evidence of open water nearby (or data unavailable)."
    if p < 5:      return "Very rarely wet — area is usually dry."
    if p < 20:     return "Occasionally wet — may pond during heavy rain."
    if p < 50:     return "Seasonally wet — expect water presence in some months."
    return "Frequently/permanently wet — likely near river/pond or flood-prone."

def interpret_ph(ph):
    if ph is None:       return "Soil pH unknown here."
    if ph < 5.5:         return f"Acidic (pH {ph}) — choose tolerant species."
    if ph <= 7.5:        return f"Near neutral (pH {ph}) — good for most plants."
    return f"Alkaline (pH {ph}) — choose tolerant species."

def interpret_texture(sand_pct, clay_pct):
    if sand_pct is None or clay_pct is None:
        return "Soil texture unknown."
    if sand_pct >= 60 and clay_pct < 20:
        return f"Sandy ({sand_pct}% sand) — drains fast; add organic matter."
    if clay_pct >= 35:
        return f"Clayey ({clay_pct}% clay) — slow drainage; raised beds help."
    return f"Loamy mix (sand {sand_pct}%, clay {clay_pct}%) — generally good."

def interpret_distance_to_water(d):
    if d is None: return "Distance to water unknown."
    if d < 100:   return "Very close to water (<100 m)."
    if d < 500:   return "Near water (100–500 m)."
    return "Far from water (>500 m)."

def interpret_density(building_pct, road_km_km2):
    msgs = []
    if building_pct is not None:
        if building_pct < 5:   msgs.append("very low building coverage")
        elif building_pct < 20:msgs.append("low building coverage")
        elif building_pct < 40:msgs.append("moderate building coverage")
        else:                  msgs.append("dense built-up surroundings")
    if road_km_km2 is not None:
        if road_km_km2 < 5:    msgs.append("sparse road network")
        elif road_km_km2 < 15: msgs.append("moderate road network")
        else:                  msgs.append("very dense road network")
    return "; ".join(msgs) if msgs else "Urban density unknown."

def interpret_hand_proxy(hm, slope_deg, gsw_mean):
    parts = []
    if hm is None:
        parts.append("Low-lying risk unknown")
    else:
        if hm < 1:       parts.append("Very low ground (likely ponding)")
        elif hm < 3:     parts.append("Low ground")
        elif hm < 7:     parts.append("Moderate elevation")
        else:            parts.append("High relative elevation")
    if slope_deg is not None:
        if slope_deg < 1: parts.append("very flat; slow drainage")
        elif slope_deg < 3:parts.append("gentle slope")
        else:              parts.append("noticeable slope")
    if gsw_mean is not None and gsw_mean >= 20:
        parts.append("historic water nearby")
    return " • ".join(parts)

def interpret_heat(modis_c, eco_c):
    val = eco_c if eco_c is not None else modis_c
    if val is None: return "Heat unknown."
    if val >= 42:   return f"Heat: very high (~{val}°C daytime LST)."
    if val >= 38:   return f"Heat: high (~{val}°C)."
    if val >= 34:   return f"Heat: moderate (~{val}°C)."
    return f"Heat: mild (~{val}°C)."

# ----------------------------
# OSM helpers
# ----------------------------
POI_CATEGORIES = {
    "schools":      ("amenity", ["school"]),
    "colleges":     ("amenity", ["college"]),
    "universities": ("amenity", ["university"]),
    "hospitals":    ("amenity", ["hospital"]),
    "clinics":      ("amenity", ["clinic"]),
    "pharmacies":   ("amenity", ["pharmacy"]),
    "markets":      ("amenity", ["marketplace"]),
    "libraries":    ("amenity", ["library"]),
    "community":    ("amenity", ["community_centre"]),
    "police":       ("amenity", ["police"]),
    "fire":         ("amenity", ["fire_station"]),
    "worship":      ("amenity", ["place_of_worship"]),
    "playgrounds":  ("leisure", ["playground"]),
    "sports":       ("leisure", ["sports_centre"]),
    "parks_gardens":("leisure", ["park", "garden"]),
    "supermarkets": ("shop", ["supermarket"]),
}

OSM_CONTEXT_TAGS = {
    "amenity": list({v for _, vs in [POI_CATEGORIES[k] for k in POI_CATEGORIES if POI_CATEGORIES[k][0] == "amenity"] for v in vs}),
    "leisure": list({v for _, vs in [POI_CATEGORIES[k] for k in POI_CATEGORIES if POI_CATEGORIES[k][0] == "leisure"] for v in vs}),
    "shop":    ["supermarket"],
    "building": True,
    "natural": ["water", "wetland", "wood"],
    "waterway": True,
    "landuse": ["residential", "commercial", "industrial", "retail", "recreation_ground", "grass"],
}

def fetch_osm_context(aoi_polygon):
    layers = []
    for k, v in OSM_CONTEXT_TAGS.items():
        try:
            g = osm_features_from_polygon(aoi_polygon, tags={k: v})
            if g is not None and not g.empty:
                layers.append(g)
        except Exception:
            pass
    if not layers:
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        return empty, empty, empty

    base_crs = getattr(layers[0], "crs", None) or "EPSG:4326"
    all_feats = gpd.GeoDataFrame(gpd.pd.concat(layers, ignore_index=True), crs=base_crs)
    all_feats = all_feats[all_feats.geometry.notna()].copy()

    is_water = (
        (all_feats.get("natural").isin(["water", "wetland"])) |
        (all_feats.get("waterway").notna())
    ).fillna(False)

    is_building = all_feats.get("building").notna().fillna(False)

    water_gdf = all_feats[is_water].copy()
    buildings_gdf = all_feats[is_building].copy()
    pois_gdf = all_feats[~is_water].copy()
    return pois_gdf.to_crs(epsg=4326), water_gdf.to_crs(epsg=4326), buildings_gdf.to_crs(epsg=4326)

def count_features_in_buffer(pois_proj, buffer_geom, key, values):
    if pois_proj is None or pois_proj.empty or key not in pois_proj.columns:
        return 0
    subset = pois_proj[pois_proj[key].isin(values)]
    if subset.empty:
        return 0
    try:
        idx = subset.sindex
        cand_idx = list(idx.intersection(buffer_geom.bounds))
        subset = subset.iloc[cand_idx]
    except Exception:
        minx, miny, maxx, maxy = buffer_geom.bounds
        subset = subset[
            (subset.geometry.bounds["maxx"] >= minx) &
            (subset.geometry.bounds["minx"] <= maxx) &
            (subset.geometry.bounds["maxy"] >= miny) &
            (subset.geometry.bounds["miny"] <= maxy)
        ]
    if subset.empty:
        return 0
    return int(subset.intersects(buffer_geom).sum())

def building_and_road_density(buildings_proj, edges_proj, buffer_geom):
    try:
        area_m2 = float(buffer_geom.area)
        bldg_pct = None
        road_density = None

        if buildings_proj is not None and not buildings_proj.empty:
            try:
                idx = buildings_proj.sindex
                cand_idx = list(idx.intersection(buffer_geom.bounds))
                bsub = buildings_proj.iloc[cand_idx]
            except Exception:
                bsub = buildings_proj
            bsub = bsub[bsub.geometry.intersects(buffer_geom)]
            if not bsub.empty:
                inter = bsub.geometry.intersection(buffer_geom)
                built_area = float(inter.area.sum())
                if area_m2 > 0:
                    bldg_pct = round(100.0 * built_area / area_m2, 1)

        if edges_proj is not None and not edges_proj.empty:
            try:
                idx = edges_proj.sindex
                cand_idx = list(idx.intersection(buffer_geom.bounds))
                esub = edges_proj.iloc[cand_idx]
            except Exception:
                esub = edges_proj
            esub = esub[esub.geometry.intersects(buffer_geom)]
            if not esub.empty:
                ilen = esub.geometry.intersection(buffer_geom).length.sum()  # meters
                km = float(ilen) / 1000.0
                km2 = area_m2 / 1e6
                if km2 > 0:
                    road_density = round(km / km2, 1)
        return bldg_pct, road_density
    except Exception:
        return None, None

# ----------------------------
# CORE LOGIC
# ----------------------------
def line_midpoint(geom: LineString):
    try:
        return geom.interpolate(0.5, normalized=True)
    except Exception:
        if geom.geom_type == "LineString" and len(geom.coords) >= 2:
            (x1, y1), (x2, y2) = geom.coords[0], geom.coords[-1]
            return Point((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return geom.centroid

def make_iso_polygon(edges_subset, buffer_m=EDGE_BUFFER_M):
    if edges_subset is None or edges_subset.empty:
        return None
    buffered = edges_subset.geometry.buffer(buffer_m)
    merged = unary_union(list(buffered.values))
    return gpd.GeoSeries([merged], crs=edges_subset.crs)

def edges_within_time(Gp, edges_gdf, source_node, cutoff_s):
    times = nx.single_source_dijkstra_path_length(Gp, source=source_node, cutoff=cutoff_s, weight="time_s")
    nodes_in = set(times.keys())
    mask = edges_gdf.apply(lambda r: (r["u"] in nodes_in) or (r["v"] in nodes_in), axis=1)
    return edges_gdf[mask].copy(), times

# ----------------------------
# MAIN
# ----------------------------
# session_id = "aaa"
# gdf = location_gdf
# @profile
def run(session_id = None, gdf = None):
    print("Started green access NDVI analysis. Session ID =", session_id)
    ee_init_headless()

    # Soil sources & images
    soil_imgs = soil_sources_images() if DO_SOIL else []
    dem_img, slope_img = dem_and_slope() if DO_HAND else (None, None)

    # OSMnx
    ox.settings.log_console = True
    ox.settings.use_cache = True
    ox.settings.timeout = 180

    print("Geocoding AOI…")
    aoi = gdf 
    if aoi.empty:
        raise SystemExit("Could not geocode the AOI name.")
    aoi_polygon = aoi.geometry.iloc[0]
    aoi_bounds = aoi.to_crs(epsg=4326).total_bounds
    gee_aoi = ee.Geometry.Rectangle(list(aoi_bounds))

    print("Downloading pedestrian network…")
    G = ox.graph_from_polygon(aoi_polygon, network_type="walk", simplify=True)

    print("Projecting graph to local metric CRS…")
    Gp = ox.project_graph(G)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(Gp)
    if "u" not in edges_gdf.columns or "v" not in edges_gdf.columns:
        edges_gdf = edges_gdf.reset_index()
    graph_crs = nodes_gdf.crs

    # OSM green areas
    print("Downloading OSM green areas…")
    green_layers = []
    for k, v in GREEN_TAGS.items():
        try:
            g = osm_features_from_polygon(aoi_polygon, tags={k: v})
            if g is not None and not g.empty:
                green_layers.append(g)
        except Exception:
            pass
    osm_greens = None
    if green_layers:
        base_crs = getattr(green_layers[0], "crs", None) or "EPSG:4326"
        osm_greens = gpd.GeoDataFrame(gpd.pd.concat(green_layers, ignore_index=True), crs=base_crs)
        osm_greens = osm_greens[osm_greens.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    # NDVI greens
    print("Vectorizing NDVI green polygons (GEE)…")
    ndvi_greens = gee_green_polygons(gee_aoi, ndvi_min=NDVI_GREEN_MIN, scale=30, max_features=700)

    greens_list = []
    if osm_greens is not None and not osm_greens.empty:
        greens_list.append(osm_greens.to_crs(epsg=4326))
    if ndvi_greens is not None and not ndvi_greens.empty:
        greens_list.append(ndvi_greens.to_crs(epsg=4326))
    if not greens_list:
        raise SystemExit("No green polygons found from OSM or NDVI.")

    greens_all = gpd.GeoDataFrame(gpd.pd.concat(greens_list, ignore_index=True), crs="EPSG:4326")
    print(f"Green polygons: OSM={0 if osm_greens is None else len(osm_greens)} | NDVI={len(ndvi_greens)} | merged={len(greens_all)}")

    # Project greens
    greens_poly_proj = greens_all.to_crs(graph_crs)

    # Destination nodes from green centroids
    print("Computing destination nodes from green centroids…")
    greens_poly_proj["centroid"] = greens_poly_proj.geometry.centroid
    dest_nodes = set()
    for c in greens_poly_proj["centroid"]:
        try:
            dest_nodes.add(ox.distance.nearest_nodes(Gp, X=c.x, Y=c.y))
        except Exception:
            pass
    if not dest_nodes:
        raise SystemExit("No destination nodes from green centroids.")

    # Edge costs
    print("Assigning time costs to edges…")
    for u, v, k, data in Gp.edges(keys=True, data=True):
        length_m = float(data.get("length", 0.0)) or 0.0
        data["time_s"] = length_m / WALK_MPS

    # Multi-source Dijkstra (reverse trick)
    print("Running multi-source shortest path (Dijkstra)…")
    Gr = Gp.reverse()
    min_time_s = nx.multi_source_dijkstra_path_length(Gr, sources=list(dest_nodes), weight="time_s")

    def covered_by_threshold(u, v, threshold_s):
        tu = min_time_s.get(u, math.inf); tv = min_time_s.get(v, math.inf)
        return (tu <= threshold_s) or (tv <= threshold_s)

    def both_beyond_10(u, v):
        return (min_time_s.get(u, math.inf) > T10) and (min_time_s.get(v, math.inf) > T10)

    print("Classifying edges by coverage…")
    edges_gdf["covered_5min"] = edges_gdf.apply(lambda r: covered_by_threshold(r["u"], r["v"], T5), axis=1)
    edges_gdf["covered_10min"] = edges_gdf.apply(lambda r: covered_by_threshold(r["u"], r["v"], T10), axis=1)
    edges_gdf["uncovered_10min"] = edges_gdf.apply(lambda r: both_beyond_10(r["u"], r["v"]), axis=1)
    uncovered = edges_gdf[edges_gdf["uncovered_10min"]].copy()
    print(f"Uncovered road segments >10 min: {len(uncovered)}")

    # Isochrones (for display)
    print("Building isochrone polygons…")
    iso5_edges = edges_gdf[edges_gdf["covered_5min"]]
    iso10_edges = edges_gdf[edges_gdf["covered_10min"]]
    iso5_poly = make_iso_polygon(iso5_edges, buffer_m=EDGE_BUFFER_M)
    iso10_poly = make_iso_polygon(iso10_edges, buffer_m=EDGE_BUFFER_M)

    # Candidates: midpoints of longest uncovered segments (dedup)
    print("Selecting candidate micro-park points…")
    uncovered["length_m"] = uncovered.geometry.length
    candidates = uncovered.sort_values("length_m", ascending=False).head(3 * TOP_N_CANDIDATES).copy()
    candidates["midpt"] = candidates.geometry.apply(line_midpoint)
    cand_proj = gpd.GeoDataFrame(geometry=candidates["midpt"], crs=edges_gdf.crs)
    cand_wgs84 = cand_proj.to_crs(epsg=4326)  # <-- FIX: transform the GeoDataFrame (not Points)
    cand_wgs84["xy_round"] = cand_wgs84.geometry.apply(lambda g: (round(g.x, 6), round(g.y, 6)))
    cand_wgs84 = cand_wgs84.drop_duplicates(subset="xy_round").head(TOP_N_CANDIDATES).copy()
    cand_proj = cand_wgs84.to_crs(edges_gdf.crs)[["geometry"]].copy()  # keep projected too

    # OSM context
    print("Downloading OSM context layers…")
    pois_wgs84, water_wgs84, buildings_wgs84 = fetch_osm_context(aoi_polygon)
    pois_proj = pois_wgs84.to_crs(graph_crs) if not pois_wgs84.empty else pois_wgs84
    water_proj = water_wgs84.to_crs(graph_crs) if not water_wgs84.empty else water_wgs84
    buildings_proj = buildings_wgs84.to_crs(graph_crs) if not buildings_wgs84.empty else buildings_wgs84
    water_union = unary_union(list(water_proj.geometry.values)) if (water_proj is not None and not water_proj.empty) else None

    # Assemble candidate basics
    ids = list(range(1, len(cand_proj) + 1))
    # Use the already-transformed WGS84 points for lon/lat lists  <-- FIXED
    lonlats = [(pt.x, pt.y) for pt in cand_wgs84.geometry]
    nearest_nodes = [ox.distance.nearest_nodes(Gp, X=geom.x, Y=geom.y) for geom in cand_proj.geometry]

    # Build 10-min isochrone polygons once (local; 20 sites → OK)
    iso_polys = []
    for nid in nearest_nodes:
        edges_iso, _times = edges_within_time(Gp, edges_gdf, nid, T10)
        iso_polys.append(make_iso_polygon(edges_iso, buffer_m=EDGE_BUFFER_M).iloc[0] if (edges_iso is not None and not edges_iso.empty) else None)

    # ----------------------------
    # BATCHED EE CALLS
    # ----------------------------
    print("\nComputing batched metrics (EE)…")

    # Population (sum over isochrone polygons)
    pop_results = {}
    if DO_POP:
        pop_img = population_image(gee_aoi)
        if pop_img is not None:
            fc_iso = fc_from_polys(ids, iso_polys)
            pop_results = reduce_regions_to_dict(
                image=pop_img, fc=fc_iso, reducer=ee.Reducer.sum(), scale=POP_SCALE
            )
        else:
            print("Population raster not available; walkshed population will be 'n/a'.")

    # JRC water (mean and max in 150 m) — do as two simple batched calls (robust)
    jrc_mean_results = {}
    jrc_max_results = {}
    occ = jrc_occurrence_img()
    if occ is not None:
        fc_water = fc_from_buffers(ids, lonlats, WATER_STATS_RADIUS_M)
        jrc_mean_results = reduce_regions_to_dict(
            image=occ, fc=fc_water, reducer=ee.Reducer.mean(), scale=JRC_SCALE
        )
        jrc_max_results = reduce_regions_to_dict(
            image=occ, fc=fc_water, reducer=ee.Reducer.max(), scale=JRC_SCALE
        )

    # Soil (buffered mean 150 m) with fallbacks (run per source, but batched)
    soil_results = {}
    if DO_SOIL and soil_imgs:
        fc_soil = fc_from_buffers(ids, lonlats, 150)
        for label, img in soil_imgs:
            tmp = reduce_regions_to_dict(
                image=img, fc=fc_soil, reducer=ee.Reducer.mean(), scale=SOIL_SCALE
            )
            for cid in ids:
                if cid in tmp and cid not in soil_results:
                    # store label + properties dict
                    soil_results[cid] = (label, tmp[cid])
    for cid in ids:
        if cid not in soil_results:
            soil_results[cid] = (None, {})

    # DEM low-lying proxy & slope (batched)
    hand_results = {}
    slope_results = {}
    if DO_HAND and dem_img is not None and slope_img is not None:
        kernel = ee.Kernel.circle(HAND_RADIUS_M, 'meters')
        try:
            p5 = dem_img.reduceNeighborhood(ee.Reducer.percentile([5]), kernel)
            hand_img = dem_img.subtract(p5).rename("hand_proxy")
            fc_hand = fc_from_buffers(ids, lonlats, 30)   # small buffer
            fc_slope = fc_from_buffers(ids, lonlats, 60)  # slope a bit larger
            hand_results = reduce_regions_to_dict(
                image=hand_img, fc=fc_hand, reducer=ee.Reducer.mean(), scale=DEM_SCALE
            )
            slope_results = reduce_regions_to_dict(
                image=slope_img, fc=fc_slope, reducer=ee.Reducer.mean(), scale=DEM_SCALE
            )
        except Exception:
            pass

    # Heat (batched)
    heat_modis = {}
    heat_eco = {}
    if DO_HEAT:
        modis_img, eco_img = heat_images(gee_aoi)
        fc_heat = fc_from_buffers(ids, lonlats, 300)
        if modis_img is not None:
            heat_modis = reduce_regions_to_dict(
                image=modis_img, fc=fc_heat, reducer=ee.Reducer.mean(), scale=MODIS_SCALE
            )
        if eco_img is not None:
            heat_eco = reduce_regions_to_dict(
                image=eco_img, fc=fc_heat, reducer=ee.Reducer.mean(), scale=ECOS_SCALE
            )

    # SMAP (batched)
    smap_results = {}
    if DO_SMAP:
        smap_img = smap_image(gee_aoi)
        if smap_img is not None:
            fc_smap = fc_from_buffers(ids, lonlats, 800)  # trimmed buffer
            smap_results = reduce_regions_to_dict(
                image=smap_img, fc=fc_smap, reducer=ee.Reducer.mean(), scale=SMAP_SCALE
            )

    # ----------------------------
    # Local OSM-derived metrics (fast w/ sindex)
    # ----------------------------
    print("Computing local OSM metrics…")
    all_counts = []
    all_build_road = []
    for cid, proj_geom in zip(ids, cand_proj.geometry):
        radius_used = SITE_BUFFER_M
        buf = proj_geom.buffer(radius_used)

        def compute_counts(buffer_geom):
            counts_local = {}
            if not DO_COUNTS or pois_proj is None or pois_proj.empty:
                for label in POI_CATEGORIES: counts_local[label] = 0
                return counts_local
            for label, (key, values) in POI_CATEGORIES.items():
                counts_local[label] = count_features_in_buffer(pois_proj, buffer_geom, key, values)
            return counts_local

        counts = compute_counts(buf)
        if DO_COUNTS and sum(counts.values()) == 0 and SITE_BUFFER_M_FALLBACK and SITE_BUFFER_M_FALLBACK > SITE_BUFFER_M:
            radius_used = SITE_BUFFER_M_FALLBACK
            buf = proj_geom.buffer(radius_used)
            counts = compute_counts(buf)

        bldg_pct, road_density = building_and_road_density(buildings_proj, edges_gdf, buf)
        all_counts.append((radius_used, counts))
        all_build_road.append((bldg_pct, road_density))

    # ----------------------------
    # Assemble per-candidate results
    # ----------------------------
    print("\nBuilding Folium map…")
    aoi_latlon = aoi.to_crs(epsg=4326)
    center = [aoi_latlon.geometry.iloc[0].centroid.y, aoi_latlon.geometry.iloc[0].centroid.x]

    edges_latlon = edges_gdf.to_crs(epsg=4326)
    uncovered_latlon = uncovered.to_crs(epsg=4326)
    greens_latlon = greens_poly_proj.to_crs(epsg=4326)
    cand_latlon_final = cand_proj.to_crs(epsg=4326)  # for plotting markers
    iso5_latlon = iso5_poly.to_crs(epsg=4326) if iso5_poly is not None else None
    iso10_latlon = iso10_poly.to_crs(epsg=4326) if iso10_poly is not None else None

    m = folium.Map(location=center, zoom_start=12, control_scale=True, tiles="cartodbpositron")
    folium.GeoJson(
        greens_latlon[["geometry"]],
        name=f"Green areas (OSM + NDVI≥{NDVI_GREEN_MIN:.2f})",
        style_function=lambda _: {"color": "#2e7d32", "weight": 1, "fillColor": "#66bb6a", "fillOpacity": 0.35},
    ).add_to(m)
    if iso10_latlon is not None:
        folium.GeoJson(iso10_latlon.__geo_interface__, name="Within 10 min of green",
                        style_function=lambda _: {"color": "#ff9800", "weight": 1, "fillColor": "#ffcc80", "fillOpacity": 0.25}).add_to(m)
    if iso5_latlon is not None:
        folium.GeoJson(iso5_latlon.__geo_interface__, name="Within 5 min of green",
                        style_function=lambda _: {"color": "#1976d2", "weight": 1, "fillColor": "#90caf9", "fillOpacity": 0.25}).add_to(m)
    folium.GeoJson(
        uncovered_latlon[["geometry"]],
        name="Road segments beyond 10 min (need green access)",
        style_function=lambda _: {"color": "#e53935", "weight": 2, "opacity": 0.9},
    ).add_to(m)

    description = ""

    description += ("\n================= Candidate Site Context =================\n")
    summary_rows = []

    for idx, (cid, latlon_geom, proj_geom) in enumerate(zip(ids, cand_latlon_final.geometry, cand_proj.geometry), start=1):
        lat, lon = latlon_geom.y, latlon_geom.x

        # Lookups from batched dicts
        pop_val = first_number(pop_results.get(cid, {}), ["pop", "sum"]) if DO_POP else None

        gsw_mean = first_number(jrc_mean_results.get(cid, {}), ["occurrence", "mean"])
        gsw_max  = first_number(jrc_max_results.get(cid, {}), ["occurrence", "max"])

        soil_label, soil_props = soil_results.get(cid, (None, {}))
        def _get(d, keys):
            for k in d.keys():
                lk = k.lower()
                if all(t in lk for t in keys):
                    try:
                        return float(d[k])
                    except Exception:
                        pass
            return None
        soil_ph   = round(_get(soil_props, ("ph",))        , 2) if _get(soil_props, ("ph",))         is not None else None
        soil_clay = round(_get(soil_props, ("clay",))      , 1) if _get(soil_props, ("clay",))       is not None else None
        soil_sand = round(_get(soil_props, ("sand",))      , 1) if _get(soil_props, ("sand",))       is not None else None
        soil_soc  = round(_get(soil_props, ("org","carb")) , 1) if _get(soil_props, ("org","carb"))  is not None else None
        if soil_ph is None and soil_clay is None and soil_sand is None and soil_soc is None:
            soil_label = None

        hand_val  = first_number(hand_results.get(cid, {}), ["hand_proxy", "mean"]) if DO_HAND else None
        slope_val = first_number(slope_results.get(cid, {}), ["slope", "mean"]) if DO_HAND else None
        modis_c   = first_number(heat_modis.get(cid, {}), ["LST_modis_C", "mean"]) if DO_HEAT else None
        eco_c     = first_number(heat_eco.get(cid, {}), ["LST_eco_C", "mean"]) if DO_HEAT else None
        smap_sm   = first_number(smap_results.get(cid, {}), ["soil_moisture", "mean"]) if DO_SMAP else None

        # Distance to OSM water
        dist_to_water_m = None
        if water_union is not None:
            try:
                dist_to_water_m = round(float(proj_geom.distance(water_union)), 1)
            except Exception:
                pass

        # Counts & urban form (local)
        radius_used, counts = all_counts[idx-1]
        bldg_pct, road_density = all_build_road[idx-1]

        # Interpretations
        water_distance_msg = f"{interpret_distance_to_water(dist_to_water_m)} — {fmt_meters_and_walk(dist_to_water_m)}."
        water_occ_msg = interpret_water_occurrence(gsw_mean)
        ph_msg = interpret_ph(soil_ph)
        texture_msg = interpret_texture(soil_sand, soil_clay)
        density_msg = interpret_density(bldg_pct, road_density)
        heat_msg = interpret_heat(modis_c, eco_c)
        hand_msg = interpret_hand_proxy(hand_val, slope_val, gsw_mean)

        # ---- CLEAN CONSOLE OUTPUT (no nearby-counts line) ----
        description += (f"\n📍 Candidate #{cid}  (Lat, Lon: {lat:.6f}, {lon:.6f})\n")
        if DO_POP and pop_val is not None:
            description += (f"  People within a 10-min walk (estimated): ~{int(pop_val):,}\n")
        description += (f"  Water: nearest mapped water is {fmt_meters_and_walk(dist_to_water_m)}. {water_distance_msg}\n")
        description += (f"  Water presence (satellite history ≤{WATER_STATS_RADIUS_M} m): mean {gsw_mean}, max {gsw_max}. {water_occ_msg}\n")
        description += (f"  Soil (0–5 cm): pH={soil_ph} (src: {soil_label}), clay%={soil_clay}, sand%={soil_sand}, SOC g/kg={soil_soc}.\n")
        description += (f"  ↳ {ph_msg}  |  {texture_msg}\n")
        if DO_HAND:
            description += (f"  Terrain: HAND-proxy {hand_val} m; slope ~{slope_val}°. {hand_msg}\n")
        if DO_HEAT:
            description += (f"  Heat (Apr–Jun): MODIS≈{modis_c}°C"
                    f"{' | ECOSTRESS≈'+str(eco_c)+'°C' if eco_c is not None else ''}. {heat_msg}\n")
        if DO_SMAP and smap_sm is not None:
            description += (f"  Soil moisture (SMAP {SMAP_DAYS}-day mean): {smap_sm} m³/m³\n")
        if bldg_pct is not None or road_density is not None:
            description += (f"  Urban form: building cover ~{bldg_pct if bldg_pct is not None else 'n/a'}% "
                    f"& roads ~{road_density if road_density is not None else 'n/a'} km/km² → {density_msg}\n")
        # Map popup (keeps counts)
        description_html = f"""
        <div style='font-size:12px;line-height:1.35' id = 'popup-desc_<<cid>>'>
            <i>Lat, Lon:</i> {lat:.6f}, {lon:.6f}<br>
            <b>People within 10-min walk:</b> {('{:,}'.format(int(pop_val)) if (DO_POP and pop_val is not None) else 'n/a')}<br>
            <b>Nearby (≤{radius_used} m)</b><br>
            schools:{counts.get('schools',0)} · colleges:{counts.get('colleges',0)} · universities:{counts.get('universities',0)}<br>
            hospitals:{counts.get('hospitals',0)} · clinics:{counts.get('clinics',0)} · pharmacies:{counts.get('pharmacies',0)}<br>
            markets:{counts.get('markets',0)} · worship:{counts.get('worship',0)} · supermarkets:{counts.get('supermarkets',0)}<br>
            playgrounds:{counts.get('playgrounds',0)} · sports:{counts.get('sports',0)} · parks/gardens:{counts.get('parks_gardens',0)}<br>
            <b>Water</b><br>
            nearest: {fmt_meters_and_walk(dist_to_water_m)}<br>
            JRC occurrence (≤{WATER_STATS_RADIUS_M} m, mean/max): {gsw_mean} / {gsw_max}<br>
            <i>{water_occ_msg}</i><br>
            <b>Soil (0–5 cm)</b><br>
            pH: {soil_ph} (src: {soil_label}) — <i>{ph_msg}</i><br>
            texture: sand {soil_sand}% / clay {soil_clay}% — <i>{texture_msg}</i><br>
            SOC: {soil_soc} g/kg<br>
            <b>Terrain</b><br>
            HAND-proxy: {hand_val} m · slope: {slope_val}°<br>
            <i>{hand_msg}</i><br>
            <b>Heat (Apr–Jun)</b><br>
            MODIS: {modis_c}°C{(' · ECOSTRESS: '+str(eco_c)+'°C') if eco_c is not None else ''}<br>
            <i>{heat_msg}</i><br>
            <b>Soil moisture</b><br>
            SMAP {SMAP_DAYS}-day mean: {smap_sm} m³/m³<br>
            <b>Urban form</b><br>
            buildings ~{bldg_pct if bldg_pct is not None else 'n/a'}% · roads ~{road_density if road_density is not None else 'n/a'} km/km²<br>
            <i>{density_msg}</i>
        </div>
        """

        button_html =  """<center><button id="popup-btn_<<cid>>"
                    style="margin-top:10px; padding:6px 10px; border:none; border-radius:6px;
                            background:#2563eb; color:white; cursor:pointer; font-size:13px;"
                    onclick="getAiSuggestions(<<cid>>)">
                Generate AI Suggestions
            </button></center>

            <div id = "ai-answer_<<cid>>"><<great_blank>></div>
            
            """.replace("<<great_blank>>", "&nbsp; "*100 + "<br>"*10)
        
        html_code = description_html + button_html
        html_code = html_code.replace("<<cid>>", str(cid))

        # iframe = folium.IFrame(html=html_code, width=300, height=350)

        folium.CircleMarker(
            location=(lat, lon), radius=6, color="#2962FF",
            fill=True, fill_color="#2962FF", fill_opacity=0.95,
            popup=folium.Popup(html_code, max_width=700, max_height=600),
        ).add_to(m)

        # CSV row
        row = {
            "candidate_id": cid,
            "lat": lat, "lon": lon,
            "radius_used_m": radius_used,
            "people_walk10": int(pop_val) if (DO_POP and pop_val is not None) else None,
            "dist_to_osm_water_m": dist_to_water_m,
            "gsw_occ_mean": gsw_mean, "gsw_occ_max": gsw_max,
            "soil_ph_0_5cm": soil_ph, "soil_clay_pct_0_5cm": soil_clay,
            "soil_sand_pct_0_5cm": soil_sand, "soil_soc_gkg_0_5cm": soil_soc,
            "soil_source": soil_label,
            "hand_proxy_m": hand_val, "slope_deg": slope_val,
            "modis_lst_C": modis_c, "ecostress_lst_C": eco_c,
            "smap_sm_m3m3": smap_sm,
            "building_cover_pct": bldg_pct, "road_km_per_km2": road_density,
        }
        for k in POI_CATEGORIES.keys():
            row[f"cnt_{k}__{radius_used}m"] = counts.get(k, 0)
        summary_rows.append(row)

    # Legend & controls
    legend_html = f"""
    <div style="position: fixed; bottom: 18px; left: 18px; z-index:9999; background: white;
                padding: 10px 12px; border: 1px solid #ccc; border-radius: 6px; font-size: 13px;">
        <b>Legend</b><br>
        <span style="display:inline-block;width:12px;height:12px;background:#66bb6a;border:1px solid #2e7d32;"></span>
        Green areas (OSM + NDVI≥{NDVI_GREEN_MIN:.2f})<br>
        <span style="display:inline-block;width:12px;height:12px;background:#ffcc80;border:1px solid #ff9800;"></span>
        ≤ 10 min walk<br>
        <span style="display:inline-block;width:12px;height:12px;background:#90caf9;border:1px solid #1976d2;"></span>
        ≤ 5 min walk<br>
        <span style="display:inline-block;width:18px;height:2px;background:#e53935;vertical-align:middle;display:inline-block;"></span>
        Uncovered roads (> 10 min)<br>
        <span style="display:inline-block;width:12px;height:12px;background:#2962FF;border:1px solid #2962FF;"></span>
        Candidate micro-park
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    javascript = """<script>
            function getAiSuggestions(cid) {
                var btn = document.getElementById("popup-btn_" + cid);
                var desc = document.getElementById("popup-desc_" + cid);
                var aiBox = document.getElementById("ai-answer_" + cid);

                if (!btn || !desc) {console.log("Button or description not found"); return;}

                btn.addEventListener("click", async function() {
                    btn.disabled = true;
                    btn.innerText = "Generating...";
                    try {
                        response = await fetch(
                            "/llm-inference",
                            {
                                method: "POST",
                                headers: {"Content-Type": "application/json"},
                                body: JSON.stringify({
                                    prompt: desc.innerHTML,
                                    type: "uhi"
                                })
                            }
                        );

                        if (!response.ok) {
                            console.log("Error in response");
                            aiBox.innerHTML = "<hr><b>AI Suggestions:</b><br>" + content;
                        } else {
                            var data = await response.json();
                            var content = data.response;
                            aiBox.innerHTML = "<hr><h2>AI Suggestions:</h2><br>" + content;
                            btn.innerText = "Done";
                        }
                    } catch (error) {
                        aiBox.innerHTML = "<hr><b>AI Suggestions:</b><br>Error during request." + error;
                    }
                });
            };
            </script>"""
    m.get_root().html.add_child(folium.Element(javascript))
    folium.LayerControl(collapsed=False).add_to(m)

    out_html = os.path.join("web_outputs", session_id , "green_access.html")
    out_csv = os.path.join("web_outputs", session_id , "green_access.csv")

    # Save map
    m.save(out_html)
    print(f"\n✅ Saved map in current folder: {out_html}")

    # Save CSV
    try:
        df = gpd.pd.DataFrame(summary_rows)
        df.to_csv(out_csv, index=False)
        print(f"✅ Site context CSV saved to: {out_csv}")
    except Exception as e:
        print("Could not write CSV:", e)

    print("\nDone.")