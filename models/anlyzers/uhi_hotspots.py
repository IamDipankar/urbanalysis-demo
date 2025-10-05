import os, tempfile, base64
import math
from datetime import date, timedelta

import ee
import folium
from folium.plugins import MiniMap, Fullscreen, MousePosition, MeasureControl
from shapely.geometry import Point, box
from shapely.ops import unary_union
import dotenv
# from memory_profiler import profile
import markdown
import sys

dotenv.load_dotenv()

# Geo / OSM
try:
    import osmnx as ox
    import geopandas as gpd
except Exception as e:
    raise SystemExit(
        f"Import error: {e}\nInstall: pip install osmnx geopandas rtree\n"
        "If NumPy 2.x issues: pip install 'numpy<2' && reinstall geopandas shapely pyproj fiona rtree"
    )

# ------------------ CONFIG ------------------
AOI_BBOX = [90.32, 23.70, 90.52, 23.86]  # (W,S,E,N) — Narayanganj
DAYS_BACK = 60
END = date.today()
START = END - timedelta(days=DAYS_BACK)

# Sampling / EE
SCALE_M = 1000        # for LST sampling grid (MODIS native ~1 km)
MAX_POINTS = 5000
EE_TILE_SCALE = 4

# Hotspot selection
Z_THRESHOLD = 1.0
PCTL_THRESHOLD = 85.0

# Clustering
EPS_METERS = 1500.0
MIN_SAMPLES = 6

# Concave envelope (morphological alpha-shape)
ALPHA_M = 1200
MIN_ENVELOPE_POINTS = 5
MIN_POLY_AREA_M2 = 2000  # drop tiny artifacts

# Severity buckets by LST z-score
SEVERE_Z = 2.0
HIGH_Z   = 1.5
ELEV_Z   = 1.0

COLORS = {
    "severe": "#b71c1c",  # dark red
    "high":   "#e53935",  # red
    "elev":   "#fb8c00",  # orange
    "envelope": "#6a1b9a",# purple
}

USER = os.getenv("USER") or os.getenv("USERNAME") or "user"
OUT_HTML = f"/Users/{USER}/Downloads/narayanganj_uhi_hotspots.html"

# ------------------ EE INIT ------------------
# @profile
def ee_init_headless():
    sa = os.environ["EE_SERVICE_ACCOUNT"]       # ee-runner@<project>.iam.gserviceaccount.com
    key_b64 = os.environ["EE_KEY_B64"]          # base64 of the JSON key

    # Write key to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(base64.b64decode(key_b64).decode("utf-8"))
        key_path = f.name

    creds = ee.ServiceAccountCredentials(sa, key_path)
    ee.Initialize(credentials=creds)

# ------------------ UTILITIES ------------------
def utm_crs_from_bbox(bbox):
    minx, miny, maxx, maxy = bbox
    lon_c = (minx + maxx) / 2.0
    lat_c = (miny + maxy) / 2.0
    zone = int((lon_c + 180) // 6) + 1
    epsg = 32600 + zone if lat_c >= 0 else 32700 + zone
    return f"EPSG:{epsg}"

def aoi_polygon_wgs84():
    minx, miny, maxx, maxy = AOI_BBOX
    return box(minx, miny, maxx, maxy)

# @profile
def zscores(vals):
    good = [v for v in vals if v is not None and math.isfinite(v)]
    if len(good) < 2: return [0.0 for _ in vals]
    mean = sum(good)/len(good)
    var  = sum((v-mean)**2 for v in good)/len(good)
    std  = math.sqrt(max(var, 1e-12))
    return [0.0 if (v is None or not math.isfinite(v)) else (v-mean)/std for v in vals]

# @profile
def p_rank(all_vals, v):
    s = sorted(all_vals)
    if not s: return 0.0
    cnt = sum(1 for x in s if x <= v)
    return 100.0 * cnt / len(s)

# @profile
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

# @profile
def cluster_dbscan(points, eps_m=EPS_METERS, min_samples=MIN_SAMPLES):
    n = len(points)
    if n == 0: return []
    # light spatial bucketing (~1.1km) to prune distance calcs
    buckets = {}
    for i, p in enumerate(points):
        key = (int(p["lat"]/0.01), int(p["lon"]/0.01))
        buckets.setdefault(key, []).append(i)
    visited = [False]*n
    clusters = [-1]*n
    nbrs = [[] for _ in range(n)]
    for key, idxs in buckets.items():
        kx, ky = key
        cand = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                cand += buckets.get((kx+dx, ky+dy), [])
        for i in idxs:
            for j in cand:
                if j <= i: continue
                if haversine_m(points[i]["lat"], points[i]["lon"], points[j]["lat"], points[j]["lon"]) <= eps_m:
                    nbrs[i].append(j); nbrs[j].append(i)
    cid = 0
    for i in range(n):
        if visited[i]: continue
        visited[i] = True
        if len(nbrs[i]) + 1 < min_samples:
            clusters[i] = -1; continue
        clusters[i] = cid
        seeds = list(nbrs[i]); k = 0
        while k < len(seeds):
            j = seeds[k]
            if not visited[j]:
                visited[j] = True
                if len(nbrs[j]) + 1 >= min_samples:
                    for q in nbrs[j]:
                        if q not in seeds: seeds.append(q)
            if clusters[j] < 0: clusters[j] = cid
            k += 1
        cid += 1
    return clusters

# @profile
def ensure_clusters(hotspots):
    clusters = cluster_dbscan(hotspots, eps_m=EPS_METERS, min_samples=MIN_SAMPLES)
    if not any(c >= 0 for c in clusters):
        clusters = cluster_dbscan(hotspots, eps_m=EPS_METERS*1.6, min_samples=max(3, MIN_SAMPLES-2))
    if not any(c >= 0 for c in clusters):
        clusters = [0 for _ in hotspots]
    return clusters

# @profile
def build_concave_envelopes(hotspots, clusters, metric_crs, alpha_m=ALPHA_M, min_pts=MIN_ENVELOPE_POINTS):
    """Returns dict cid -> list[Polygon in WGS84], tiny parts removed."""
    by_cluster = {}
    for hp, cid in zip(hotspots, clusters):
        if cid < 0: continue
        by_cluster.setdefault(cid, []).append(hp)
    out = {}
    for cid, pts in by_cluster.items():
        if len(pts) < min_pts: continue
        pts_wgs = gpd.GeoSeries([Point(p["lon"], p["lat"]) for p in pts], crs="EPSG:4326").to_crs(metric_crs)
        buf = pts_wgs.buffer(alpha_m)
        merged = unary_union(list(buf.values))
        shell = merged.buffer(-alpha_m)
        geom = shell if not shell.is_empty else merged.convex_hull
        polys = []
        if geom.geom_type == "Polygon": polys=[geom]
        elif geom.geom_type == "MultiPolygon": polys=list(geom.geoms)
        kept = [g for g in polys if float(g.area) >= MIN_POLY_AREA_M2]
        if not kept: continue
        kept_wgs = gpd.GeoSeries(kept, crs=metric_crs).to_crs(epsg=4326).tolist()
        out[cid] = kept_wgs
    return out

# @profile
def severity_from_z(z):
    if z >= SEVERE_Z: return "severe"
    if z >= HIGH_Z:   return "high"
    if z >= ELEV_Z:   return "elev"
    return None

# @profile
def z_to_level_text(z):
    if z is None: return "n/a"
    if z >= 2.0: return "Very high (well above typical)"
    if z >= 1.0: return "High (above typical)"
    if z >= 0.5: return "Slightly elevated"
    if z > -0.5: return "Around typical"
    return "Below typical"

# @profile
def season_bands_today():
    """Return (pre_monsoon, monsoon, post_monsoon) windows as (start_iso,end_iso)."""
    y = date.today().year
    pre = (date(y,3,1),  date(y,5,31))
    mon = (date(y,6,1),  date(y,9,15))
    post= (date(y,9,16), date(y,11,30))
    # convert to string ISO
    return [tuple(map(str, w)) for w in (pre, mon, post)]

# ------------------ EARTH ENGINE IMAGES ------------------
# @profile
def lst_day_mean(aoi, start_iso, end_iso):
    coll = (ee.ImageCollection("MODIS/061/MOD11A2")
            .filterBounds(aoi).filterDate(start_iso, end_iso)
            .select("LST_Day_1km").map(lambda img: img.updateMask(img.gt(0))))
    lst_c = coll.mean().multiply(0.02).subtract(273.15).rename("lst_day_c").clip(aoi)
    return lst_c

# @profile
def lst_night_mean(aoi, start_iso, end_iso):
    coll = (ee.ImageCollection("MODIS/061/MOD11A2")
            .filterBounds(aoi).filterDate(start_iso, end_iso)
            .select("LST_Night_1km").map(lambda img: img.updateMask(img.gt(0))))
    lst_c = coll.mean().multiply(0.02).subtract(273.15).rename("lst_night_c").clip(aoi)
    return lst_c

# @profile
def lst_day_daily_collection(aoi, start_iso, end_iso):
    coll = (ee.ImageCollection("MODIS/061/MOD11A1")
            .filterBounds(aoi).filterDate(start_iso, end_iso)
            .select("LST_Day_1km").map(lambda img: img.updateMask(img.gt(0))))
    return coll

# @profile
def sentinel2_ndvi_recent(aoi, months_back=6):
    start = date.today() - timedelta(days=months_back*30)
    end   = date.today()
    # S2 SR (surface reflectance), simple cloud mask via QA60
    def mask_s2_sr(img):
        qa = img.select('QA60')
        cloud = qa.bitwiseAnd(1<<10).Or(qa.bitwiseAnd(1<<11))
        return img.updateMask(cloud.eq(0))
    coll = (ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(aoi).filterDate(str(start), str(end))
            .map(mask_s2_sr))
    med = coll.median()
    ndvi = med.normalizedDifference(['B8','B4']).rename('ndvi').clip(aoi)
    return ndvi

# @profile
def worldcover_map(year=2021):
    # ESA WorldCover v200 (2021/2020). Built-up class = 50, Tree cover = 10
    try:
        img = ee.Image("ESA/WorldCover/v200").select('Map')
        return img
    except Exception:
        try:
            return ee.Image("ESA/WorldCover/v100").select('Map')
        except Exception:
            return None

# @profile
def population_image(aoi):
    # Prefer WorldPop; fall back to GHSL/GPW if needed
    for yr in [2025, 2023, 2022, 2021, 2020, 2019]:
        try:
            col = (ee.ImageCollection("WorldPop/GP/100m/pop")
                   .filterBounds(aoi).filter(ee.Filter.eq('year', yr)))
            if col.size().getInfo() > 0:
                img = col.mosaic()
                bname = img.bandNames().getInfo()[0]
                return img.select(bname, ["pop"])
        except Exception:
            pass
    try:
        img = ee.Image("JRC/GHSL/P2019/POP_GLOBE_R2019A")
        bands = [b for b in img.bandNames().getInfo() if "2020" in b or "2015" in b]
        if bands: return img.select(bands[0], ["pop"])
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

# @profile
def worldpop_children_elderly(aoi):
    """Try to assemble children% and elderly% rasters (defensive). Returns (child_img, elder_img) as fractions 0..1 or (None,None)."""
    # WorldPop has age-sex layers by country; availability varies.
    candidates = [
        "WorldPop/GP/100m/pop_age_sex",  # generic
        "WorldPop/GP/100m/pop_age_sex_cons_unadj",
        "WorldPop/GP/100m/pop_age_sex_unadj"
    ]
    for ds in candidates:
        try:
            col = ee.ImageCollection(ds).filterBounds(aoi)
            if col.size().getInfo() == 0: continue
            # Heuristic: sum young (0-4, 5-9) and elderly (65+). Band names differ; try common patterns.
            first = col.first()
            bands = first.bandNames().getInfo()
            # guess bands
            young_bands = [b for b in bands if any(k in b.lower() for k in ["0","1-4","0-4","5-9"])]
            elder_bands = [b for b in bands if any(k in b.lower() for k in ["65","65-69","70","75","80","85"])]
            total = first.reduce(ee.Reducer.sum())
            young = first.select(young_bands).reduce(ee.Reducer.sum())
            elder = first.select(elder_bands).reduce(ee.Reducer.sum())
            child_frac = young.divide(total).rename("child_frac")
            elder_frac = elder.divide(total).rename("elder_frac")
            return child_frac, elder_frac
        except Exception:
            continue
    return None, None

# ------------------ REDUCERS ------------------
# @profile
def reduce_mean(image, geom, scale):
    try:
        val = image.reduceRegion(ee.Reducer.mean(), geom, scale=scale,
                                 maxPixels=1e13, bestEffort=True, tileScale=EE_TILE_SCALE)
        b = image.bandNames().getInfo()[0]
        v = val.get(b)
        return float(ee.Number(ee.Algorithms.If(v, v, None)).getInfo())
    except Exception:
        return None

# @profile
def reduce_sum(image, geom, scale):
    try:
        val = image.reduceRegion(ee.Reducer.sum(), geom, scale=scale,
                                 maxPixels=1e13, bestEffort=True, tileScale=EE_TILE_SCALE)
        v = val.get(image.bandNames().get(0))
        return float(ee.Number(ee.Algorithms.If(v, v, 0)).getInfo())
    except Exception:
        return None

# @profile
def fraction_of_mask(mask_img, geom, scale):
    """mask_img: 1 where class present, else 0. Returns fraction 0..1."""
    try:
        stats = mask_img.reduceRegion(ee.Reducer.mean(), geom, scale=scale,
                                      maxPixels=1e13, bestEffort=True, tileScale=EE_TILE_SCALE)
        v = stats.get(mask_img.bandNames().get(0))
        return float(ee.Number(ee.Algorithms.If(v, v, 0)).getInfo())
    except Exception:
        return None

# ------------------ OSM ------------------
# @profile
def osm_geoms_from_polygon(aoi_poly_wgs84, tags_dict):
    ox.settings.use_cache = True
    ox.settings.timeout = 180
    try:
        from osmnx.features import features_from_polygon as osm_features_from_polygon
    except Exception:
        try:
            from osmnx import geometries_from_polygon as osm_features_from_polygon
        except Exception:
            raise SystemExit("OSMnx missing polygon geometries. pip install --upgrade osmnx")
    layers = []
    for k, v in tags_dict.items():
        try:
            g = osm_features_from_polygon(aoi_poly_wgs84, tags={k: v})
            if g is not None and not g.empty:
                layers.append(g)
        except Exception:
            pass
    if not layers:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    base_crs = layers[0].crs or "EPSG:4326"
    all_feats = gpd.GeoDataFrame(gpd.pd.concat(layers, ignore_index=True), crs=base_crs)
    all_feats = all_feats[all_feats.geometry.notna()].copy()
    return all_feats.to_crs(epsg=4326)

# @profile
def count_sensitive_inside(sens_gdf, polygon):
    if sens_gdf is None or sens_gdf.empty:
        return dict(schools=0, clinics=0, hospitals=0, elder_homes=0)
    try:
        idx = sens_gdf.sindex
        sub = sens_gdf.iloc[list(idx.intersection(polygon.bounds))]
        inside = sub[sub.geometry.intersects(polygon)]
    except Exception:
        inside = sens_gdf[sens_gdf.geometry.intersects(polygon)]
    res = dict(schools=0, clinics=0, hospitals=0, elder_homes=0)
    if inside.empty:
        return res
    if "amenity" in inside.columns:
        res["schools"] = int((inside["amenity"]=="school").sum())
        res["clinics"] = int(((inside["amenity"]=="clinic") | (inside["amenity"]=="doctors")).sum())
        res["hospitals"] = int((inside["amenity"]=="hospital").sum())
    if "social_facility" in inside.columns:
        res["elder_homes"] = int(inside["social_facility"].isin(["nursing_home","assisted_living"]).sum())
    return res

# ------------------ HEAT INDEX ------------------
# @profile
def heat_index_c_from_t_rh(t_air_c, rh_pct):
    """NOAA HI formula (approx). Inputs: air T in °C, RH in %."""
    if (t_air_c is None) or (rh_pct is None): return None
    # Convert to F
    T = t_air_c * 9/5 + 32.0
    R = max(0.0, min(100.0, rh_pct))
    HI = (-42.379 + 2.04901523*T + 10.14333127*R
          - 0.22475541*T*R - 0.00683783*T*T - 0.05481717*R*R
          + 0.00122874*T*T*R + 0.00085282*T*R*R - 0.00000199*T*T*R*R)
    # adjustments (ignored for brevity)
    hi_c = (HI - 32.0) * 5/9
    return hi_c

# ------------------ MAP ------------------
# @profile
def build_map(aoi_bbox, hotspots, selected_cluster_polys, parameters):
    lat_c = (aoi_bbox[1] + aoi_bbox[3]) / 2.0
    lon_c = (aoi_bbox[0] + aoi_bbox[2]) / 2.0
    m = folium.Map(location=[lat_c, lon_c], zoom_start=12,
                   tiles="cartodbpositron", control_scale=True)

    # Cluster polygons (Top-3)
    for rank, (cid, poly) in enumerate(selected_cluster_polys, start=1):
        area_marking = folium.GeoJson(
            data=poly.__geo_interface__,
            name=f"Hot zone #{rank} (cluster {cid})",
            style_function=lambda _ : {"color": COLORS["envelope"], "weight": 3, "fillColor": COLORS["envelope"], "fillOpacity": 0.10},
            tooltip=f"Hot zone #{rank} (cluster {cid})"
        ).add_to(m)

        tags_html = f"""
            <div style="margin-bottom: 0.5em;">
            <span style="background:#007bff;color:white;padding:3px 7px;border-radius:5px;margin-right:5px;">
                Cluster ID: {cid}
            </span>
            <span style="background:#28a745;color:white;padding:3px 7px;border-radius:5px;">
                Rank: {rank}
            </span>
            </div>
            """
        
        button_html = """<center><button id="popup-btn"
                style="margin-top:10px; padding:6px 10px; border:none; border-radius:6px;
                        background:#2563eb; color:white; cursor:pointer; font-size:13px;">
            Generate AI Suggestions
        </button></center>

        <div id = "ai-answer"> </div>

        <script>
        (function() {
            var btn = document.getElementById("popup-btn");
            var desc = document.getElementById("popup-desc");
            var aiBox = document.getElementById("ai-answer");

            if (!btn || !desc) {console.log("Button or description not found"); return;}

            btn.addEventListener("click", async function() {
                btn.disabled = true;
                btn.innerText = "Generating...";
                try {
                    response = await fetch(
                        "https://nasa-space-app-web.onrender.com/llm-inference",
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
        })();
        </script>
        
        """


        parameters_html = '<div id = "popup-desc">'  + markdown.markdown(parameters[cid], extensions=['extra', 'toc', 'tables']) + "</div>"
        description = tags_html + parameters_html + button_html
        iframe = folium.IFrame(html=description, width=300, height=500)
        folium.Popup(iframe, max_width=700, max_height=600).add_to(area_marking)

        area_marking.add_to(m)

    # Hotspot markers only for selected clusters
    kept_cids = {cid for cid, _ in selected_cluster_polys}
    for hp in hotspots:
        if hp.get("_cid") not in kept_cids: continue
        sev = severity_from_z(hp["lst_z"])
        if sev is None: continue
        color = COLORS[sev]
        radius = 6 if sev == "elev" else (8 if sev == "high" else 10)
        folium.CircleMarker(
            location=(hp["lat"], hp["lon"]),
            radius=radius,
            color=color, fill=True, fill_color=color, fill_opacity=0.95,
            tooltip=f"{sev.upper()} UHI hotspot",
            popup=(f"<b>{sev.upper()} UHI hotspot</b><br>"
                   f"Surface temp (day): {hp['lst_c']:.1f} °C<br>"
                   f"Vs city typical: {z_to_level_text(hp['lst_z'])} (z≈{hp['lst_z']:.2f})")
        ).add_to(m)

    MiniMap(toggle_display=True, position="bottomright").add_to(m)
    Fullscreen().add_to(m)
    MousePosition(position="topright", separator=" | ", prefix="Lat/Lon:").add_to(m)
    MeasureControl(position="topright", primary_length_unit='kilometers').add_to(m)

    legend = f"""
    <div style="position: fixed; bottom: 18px; left: 18px; z-index:9999; background: white;
                padding: 10px 12px; border: 1px solid #ccc; border-radius: 6px; font-size: 13px;">
      <b>Urban Heat Island Hotspots</b> (last {DAYS_BACK} days)<br>
      <span style="display:inline-block;width:12px;height:12px;background:{COLORS['severe']};border:1px solid {COLORS['severe']};"></span>
      Severe (≥{SEVERE_Z}σ) &nbsp;
      <span style="display:inline-block;width:12px;height:12px;background:{COLORS['high']};border:1px solid {COLORS['high']};"></span>
      High ({HIGH_Z}–{SEVERE_Z}σ) &nbsp;
      <span style="display:inline-block;width:12px;height:12px;background:{COLORS['elev']};border:1px solid {COLORS['elev']};"></span>
      Elevated ({ELEV_Z}–{HIGH_Z}σ)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m

# ------------------ MAIN ------------------
# def run(session_id = None, ee_geometry = None, ee_bbox = None):
# @profile
def run(session_id, ee_geometry, ee_bbox):
    print("Initializing Earth Engine…")
    ee_init_headless()
    aoi = ee_geometry
    start_iso, end_iso = str(START), str(END)
    print(f"AOI: given | Window: {start_iso} → {end_iso}")

    # Day & night LST means
    print("Day & night LST Means...")
    lst_day_img   = lst_day_mean(aoi, start_iso, end_iso)
    lst_night_img = lst_night_mean(aoi, start_iso, end_iso)

    # Sample grid for clustering
    print("Making sample grid for clustering...")
    fc = lst_day_img.sample(region=aoi, scale=SCALE_M, geometries=True)
    feats = fc.limit(MAX_POINTS).getInfo().get("features", [])
    rows = []
    print("for f in feats")
    for f in feats:
        geom = f.get("geometry", {})
        if geom.get("type") != "Point": continue
        lon, lat = geom["coordinates"]
        v = f.get("properties", {}).get("lst_day_c", None)
        if v is None or not math.isfinite(v): continue
        rows.append({"lat": float(lat), "lon": float(lon), "lst_c": float(v)})

    print("Done... got", len(rows), "valid samples.")
    
    if not rows:
        raise SystemExit("No samples. Try increasing DAYS_BACK or MAX_POINTS.")

    # z-scores & pick hotspots
    print(" z-scores & pick hotspots...")
    lst_vals = [r["lst_c"] for r in rows]
    lst_z = zscores(lst_vals)
    pcts  = [p_rank(lst_z, v) for v in lst_z]
    hotspots = []
    for r, z, pr in zip(rows, lst_z, pcts):
        if (z >= Z_THRESHOLD) or (pr >= PCTL_THRESHOLD):
            hotspots.append({"lat": r["lat"], "lon": r["lon"], "lst_c": r["lst_c"], "lst_z": z, "percentile": pr})
    hotspots.sort(key=lambda x: x["lst_z"], reverse=True)

    # Cluster & envelopes
    print(" Cluster & envelopes...")
    clusters = ensure_clusters(hotspots)
    print(" Starting zip(hotspots, clusters) loop")
    for hp, cid in zip(hotspots, clusters):
        hp["_cid"] = cid
    print(" done")
    metric_crs = utm_crs_from_bbox(AOI_BBOX)
    envelopes_by_cid = build_concave_envelopes(hotspots, clusters, metric_crs, alpha_m=ALPHA_M, min_pts=MIN_ENVELOPE_POINTS)
    if not envelopes_by_cid:
        envelopes_by_cid = build_concave_envelopes(hotspots, [0]*len(hotspots), metric_crs, alpha_m=ALPHA_M, min_pts=3)

    # Union per cluster & areas
    print(" Union per cluster & areas...")
    cluster_union = {}
    cluster_area_km2 = {}
    for cid, polys in envelopes_by_cid.items():
        if not polys: continue
        polys_proj = gpd.GeoSeries(polys, crs="EPSG:4326").to_crs(metric_crs)
        union_geom = unary_union(list(polys_proj.values))
        area_km2 = float(union_geom.area / 1e6)
        if area_km2 <= 0: continue
        union_wgs = gpd.GeoSeries([union_geom], crs=metric_crs).to_crs(epsg=4326).iloc[0]
        cluster_union[cid] = union_wgs
        cluster_area_km2[cid] = area_km2

    # Top-3 by area
    print(" Top-3 by area...")
    top_cids = sorted(cluster_area_km2.keys(), key=lambda c: cluster_area_km2[c], reverse=True)[:3]
    selected = [(cid, cluster_union[cid]) for cid in top_cids]

    # ---- Ancillary datasets ----
    print(" Ancillary datasets…")
    wc = worldcover_map(year=2021)
    ndvi_img = sentinel2_ndvi_recent(aoi, months_back=6)
    pop_img  = population_image(aoi)
    child_img, elder_img = worldpop_children_elderly(aoi)  # may be None

    # ERA5 (air temp & dewpoint for heat index proxy)
    print(" ERA5 air temp & dewpoint…")
    try:
        era5 = ee.ImageCollection("ECMWF/ERA5/DAILY").filterDate(start_iso, end_iso).filterBounds(aoi)
        t2m = era5.select("mean_2m_air_temperature").mean().subtract(273.15).rename("t2m_c").clip(aoi)
        td2m = era5.select("mean_2m_dewpoint_temperature").mean().subtract(273.15).rename("td2m_c").clip(aoi)
    except Exception:
        print(" exception getting ERA5")
        t2m = None; td2m = None

    # Season windows
    print(" Season windows & pre-monsoon, monsoon, post-monsoon LST means…")
    (pre_s, pre_e), (mon_s, mon_e), (post_s, post_e) = season_bands_today()
    lst_pre  = lst_day_mean(aoi, pre_s,  pre_e)
    lst_mon  = lst_day_mean(aoi, mon_s,  mon_e)
    lst_post = lst_day_mean(aoi, post_s, post_e)

    # Extreme hot periods (use 8-day MOD11A2; count # composites above 90th pct over AOI)
    print(" Extreme hot periods (MODIS 8-day)…")
    coll_8d = (ee.ImageCollection("MODIS/061/MOD11A2")
                .filterBounds(aoi).filterDate(start_iso, end_iso)
                .select("LST_Day_1km").map(lambda img: img.updateMask(img.gt(0))))
    # 90th percentile over AOI
    print(" 90th percentile LST over AOI…")
    try:
        pct90 = coll_8d.reduce(ee.Reducer.percentile([90])).multiply(0.02).subtract(273.15).rename("p90")
    except Exception:
        pct90 = None

    # # OSM: buildings, sensitive sites, water (Ram culprit)
    # print(" OSM: buildings, sensitive sites, water…")
    # aoi_poly = aoi_polygon_wgs84()
    # print("aoi_poly done")
    # sys.stdout.flush()
    # try:
    #     print("here1")
    #     sys.stdout.flush()
    #     buildings = osm_geoms_from_polygon(aoi_poly, {"building": True})
    # except Exception:
    #     print("here2")
    #     sys.stdout.flush()
    #     buildings = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    # try:
    #     print("here3")
    #     sys.stdout.flush()
    #     sensitive = osm_geoms_from_polygon(aoi_poly, {"amenity": ["school","clinic","hospital","doctors"],
    #                                                     "social_facility": ["nursing_home","assisted_living"]})
    # except Exception:
    #     print("here4")
    #     sys.stdout.flush()
    #     sensitive = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    # try:
    #     print("here5")
    #     sys.stdout.flush()
    #     water = osm_geoms_from_polygon(aoi_poly, {"natural": ["water"], "waterway": ["river","canal"], "landuse": ["reservoir"]})
    # except Exception:
    #     print("here6")
    #     sys.stdout.flush()
    #     water = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Print summaries
    print(" Summarizing top clusters…")
    print("\n================= Top UHI clusters (area-wise) =================")
    print("(z (σ): standardized vs city typical; 0 ≈ typical)")

    parameters = dict()

    # Helpers for class fractions
    print(" Defining helpers for class fractions…")
    def frac_worldcover(mask_vals, poly):
        if wc is None: return None
        geom = ee.Geometry(poly.__geo_interface__)
        # allow list or int
        mask = None
        if isinstance(mask_vals, (list, tuple)):
            mask = wc.remap(mask_vals, [1]*len(mask_vals), 0).rename("m")
        else:
            mask = wc.eq(mask_vals).rename("m")
        return fraction_of_mask(mask, geom, scale=20)


    # Loop over selected clusters
    print("Looping over selected clusters…")
    for rank, (cid, poly) in enumerate(selected, start=1):
        print(f" \n--- Hot zone #{rank} (cluster {cid}) ---")
        # Geometry conversions
        poly_series = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(utm_crs_from_bbox(AOI_BBOX))
        area_km2 = float(poly_series.area.iloc[0] / 1e6)
        geom = ee.Geometry(poly.__geo_interface__)

        # Population & vulnerability
        pop_sum = reduce_sum(pop_img, geom, scale=100) if pop_img is not None else None
        child_pct = None; elder_pct = None
        if (child_img is not None) and (elder_img is not None):
            child_mean = reduce_mean(child_img, geom, scale=100)
            elder_mean = reduce_mean(elder_img, geom, scale=100)
            child_pct = None if child_mean is None else (100.0*child_mean)
            elder_pct = None if elder_mean is None else (100.0*elder_mean)

        # Impervious & canopy; NDVI mean
        imperv_pct = None
        tree_pct = None
        if wc is not None:
            imperv_pct = None if (frac_worldcover(50, poly) is None) else (100.0 * frac_worldcover(50, poly))
            tree_pct   = None if (frac_worldcover(10, poly) is None) else (100.0 * frac_worldcover(10, poly))
        ndvi_mean = reduce_mean(ndvi_img, geom, scale=20) if ndvi_img is not None else None

        # Buildings: roof area, cool-roof potential, height/density proxy
        large_roof_threshold_m2 = 500.0
        # b_in = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        # try:
        #     if buildings is not None and not buildings.empty:
        #         try:
        #             idx = buildings.sindex
        #             cand = buildings.iloc[list(idx.intersection(poly.bounds))]
        #             b_in = cand[cand.geometry.intersects(poly)].copy()
        #         except Exception:
        #             b_in = buildings[buildings.geometry.intersects(poly)].copy()
        # except Exception:
        #     pass
        roof_area_m2 = 0.0
        large_roofs_m2 = 0.0
        large_roofs_n  = 0
        mean_levels = None
        # if b_in is not None and not b_in.empty:
        #     b_proj = b_in.to_crs(utm_crs_from_bbox(AOI_BBOX))
        #     areas = b_proj.geometry.area.fillna(0.0)
        #     roof_area_m2 = float(areas.sum())
        #     large_mask = areas >= large_roof_threshold_m2
        #     large_roofs_m2 = float(areas[large_mask].sum())
        #     large_roofs_n  = int(large_mask.sum())
        #     # height/levels proxy
        #     lvl = []
        #     for _, r in b_in.iterrows():
        #         lv = r.get("building:levels") or r.get("levels") or r.get("height")
        #         try:
        #             if isinstance(lv, str) and "m" in lv: lv = lv.replace("m","").strip()
        #             lvf = float(lv)
        #             # if height in meters, roughly convert to levels (3 m/level)
        #             if lvf > 40: lvf = lvf / 3.0
        #             lvl.append(lvf)
        #         except Exception:
        #             continue
        #     if lvl: mean_levels = sum(lvl)/len(lvl)

        # Informal housing proxy: roof m² per person (lower → denser/informal)
        informal_proxy = None
        if (pop_sum is not None) and pop_sum > 0 and roof_area_m2 > 0:
            m2_per_person = roof_area_m2 / pop_sum
            informal_proxy = ("likely informal/very dense" if m2_per_person < 8.0
                                else "medium density" if m2_per_person < 20.0
                                else "lower density")
        # Sensitive sites inside
        # sens = count_sensitive_inside(sensitive, poly)

        # # Water distance
        # dist_water_m = None
        # if water is not None and not water.empty:
        #     try:
        #         w_proj = water.to_crs(utm_crs_from_bbox(AOI_BBOX))
        #         p_proj = poly_series.iloc[0]
        #         dist_water_m = float(w_proj.distance(p_proj).min())
        #     except Exception:
        #         dist_water_m = None

        # Day/Night means & delta
        day_c   = reduce_mean(lst_day_img,   geom, scale=1000)
        night_c = reduce_mean(lst_night_img, geom, scale=1000)
        dn_delta = None if (day_c is None or night_c is None) else (day_c - night_c)

        # Seasonality (pre/monsoon/post)
        pre_c  = reduce_mean(lst_pre,  geom, scale=1000)
        mon_c  = reduce_mean(lst_mon,  geom, scale=1000)
        post_c = reduce_mean(lst_post, geom, scale=1000)

        # Extreme hot periods (8-day composites above AOI 90th pct)
        extreme_cnt = None
        if pct90 is not None:
            try:
                # Count composites whose mean over poly > p90 over AOI
                # Build list of images in period
                imgs = coll_8d.toList(coll_8d.size())
                n = int(coll_8d.size().getInfo())
                cnt = 0
                thr_c = reduce_mean(pct90, ee.Geometry(aoi), scale=1000)
                for k in range(n):
                    im = ee.Image(imgs.get(k)).multiply(0.02).subtract(273.15).rename("c")
                    mval = reduce_mean(im, geom, scale=1000)
                    if (mval is not None) and (thr_c is not None) and (mval > thr_c):
                        cnt += 1
                extreme_cnt = cnt
            except Exception:
                extreme_cnt = None

        # Heat index proxy (ERA5)
        hi_c = None
        if (t2m is not None) and (td2m is not None):
            t_mean = reduce_mean(t2m, geom, scale=9000)
            td_mean= reduce_mean(td2m, geom, scale=9000)
            if (t_mean is not None) and (td_mean is not None):
                # RH from T & Td (Magnus)
                T = t_mean
                Td = td_mean
                es = 6.1094 * math.exp(17.625*Td/(243.04+Td))
                e  = 6.1094 * math.exp(17.625*T /(243.04+T))
                rh = max(0.0, min(100.0, 100.0*es/e)) if e > 0 else None
                hi_c = heat_index_c_from_t_rh(T, rh) if rh is not None else None

        description = ""

        # Print (markdown formatted)
        description += (f"\n### Hot zone #{rank} (cluster {cid})\n")
        description += (f"- **Area:** ~{area_km2:.2f} km²\n")
        description += (f"- **People living inside:** {(f'{int(pop_sum):,}' if pop_sum is not None else 'n/a')}\n")
        if (child_pct is not None) or (elder_pct is not None):
            ch = f"{child_pct:.1f}%" if child_pct is not None else "n/a"
            el = f"{elder_pct:.1f}%" if elder_pct is not None else "n/a"
            description += (f"- **Vulnerable groups:** children {ch}, elderly {el}\n")
        else:
            description += (f"- **Vulnerable groups:** n/a\n")
        if informal_proxy is not None:
            description += (f"- **Density / informal proxy:** {informal_proxy}\n")
        else:
            description += (f"- **Density / informal proxy:** n/a\n")

        imp = f"{imperv_pct:.1f}%" if imperv_pct is not None else "n/a"
        trp = f"{tree_pct:.1f}%" if tree_pct is not None else "n/a"
        ndv = f"{ndvi_mean:.2f}" if ndvi_mean is not None else "n/a"
        description += (f"- **Surfaces:** impervious {imp} | tree canopy {trp} | mean NDVI {ndv}\n")

        description += (f"- **Roof area total:** {int(roof_area_m2):,} m² | large roofs: {int(large_roofs_n)} bldgs / {int(large_roofs_m2):,} m² (cool-roof potential)\n")
        if mean_levels is not None:
            dens = (roof_area_m2 / (area_km2*1e6)) if area_km2>0 else None
            dens_txt = f"{dens*100:.1f}% footprint cover" if dens is not None else "n/a"
            description += (f"- **Building height/density:** mean levels ≈ {mean_levels:.1f} | {dens_txt}\n")
        else:
            description += (f"- **Building height/density:** n/a\n")

        # if dist_water_m is not None:
        #     description += (f"- **Nearest water:** ~{dist_water_m:.0f} m → *blue-corridor greening potential*\n")
        # else:
        #     description += (f"- **Nearest water:** n/a\n")

        day_txt = f"{day_c:.1f} °C" if day_c is not None else "n/a"
        night_txt = f"{night_c:.1f} °C" if night_c is not None else "n/a"
        dnd_txt = (f"{dn_delta:+.1f} °C (day − night)" if dn_delta is not None else "n/a")
        description += (f"- **Day vs Night:** day {day_txt} | night {night_txt} | Δ {dnd_txt}\n")

        seas_txt = []
        seas_txt.append(f"pre-monsoon {pre_c:.1f} °C" if pre_c is not None else "pre-monsoon n/a")
        seas_txt.append(f"monsoon {mon_c:.1f} °C" if mon_c is not None else "monsoon n/a")
        seas_txt.append(f"post-monsoon {post_c:.1f} °C" if post_c is not None else "post-monsoon n/a")
        description += ("- **Seasonality:** " + " | ".join(seas_txt) + "\n\n")

        if extreme_cnt is not None:
            description += (f"- **Extreme hot periods (8-day composites above local 90th pct):** {extreme_cnt}\n")
        else:
            description += (f"- **Extreme hot periods:** n/a\n")

        if hi_c is not None:
            description += (f"- **Heat index (air temp + humidity proxy):** ~{hi_c:.1f} °C\n")
        else:
            description += (f"- **Heat index:** n/a\n")

        # Sensitive sites
        # description += (f"- **Sensitive sites:** schools:{sens.get('schools',0)}, clinics:{sens.get('clinics',0)}, "
        #         f"hospitals:{sens.get('hospitals',0)}, elder homes:{sens.get('elder_homes',0)}\n")

        print(description)
        parameters[cid] = description

    # Map (Top-3 only)

    m = build_map(ee_bbox, hotspots, selected, parameters)
    output_path = 'web_outputs/{session_id}/uhi_hotspots.html'.format(session_id=session_id)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"\n✅ Saved UHI map to: {output_path}\nOpen in your browser to explore.")