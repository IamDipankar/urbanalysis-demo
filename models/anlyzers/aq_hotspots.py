# from . import router

# location_ee_geometry, location_bbox = router.get_polygon_and_bbox(district_name="Mymensingh", upazila_name="Haluaghat")
# location_gdf = router.get_gdf(district_name="Mymensingh", upazila_name="Haluaghat")


# narayanganj_aq_hotspots_readable.py
# Hotspots → Top-3 area-wise clusters with concave envelopes + summaries
# (population, plain-language AQ severity + seasonality, sensitive sites inside,
# named industries/point sources inside). Speed-optimized.

import markdown
import json
from models.llms import groq_api
# from memory_profiler import profile

import os, tempfile, base64
import math
from datetime import date, timedelta

import ee
import folium
from folium.plugins import MiniMap, Fullscreen, MousePosition, MeasureControl
from shapely.geometry import Point, MultiPoint, box
from shapely.ops import unary_union

import re
import dotenv

dotenv.load_dotenv()

# OSM / Geo deps
try:
    import osmnx as ox
    import geopandas as gpd
except Exception as e:
    raise SystemExit(
        f"Import error: {e}\nInstall: pip install osmnx geopandas rtree\n"
        "If NumPy 2.x issues: pip install 'numpy<2' && reinstall geopandas shapely pyproj fiona rtree"
    )

# ------------------ CONFIG ------------------
# AOI_BBOX = [90.32, 23.70, 90.52, 23.86]  # (W,S,E,N)

DAYS_BACK = 60
END = date.today()
START = END - timedelta(days=DAYS_BACK)

# EE sampling scale + limits (tuned for speed)
SCALE_M = 1200
MAX_POINTS = 3000
MAX_HOTSPOTS = 120
EE_TILE_SCALE = 4

# Weights for combined AQ index (z-space)
W_NO2 = 0.6
W_PM25 = 0.6
W_CO  = 0.3

# Hotspot selection
Z_THRESHOLD = 1.0
PCTL_THRESHOLD = 85.0

# Clustering
EPS_METERS = 1500.0
MIN_SAMPLES = 6

# Concave envelope controls
ALPHA_M = 1200            # “tightness” (m)
MIN_ENVELOPE_POINTS = 5   # min pts to build polygon
MIN_POLY_AREA_M2 = 2000   # drop tiny artifacts (~0.002 km²)

# Severity buckets
SEVERE_Z = 2.0
HIGH_Z   = 1.0
ELEV_Z   = 0.5

COLORS = {
    "severe": "#d32f2f",
    "high":   "#fb8c00",
    "elev":   "#ffd54f",
    "envelope": "#673ab7"
}

USER = os.getenv("USER") or os.getenv("USERNAME") or "user"
# OUT_HTML = f"/Users/{USER}/Downloads/narayanganj_aq_hotspots_readable.html"

# ------------------ EE INIT ------------------
def ee_init_headless():
    sa = os.environ["EE_SERVICE_ACCOUNT"]       # ee-runner@<project>.iam.gserviceaccount.com
    key_b64 = os.environ["EE_KEY_B64"]          # base64 of the JSON key

    # Write key to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(base64.b64decode(key_b64).decode("utf-8"))
        key_path = f.name

    creds = ee.ServiceAccountCredentials(sa, key_path)
    ee.Initialize(credentials=creds)

# ------------------ UTIL ------------------
def season_windows(today: date):
    y = today.year
    if today.month >= 4:
        dry_start = date(y - 1, 12, 1); dry_end = date(y, 3, 31)
    else:
        dry_start = date(y - 2, 12, 1); dry_end = date(y - 1, 3, 31)
    if (today.month, today.day) >= (9, 15):
        mon_start = date(y, 6, 1); mon_end = date(y, 9, 15)
    else:
        mon_start = date(y - 1, 6, 1); mon_end = date(y - 1, 9, 15)
    return dry_start, dry_end, mon_start, mon_end

def utm_crs_from_bbox(bbox):
    minx, miny, maxx, maxy = bbox
    lon_c = (minx + maxx) / 2.0
    lat_c = (miny + maxy) / 2.0
    zone = int((lon_c + 180) // 6) + 1
    epsg = 32600 + zone if lat_c >= 0 else 32700 + zone
    return f"EPSG:{epsg}"

# Plain-language helpers
def z_to_level_text(z):
    if z is None:
        return "n/a"
    if z >= 2.0: return "Very high (well above typical)"
    if z >= 1.0: return "High (above typical)"
    if z >= 0.5: return "Slightly elevated"
    if z > -0.5: return "Around typical"
    return "Below typical"

def seasonality_plain(zd, zm):
    """Return a readable sentence comparing monsoon vs dry."""
    if zd is None or zm is None:
        return "not enough data"
    diff = zm - zd  # monsoon minus dry
    if diff > 0.25:
        trend = "worse in the monsoon than the dry season"
    elif diff < -0.25:
        trend = "worse in the dry season than the monsoon"
    else:
        trend = "similar between seasons"
    return f"{trend} (monsoon {zm:.2f}, dry {zd:.2f} in standardized units; Δ={diff:+.2f})"

# ------------------ EE IMAGES ------------------
def build_mean_images(aoi, start_iso, end_iso):
    no2 = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
           .filterBounds(aoi).filterDate(start_iso, end_iso)
           .select("tropospheric_NO2_column_number_density")
           .mean().rename("no2").unmask(0)).clip(aoi)
    co = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CO")
          .filterBounds(aoi).filterDate(start_iso, end_iso)
          .select("CO_column_number_density")
          .mean().rename("co").unmask(0)).clip(aoi)
    aod = (ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
           .filterBounds(aoi).filterDate(start_iso, end_iso)
           .select("Optical_Depth_047")
           .mean().rename("aod").unmask(0)).clip(aoi)
    pm25 = aod.multiply(60.0).rename("pm25")
    return no2, pm25, co

def image_to_z(img, aoi, band_name):
    stats_mean = img.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=SCALE_M,
        maxPixels=1e13, bestEffort=True, tileScale=EE_TILE_SCALE
    )
    stats_std = img.reduceRegion(
        reducer=ee.Reducer.stdDev(), geometry=aoi, scale=SCALE_M,
        maxPixels=1e13, bestEffort=True, tileScale=EE_TILE_SCALE
    )
    mean_val = stats_mean.get(band_name)
    std_val  = stats_std.get(band_name)
    mean_num = ee.Number(ee.Algorithms.If(mean_val, mean_val, 0))
    std_num  = ee.Number(ee.Algorithms.If(std_val,  std_val,  1)).max(1e-6)
    return img.subtract(mean_num).divide(std_num).rename(f"{band_name}_z")

def combined_z_image(aoi, start_iso, end_iso):
    no2, pm25, co = build_mean_images(aoi, start_iso, end_iso)
    no2z = image_to_z(no2, aoi, "no2")
    pmz  = image_to_z(pm25, aoi, "pm25")
    coz  = image_to_z(co, aoi, "co")
    comb = (no2z.multiply(W_NO2).add(pmz.multiply(W_PM25)).add(coz.multiply(W_CO))).rename("aq_index_z")
    return no2z, pmz, coz, comb

# ------------------ POPULATION ------------------
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

def reduce_region_sum(image, region, scale):
    try:
        val = image.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=region, scale=scale,
            maxPixels=1e13, bestEffort=True, tileScale=EE_TILE_SCALE
        ).get("pop")
        return float(ee.Number(ee.Algorithms.If(val, val, 0)).getInfo())
    except Exception:
        return None

def ee_means_in_poly(img_dict, poly):
    geom = ee.Geometry(poly.__geo_interface__)
    bands = []
    for name, im in img_dict.items():
        bands.append(im.rename(name))
    stack = bands[0]
    for i in range(1, len(bands)):
        stack = stack.addBands(bands[i])
    try:
        vals = stack.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=SCALE_M,
            maxPixels=1e13, bestEffort=True, tileScale=EE_TILE_SCALE
        )
        out = {}
        for name in img_dict.keys():
            v = vals.get(name)
            out[name] = float(ee.Number(ee.Algorithms.If(v, v, 0)).getInfo())
        return out
    except Exception:
        return {name: None for name in img_dict.keys()}

# ------------------ SAMPLING & STATS ------------------
def sample_grid(aoi, img_stack, scale_m=SCALE_M, max_points=MAX_POINTS):
    fc = img_stack.sample(region=aoi, scale=scale_m, geometries=True)
    feats = fc.limit(max_points).getInfo().get("features", [])
    rows = []
    for f in feats:
        geom = f.get("geometry", {})
        if geom.get("type") != "Point": continue
        lon, lat = geom["coordinates"]
        p = f.get("properties", {})
        no2, pm25, co = p.get("no2"), p.get("pm25"), p.get("co")
        if None in (no2, pm25, co): continue
        rows.append({"lat": float(lat), "lon": float(lon),
                     "no2": float(no2), "pm25": float(pm25), "co": float(co)})
    return rows

def zscores(vals):
    good = [v for v in vals if v is not None and math.isfinite(v)]
    if len(good) < 2: return [0.0 for _ in vals]
    mean = sum(good)/len(good)
    var  = sum((v-mean)**2 for v in good)/len(good)
    std  = math.sqrt(max(var, 1e-12))
    return [0.0 if (v is None or not math.isfinite(v)) else (v-mean)/std for v in vals]

def p_rank(all_vals, v):
    s = sorted(all_vals)
    if not s: return 0.0
    cnt = sum(1 for x in s if x <= v)
    return 100.0 * cnt / len(s)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def cluster_dbscan(points, eps_m=EPS_METERS, min_samples=MIN_SAMPLES):
    n = len(points)
    if n == 0: return []
    buckets = {}
    for i, p in enumerate(points):
        key = (int(p["lat"]/0.01), int(p["lon"]/0.01))  # ~1.1 km cells
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

def ensure_clusters(hotspots):
    clusters = cluster_dbscan(hotspots, eps_m=EPS_METERS, min_samples=MIN_SAMPLES)
    if not any(c >= 0 for c in clusters):
        clusters = cluster_dbscan(hotspots, eps_m=EPS_METERS*1.6, min_samples=max(3, MIN_SAMPLES-2))
    if not any(c >= 0 for c in clusters):
        clusters = [0 for _ in hotspots]
    return clusters

# ------------------ OSM HELPERS ------------------
def aoi_polygon_wgs84(aoi_bbox):
    minx, miny, maxx, maxy = aoi_bbox
    return box(minx, miny, maxx, maxy)

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

def friendly_unnamed(row):
    """Produce a readable label when OSM has no name."""
    # Try a few informative tags to hint the type
    for key in ("industrial","power","man_made","landuse","waterway","harbour","amenity"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            if key == "power" and val == "plant":
                return "Unnamed facility (power plant)"
            if key == "industrial":
                return "Unnamed facility (industrial)"
            if key == "man_made":
                return f"Unnamed facility (man_made={val})"
            if key == "waterway":
                return f"Unnamed riverside/port feature ({val})"
            if key == "harbour":
                return "Unnamed harbour/jetty"
            if key == "landuse" and val == "industrial":
                return "Unnamed facility (industrial landuse)"
            if key == "amenity":
                return f"Unnamed amenity ({val})"
            return f"Unnamed facility ({key}={val})"
    return "Unnamed facility (OSM)"

def list_osm_names_in_poly(gdf, polygon, max_show=40):
    if gdf is None or gdf.empty:
        return [], 0
    try:
        idx = gdf.sindex
        cand = gdf.iloc[list(idx.intersection(polygon.bounds))]
        inside = cand[cand.geometry.intersects(polygon)]
    except Exception:
        inside = gdf[gdf.geometry.intersects(polygon)]
    count = len(inside)
    if count == 0:
        return [], 0
    name_cols = [c for c in ["name", "name:en", "operator", "brand"] if c in inside.columns]
    names = []
    for _, r in inside.iterrows():
        nm = None
        for c in name_cols:
            val = r.get(c)
            if isinstance(val, str) and val.strip():
                nm = val.strip(); break
        if nm is None:
            nm = friendly_unnamed(r)
        names.append(nm)
    names = sorted(set(names))
    if len(names) > max_show:
        names = names[:max_show] + [f"... (+{len(set(names))-max_show} more)"]
    return names, count

# ------------------ ENVELOPES (concave polygons) ------------------
def build_concave_envelopes(hotspots, clusters, metric_crs, alpha_m=ALPHA_M, min_pts=MIN_ENVELOPE_POINTS):
    """Returns dict cid -> list[Polygon in WGS84], with tiny/degenerate parts removed."""
    by_cluster = {}
    for hp, cid in zip(hotspots, clusters):
        if cid < 0: continue
        by_cluster.setdefault(cid, []).append(hp)

    out = {}
    for cid, pts in by_cluster.items():
        if len(pts) < min_pts:
            continue
        pts_wgs = gpd.GeoSeries([Point(p["lon"], p["lat"]) for p in pts], crs="EPSG:4326").to_crs(metric_crs)
        buf = pts_wgs.buffer(alpha_m)
        merged = unary_union(list(buf.values))
        shell = merged.buffer(-alpha_m)
        geom = shell if not shell.is_empty else merged.convex_hull
        polys = []
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        kept = [g for g in polys if float(g.area) >= MIN_POLY_AREA_M2]
        if not kept:
            continue
        kept_wgs = gpd.GeoSeries(kept, crs=metric_crs).to_crs(epsg=4326).tolist()
        out[cid] = kept_wgs
    return out

# ------------------ MAP ------------------

def build_map(aoi_bbox, hotspots, selected_cluster_polys, parameters):
    lat_c = (aoi_bbox[1] + aoi_bbox[3]) / 2.0
    lon_c = (aoi_bbox[0] + aoi_bbox[2]) / 2.0
    m = folium.Map(location=[lat_c, lon_c], zoom_start=12,
                   tiles="cartodbpositron", control_scale=True)

    # Cluster polygons (Top-3)
    for rank, (cid, poly) in enumerate(selected_cluster_polys, start=1):
        area_marking = folium.GeoJson(
            data=poly.__geo_interface__,
            name=f"Top cluster #{rank} (cluster {cid})",
            style_function=lambda _ : {"color": COLORS["envelope"], "weight": 3, "fillColor": COLORS["envelope"], "fillOpacity": 0.10},
            tooltip=f"Top cluster #{rank} (cluster {cid})",
            popup=f"Test popup for cluster {cid}"
        )

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
        
        button_html = """<center><button id="popup-btn_<<cid>>"
                style="margin-top:10px; padding:6px 10px; border:none; border-radius:6px;
                        background:#2563eb; color:white; cursor:pointer; font-size:13px;"
                onclick="getAiSuggestions(<<cid>>);">
            Generate AI Suggestions
        </button></center>

        <div id = "ai-answer_<<cid>>"><<great_blank>></div>

        """.replace("<<cid>>", str(cid)).replace("<<great_blank>>", "</br>"*50)


        parameters_html = f'<div id = "popup-desc_{str(cid)}">'  + markdown.markdown(parameters[cid], extensions=['extra', 'toc', 'tables']) + "</div>"
        description = tags_html + parameters_html + button_html
        # iframe = folium.IFrame(html=description, width=300, height=500)
        folium.Popup(description, max_width=700, max_height=600).add_to(area_marking)

        area_marking.add_to(m)

    # Hotspots belonging to selected clusters only
    kept_cids = {cid for cid, _ in selected_cluster_polys}
    for hp in hotspots:
        cid = hp.get("_cid")
        if cid not in kept_cids:
            continue
        sev = 'severe' if hp["aq_index_z"] >= SEVERE_Z else ('high' if hp["aq_index_z"] >= HIGH_Z else ('elev' if hp["aq_index_z"] >= ELEV_Z else None))
        if sev is None: 
            continue
        color = COLORS[sev]
        radius = 6 if sev == "elev" else (8 if sev == "high" else 10)
        z_dry = hp.get("aq_z_dry"); z_mon = hp.get("aq_z_monsoon")
        season_txt = seasonality_plain(z_dry, z_mon)
        hint = hp.get("driver_hint","Mixed drivers")
        popup_html = (
            f"<b>{sev.upper()} hotspot</b><br>"
            f"Current level: {z_to_level_text(hp['aq_index_z'])} (z≈{hp['aq_index_z']:.2f}; 0≈typical)<br>"
            f"NO₂ z: {hp['no2_z']:.2f} | PM₂.₅ z: {hp['pm25_z']:.2f} | CO z: {hp['co_z']:.2f}<br>"
            f"<b>Likely driver:</b> {hint}<br>"
            f"<b>Seasonality:</b> {season_txt}"
        )
        folium.CircleMarker(
            location=(hp["lat"], hp["lon"]),
            radius=radius,
            color=color, fill=True, fill_color=color, fill_opacity=0.95,
            tooltip=f"{sev.upper()} hotspot", popup=popup_html
        ).add_to(m)

    MiniMap(toggle_display=True, position="bottomright").add_to(m)
    Fullscreen().add_to(m)
    MousePosition(position="topright", separator=" | ", prefix="Lat/Lon:").add_to(m)
    MeasureControl(position="topright", primary_length_unit='kilometers').add_to(m)

    legend = """
    <div style="position: fixed; bottom: 18px; left: 18px; z-index:9999; background: white;
                padding: 10px 12px; border: 1px solid #ccc; border-radius: 6px; font-size: 13px;">
      <b>Top hotspot clusters (area-wise)</b><br>
      <span style="display:inline-block;width:12px;height:12px;background:#d32f2f;border:1px solid #d32f2f;"></span>
      Severe (≥ 2σ above city typical)<br>
      <span style="display:inline-block;width:12px;height:12px;background:#fb8c00;border:1px solid #fb8c00;"></span>
      High (1–2σ)<br>
      <span style="display:inline-block;width:12px;height:12px;background:#ffd54f;border:1px solid #ffd54f;"></span>
      Elevated (0.5–1σ)<br>
      <span style="display:inline-block;width:12px;height:12px;background:#673ab7;border:1px solid #673ab7;"></span>
      Cluster polygon(s)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

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
                                type: "aq"
                            })
                        }
                    );

                    if (!response.ok) {
                        console.log("Error in response");
                        aiBox.innerHTML = "<hr><b>AI Suggestions:</b><br>" + "Error during request. Status: " + response.status;
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
    return m

# ------------------ MAIN ------------------
# @profile
def run(session_id=None, ee_geometry=None, aoi_bbox=None, geoJson = None):
    print("Initializing Earth Engine…2")
    ee_init_headless()
    if geoJson:
        feature = geoJson if geoJson.get('type') == 'Feature' else geoJson.get('features', [])[0]
        if not feature:
            raise ValueError("Invalid GeoJSON provided.")
        ee_geometry = ee.Geometry(feature.get('geometry'))
        aoi_bbox = ee_geometry.bounds().getInfo().get('coordinates', None)
        if aoi_bbox:
            aoi_bbox = aoi_bbox[0]
            longs, lats = zip(*aoi_bbox)
            aoi_bbox = [min(longs), min(lats), max(longs), max(lats)]

        print(f"BBOX from GeoJSON: {aoi_bbox}")

    print("Started aq hotspot analysis… Session:", session_id)

    aoi = ee_geometry
    start_iso, end_iso = str(START), str(END)
    print(f"Geometry: Given | Window: {start_iso} → {end_iso}")

    # Current-window images & z stacks
    no2_img, pm25_img, co_img = build_mean_images(aoi, start_iso, end_iso)
    stack = no2_img.addBands(pm25_img).addBands(co_img)
    no2z_now, pmz_now, coz_now, aqz_now_img = combined_z_image(aoi, start_iso, end_iso)

    # Sample grid (capped)
    rows = sample_grid(aoi, stack, scale_m=SCALE_M, max_points=MAX_POINTS)
    if not rows:
        raise SystemExit("No samples. Try expanding AOI or increasing DAYS_BACK.")

    # Z-scores (current window)
    no2_z = zscores([r["no2"] for r in rows])
    pm25_z = zscores([r["pm25"] for r in rows])
    co_z   = zscores([r["co"] for r in rows])
    aq_raw = [W_NO2*n + W_PM25*p + W_CO*c for n, p, c in zip(no2_z, pm25_z, co_z)]
    aq_index_z = zscores(aq_raw)

    # Hotspot selection (cap)
    def prc(vs, v): return p_rank(vs, v)
    pcts = [prc(aq_index_z, v) for v in aq_index_z]
    candidates = []
    for r, nz, pz, cz, az, pr in zip(rows, no2_z, pm25_z, co_z, aq_index_z, pcts):
        if (az >= Z_THRESHOLD) or (pr >= PCTL_THRESHOLD):
            if (nz >= 1.0) and (cz >= 1.0):
                driver = "Traffic / combustion (high NO₂ + CO)"
            elif (pz >= 1.0) and (nz < 0.5):
                driver = "Dust / construction / open burning (high PM proxy, low NO₂)"
            elif (nz >= 1.0) and (cz < 0.5):
                driver = "Point sources / industry (high NO₂, low CO)"
            else:
                driver = "Mixed drivers"
            candidates.append({
                "lat": r["lat"], "lon": r["lon"],
                "no2_z": nz, "pm25_z": pz, "co_z": cz,
                "aq_index_z": az, "percentile": pr,
                "driver_hint": driver
            })
    if not candidates:
        raise SystemExit("No hotspots met the threshold; relax Z_THRESHOLD/PCTL_THRESHOLD.")

    candidates.sort(key=lambda x: x["aq_index_z"], reverse=True)
    hotspots = candidates[:MAX_HOTSPOTS]

    # Cluster (with fallback)
    clusters = ensure_clusters(hotspots)
    for hp, cid in zip(hotspots, clusters):
        hp["_cid"] = cid

    # Seasonality quick-look (batched)
    dry_start, dry_end, mon_start, mon_end = season_windows(date.today())
    _, _, _, aqz_dry_img = combined_z_image(aoi, str(dry_start), str(dry_end))
    _, _, _, aqz_mon_img = combined_z_image(aoi, str(mon_start), str(mon_end))

    def fc_from_points(hps):
        feats = [ee.Feature(ee.Geometry.Point([hp["lon"], hp["lat"]]), {"idx": i})
                 for i, hp in enumerate(hps)]
        return ee.FeatureCollection(feats)

    pts_fc = fc_from_points(hotspots)
    aq_dry_coll = aqz_dry_img.sampleRegions(collection=pts_fc, scale=SCALE_M, geometries=False, tileScale=EE_TILE_SCALE)
    aq_mon_coll = aqz_mon_img.sampleRegions(collection=pts_fc, scale=SCALE_M, geometries=False, tileScale=EE_TILE_SCALE)
    aq_dry = aq_dry_coll.getInfo().get("features", [])
    aq_mon = aq_mon_coll.getInfo().get("features", [])
    for feat in aq_dry:
        i = int(feat["properties"]["idx"])
        val = feat["properties"].get("aq_index_z")
        hotspots[i]["aq_z_dry"] = float(val) if val is not None else None
    for feat in aq_mon:
        i = int(feat["properties"]["idx"])
        val = feat["properties"].get("aq_index_z")
        hotspots[i]["aq_z_monsoon"] = float(val) if val is not None else None

    # OSM context
    aoi_poly = aoi_polygon_wgs84(aoi_bbox)
    try:
        sens_all = osm_geoms_from_polygon(aoi_poly, {"amenity": ["school","clinic","hospital","doctors"],
                                                     "social_facility": ["nursing_home","assisted_living"]})
    except Exception:
        sens_all = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    try:
        ind_all  = osm_geoms_from_polygon(aoi_poly, {
            "landuse": ["industrial"], "industrial": True, "man_made": ["works","chimney"],
            "power": ["plant","generator"], "harbour": ["yes"], "waterway": ["dock"]
        })
        extra_port = osm_geoms_from_polygon(aoi_poly, {"man_made": ["pier"], "landuse": ["port"],
                                                       "harbour": ["yes"], "waterway": ["dock"]})
        if ind_all is not None and not ind_all.empty and extra_port is not None and not extra_port.empty:
            ind_all = gpd.GeoDataFrame(gpd.pd.concat([ind_all, extra_port], ignore_index=True), crs=ind_all.crs or "EPSG:4326")
        elif (ind_all is None or ind_all.empty) and extra_port is not None and not extra_port.empty:
            ind_all = extra_port
    except Exception:
        ind_all = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Build concave envelopes per cluster (filter tiny/degenerate)
    metric_crs = utm_crs_from_bbox(aoi_bbox)
    envelopes_by_cid = build_concave_envelopes(hotspots, clusters, metric_crs, alpha_m=ALPHA_M, min_pts=MIN_ENVELOPE_POINTS)
    if not envelopes_by_cid:
        envelopes_by_cid = build_concave_envelopes(hotspots, [0]*len(hotspots), metric_crs, alpha_m=ALPHA_M, min_pts=3)

    # Union to single polygon per cluster & compute area
    cluster_union = {}
    cluster_area_km2 = {}
    for cid, polys in envelopes_by_cid.items():
        if not polys: 
            continue
        polys_proj = gpd.GeoSeries(polys, crs="EPSG:4326").to_crs(metric_crs)
        union_geom = unary_union(list(polys_proj.values))
        area_km2 = float(union_geom.area / 1e6)
        if area_km2 <= 0:
            continue
        union_wgs = gpd.GeoSeries([union_geom], crs=metric_crs).to_crs(epsg=4326).iloc[0]
        cluster_union[cid] = union_wgs
        cluster_area_km2[cid] = area_km2

    # Pick Top-3 clusters by area
    top_cids = sorted(cluster_area_km2.keys(), key=lambda c: cluster_area_km2[c], reverse=True)[:3]
    selected = [(cid, cluster_union[cid]) for cid in top_cids]

    # ---------- Print summaries for Top-3 (plain language) ----------
    description_string = ""

    description_string += ("\n================= Top bad air quality hotspot clusters (area-wise) =================\n")
    description_string += ("(Note: z (σ) = standardized units; 0 means ‘typical’ for the city in the last 60 days.)\n")
    pop_img = population_image(aoi)
    z_imgs_now = {"no2_z": no2z_now, "pm25_z": pmz_now, "co_z": coz_now, "aq_index_z": aqz_now_img}

    cid_wise_descriptions = dict()

    for rank, (cid, poly) in enumerate(selected, start=1):
        area_km2 = cluster_area_km2[cid]
        pop_sum = reduce_region_sum(pop_img, ee.Geometry(poly.__geo_interface__), scale=200) if pop_img is not None else None
        means_now = ee_means_in_poly(z_imgs_now, poly)
        zn = means_now.get("aq_index_z")
        zd = ee_means_in_poly({"aq_index_z": aqz_dry_img}, poly).get("aq_index_z")
        zm = ee_means_in_poly({"aq_index_z": aqz_mon_img}, poly).get("aq_index_z")
        sens_inside = count_sensitive_inside(sens_all, poly)
        ind_names, ind_count = list_osm_names_in_poly(ind_all, poly, max_show=60)

        curr_desc = ""

        curr_desc += (f"\n### Top cluster #{rank} (cluster {cid})\n")
        curr_desc += (f"- **Area:** ~{area_km2:.2f} km²\n")
        curr_desc += (f"- **People living inside:** {(f'{int(pop_sum):,}' if pop_sum is not None else 'n/a')}\n")
        if zn is not None:
            curr_desc += (f"- **Current level:** {z_to_level_text(zn)} (z≈{zn:.2f}; 0≈typical)\n")
        else:
            curr_desc += ("- **Current level:** n/a\n")
        curr_desc += (f"- **Seasonality:** {seasonality_plain(zd, zm)}\n")
        curr_desc += (f"- **Sensitive sites inside:** schools:{sens_inside.get('schools',0)}, "
              f"clinics:{sens_inside.get('clinics',0)}, hospitals:{sens_inside.get('hospitals',0)}, "
              f"elder homes:{sens_inside.get('elder_homes',0)}\n")
        curr_desc += (f"- **Industrial/port/point-source features inside:** {ind_count}\n")
        if ind_count > 0:
            curr_desc += ("  **Names / tags:**\n")
            for nm in ind_names:
                curr_desc += (f"   - {nm}\n")

        description_string += curr_desc
        cid_wise_descriptions[cid] = curr_desc

    print(description_string)
    # # result = groq_api.inference(description_string)
    # print()
    # print()
    # print("LLM Response")
    # print(result)
    # print('\n\n\n')
    # # result = groq_api.parse_llm_response(result[0])
    # print(result)


    # ---------- Map with Top-3 only ----------
    m = build_map(aoi_bbox, hotspots, selected, cid_wise_descriptions)
    html_output_dir = f'web_outputs/{session_id}/aq_hotspots.html' if session_id else 'web_outputs/temp/aq_hotspots.html'
    os.makedirs(os.path.dirname(html_output_dir), exist_ok=True)
    m.save(html_output_dir)
    print(f"\n✅ Saved: {html_output_dir}")

    # ---------- Build client payload for Leaflet rendering ----------
    try:
        bbox = aoi_bbox
        lon_c = (bbox[0] + bbox[2]) / 2.0
        lat_c = (bbox[1] + bbox[3]) / 2.0

        # Cluster polygons
        polygon_features = []
        for rank, (cid, poly) in enumerate(selected, start=1):
            desc_md = cid_wise_descriptions.get(cid, "")
            desc_html = markdown.markdown(desc_md, extensions=['extra', 'toc', 'tables']) if desc_md else ""
            popup_html = f"<div id=\"popup-desc_{cid}\">{desc_html}</div>"
            feature = {
                "type": "Feature",
                "properties": {
                    "cid": cid,
                    "rank": rank,
                    "name": f"Hot zone #{rank} (cluster {cid})",
                    "style": {
                        "color": "#673ab7",
                        "weight": 3,
                        "fillColor": "#673ab7",
                        "fillOpacity": 0.10,
                    },
                    "tooltip": f"Hot zone #{rank} (cluster {cid})",
                    "popup_html": popup_html,
                },
                "geometry": poly.__geo_interface__,
            }
            polygon_features.append(feature)
        polygons_fc = {"type": "FeatureCollection", "features": polygon_features}

        # Hotspot points
        point_features = []
        kept_cids = {cid for cid, _ in selected}
        for hp in hotspots:
            cid = hp.get("_cid")
            if cid not in kept_cids:
                continue
            tooltip_text = "AQ hotspot"
            popup_html = (
                f"<b>AQ hotspot</b><br>Combined z: {hp.get('aq_index_z', 0):.2f}<br>"
                f"NO2 z: {hp.get('no2_z', 0):.2f} | PM25 z: {hp.get('pm25_z', 0):.2f} | CO z: {hp.get('co_z', 0):.2f}"
            )
            feature = {
                "type": "Feature",
                "properties": {
                    "cid": cid,
                    "marker": {
                        "type": "circle",
                        "radius": 7,
                        "color": "#d32f2f",
                        "fill": True,
                        "fillColor": "#d32f2f",
                        "fillOpacity": 0.9,
                    },
                    "tooltip": tooltip_text,
                    "popup_html": popup_html,
                },
                "geometry": {"type": "Point", "coordinates": [hp["lon"], hp["lat"]]},
            }
            point_features.append(feature)
        hotspots_fc = {"type": "FeatureCollection", "features": point_features}

        payload = {
            "meta": {"center": [lat_c, lon_c], "zoom": 12, "tiles": "cartodbpositron"},
            "layers": {"clusters": polygons_fc, "hotspots": hotspots_fc},
            "legend": {
                "title": "Air Quality Hotspots (last 60 days)",
                "colors": {"envelope": "#673ab7", "hotspot": "#d32f2f"},
            },
        }

        # Save JSON alongside HTML for debugging/inspection
        try:
            json_output_dir = f'web_outputs/{session_id}/aq_hotspots.json' if session_id else 'web_outputs/temp/aq_hotspots.json'
            with open(json_output_dir, 'w', encoding='utf-8') as jf:
                json.dump(payload, jf, ensure_ascii=False)
        except Exception:
            pass

        return payload
    except Exception as _:
        # If anything fails during payload generation, just return None to keep compatibility
        return None

# if __name__ == "__main__":
#     run("aaaa", location_ee_geometry, location_bbox)
