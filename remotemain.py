from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import json
import os
import importlib.util
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from models.anlyzers import router
import requests
import dotenv

dotenv.load_dotenv()

app = FastAPI()

os.makedirs("web_outputs", exist_ok=True)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

# Pydantic models
class AnalysisRequest(BaseModel):
    upazila: Optional[str] = None
    district: Optional[str] = None
    analyses: List[str]
    session_id: str
    geojson: Optional[dict] = None

class LLM_Inference_Request(BaseModel):
    prompt: str
    systemPrompt: str | None = None
    type : str | None = None
    markdown : bool = True

class RemoteAnalysisResponse(BaseModel):
    session_id: str
    analysis_type: str
    success : bool


analysis_type_modules = {
    "aq_hotspots": "aq_hotspots.py",
    "uhi_hotspots": "uhi_hotspots.py",
    "green_access": "green_access_ndvi.py"
}

analysis_status: Dict[str, Dict] = {}

def push_event(session_id: str, message: dict):
    # Send JSON in body to avoid URL-length issues
    response = requests.post(
        f"{os.getenv('MAIN_SERVER_URL')}/xxd-push-event",
        json={
            "session_id": session_id,
            "message": message,
        },
        timeout=120,
    )
    if response.status_code != 200:
        print(f"Error pushing event: {response.text}. Status: {response.status_code}")

def update_main_analysis_status(session_id: str, status: Dict[str, Any]):
    response = requests.post(f"{os.getenv('MAIN_SERVER_URL')}/xxd-update-analysis-status", params={
        "session_id": session_id,
        "status": json.dumps(status)
    })
    if response.status_code != 200:
        print(f"Error updating analysis status: {response.text}. Status: {response.status_code}")

def send_files(session_id: str):
    search_dir = os.path.join("web_outputs", session_id)
    if not os.path.exists(search_dir):
        print(f"Directory {search_dir} does not exist.")
        return
    files = os.listdir(search_dir)
    files_to_send = []
    for file in files:
        file_path = os.path.join(search_dir, file)
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                files_to_send.append(('files', (file, f.read(), "text/html")))

    if files_to_send:
        response = requests.post(
            f"{os.getenv('MAIN_SERVER_URL')}/xxd-completed-analysis-from-remote-server",
            files=files_to_send,
            data={'session_id': session_id}
        )
        if response.status_code != 200:
            print(f"Error sending files: {response.text}. Status: {response.status_code}")
    else:
        print(f"No files found to send for session {session_id}")


def run_single_analysis(analysis_type: str, session_id: str, ee_geometry, aoi_bbox, gdf, geojson: Optional[dict] = None):
    """Run a single analysis in a separate thread"""
    global analysis_type_modules
    try:
        # Send starting message
        push_event(session_id, {
            "type": "analysis_start",
            "analysis": analysis_type,
            "message": f"Started {analysis_type.replace('_', ' ').title()} analysis..."
        })
        
        # Import and run the analysis module
        module_path = os.path.join("models", "anlyzers", analysis_type_modules[analysis_type])
        spec = importlib.util.spec_from_file_location(analysis_type, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Run and capture payload if module returns it
        payload: Optional[Any] = None
        if analysis_type == "green_access":
            payload = module.run(session_id, gdf, geoJson=geojson)
        else:
            payload = module.run(session_id, ee_geometry, aoi_bbox, geoJson=geojson)

        # If payload is a JSON string, parse to dict
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                # fallback: leave as string
                pass
        
        # Send completion message
        event = {
            "type": "analysis_complete",
            "analysis": analysis_type,
            "message": f"{analysis_type.replace('_', ' ').title()} analysis completed!",
        }
        if payload is not None:
            event["data"] = payload
        push_event(session_id, event)
        
        return True
        
    except Exception as e:
        # Send error message
        push_event(session_id, {
            "type": "analysis_error",
            "analysis": analysis_type,
            "message": f"Error in {analysis_type}: {str(e)}"
        })
        return False

def run_analyses_background(analyses: List[str], session_id: str, district: Optional[str], upazila: Optional[str] = None, geojson: Optional[dict] = None):
    """Run multiple analyses in sequence"""
    # Extracting location
    ee_geometry, aoi_bbox = (None, None)
    gdf = None
    if geojson is None:
        ee_geometry, aoi_bbox = router.get_polygon_and_bbox(district, upazila)
        gdf = router.get_gdf(district, upazila)
    try:
        completed_analyses = []
        total_analyses = len(analyses)
        
        for i, analysis in enumerate(analyses):
            # Update progress
            push_event(session_id, {
                "type": "progress_update",
                "current": i + 1,
                "total": total_analyses,
                "message": f"Running {analysis.replace('_', ' ').title()} ({i + 1}/{total_analyses})"
            })

            
            
            success = run_single_analysis(analysis, session_id, ee_geometry, aoi_bbox, gdf, geojson)
            if success:
                completed_analyses.append(analysis)
            
            # Add a small delay between analyses
            time.sleep(1)
        
        
        
        # Update global status
        analysis_status[session_id] = {
            "status": "completed",
            "completed_analyses": completed_analyses,
            "requested_analyses": analyses,
            "timestamp": datetime.now().isoformat()
        }

        update_main_analysis_status(session_id, analysis_status[session_id])

        # Optionally send files for legacy HTML viewers
        # send_files(session_id)

        # Send final completion message
        push_event(session_id, {
            "type": "all_analyses_complete",
            "completed_analyses": completed_analyses,
            "message": f"All analyses completed! {len(completed_analyses)}/{total_analyses} successful."
        })
        
    except Exception as e: 
        analysis_status[session_id] = {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

        update_main_analysis_status(session_id, analysis_status[session_id])
        # send_files(session_id)

        push_event(session_id, {
            "type": "error",
            "message": f"Error running analyses: {str(e)}"
        })

@app.post("/run-analysis")
async def run_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start analysis in background and return immediately"""
    if not request.analyses:
        raise HTTPException(status_code=400, detail="No analyses selected")
    os.makedirs(os.path.join("web_outputs", request.session_id), exist_ok=True)
    
    # Initialize analysis status
    analysis_status[request.session_id] = {
        "status": "running",
        "requested_analyses": request.analyses,
        "timestamp": datetime.now().isoformat()
    }
    
    # Start analyses in background
    background_tasks.add_task(run_analyses_background, request.analyses, request.session_id, request.district, request.upazila, request.geojson)
    
    return {
        "message": "Started",
        "session_id": request.session_id
    }


@app.get("/analysis-status/{session_id}")
async def get_analysis_status(session_id: str):
    """Get the status of analyses for a given session_id"""
    status = analysis_status.get(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session ID not found")
    return status