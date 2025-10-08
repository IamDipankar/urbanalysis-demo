from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import json
import os
import importlib.util
import time
from datetime import datetime
from typing import List, Dict
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
    upazila: str | None = None
    district: str
    analyses: List[str]
    session_id: str

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

def send_websocket_message(session_id: str, message: dict):
    response = requests.post(f"{os.getenv('MAIN_SERVER_URL')}/xxd-send-websocket-message-to-the-client", params={
        "session_id": session_id,
        "message": json.dumps(message)
    })
    if response.status_code != 200:
        print(f"Error sending websocket message: {response.text}")

def update_main_analysis_status(session_id: str, status: Dict):
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


def run_single_analysis(analysis_type: str, session_id: str, ee_geometry, aoi_bbox, gdf):
    """Run a single analysis in a separate thread"""
    global analysis_type_modules
    try:
        # Send starting message
        send_websocket_message(session_id, {
            "type": "analysis_start",
            "analysis": analysis_type,
            "message": f"Started {analysis_type.replace('_', ' ').title()} analysis..."
        })
        
        # Import and run the analysis module
        module_path = os.path.join("models", "anlyzers", analysis_type_modules[analysis_type])
        spec = importlib.util.spec_from_file_location(analysis_type, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if analysis_type == "green_access":
            module.run(session_id, gdf)
        # Run the main function
        else:
            module.run(session_id, ee_geometry, aoi_bbox)
        
        # Send completion message
        send_websocket_message(session_id, {
            "type": "analysis_complete",
            "analysis": analysis_type,
            "message": f"{analysis_type.replace('_', ' ').title()} analysis completed!"
        })
        
        return True
        
    except Exception as e:
        # Send error message
        send_websocket_message(session_id, {
            "type": "analysis_error",
            "analysis": analysis_type,
            "message": f"Error in {analysis_type}: {str(e)}"
        })
        return False

def run_analyses_background(analyses: List[str], session_id: str, district: str, upazila: str = None):
    """Run multiple analyses in sequence"""
    # Extracting location
    ee_geometry, aoi_bbox = router.get_polygon_and_bbox(district, upazila)
    gdf = router.get_gdf(district, upazila)
    try:
        completed_analyses = []
        total_analyses = len(analyses)
        
        for i, analysis in enumerate(analyses):
            # Update progress
            send_websocket_message(session_id, {
                "type": "progress_update",
                "current": i + 1,
                "total": total_analyses,
                "message": f"Running {analysis.replace('_', ' ').title()} ({i + 1}/{total_analyses})"
            })

            
            
            success = run_single_analysis(analysis, session_id, ee_geometry, aoi_bbox, gdf)
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

        send_files(session_id)

        # Send final completion message
        send_websocket_message(session_id, {
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

        send_files(session_id)

        send_websocket_message(session_id, {
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
    background_tasks.add_task(run_analyses_background, request.analyses, request.session_id, request.district, request.upazila)
    
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