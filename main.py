from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks,status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import threading
import json
import os, base64, tempfile
import sys
import importlib.util
import time
from datetime import datetime
from typing import List, Dict, Optional
import uuid
from models.anlyzers import router
from models.llms import groq_api
import markdown
from markdownify import markdownify as md
from fastapi.middleware.cors import CORSMiddleware
import random


checker = 0
signature = random.random() * 1000000
tokens_used = 0

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'anlyzers'))

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/statics", StaticFiles(directory="statics"), name="static")
app.mount("/web_outputs", StaticFiles(directory="web_outputs"), name="web_outputs")

# Store active connections and analysis status
active_connections: Dict[str, WebSocket] = {}
analysis_status: Dict[str, Dict] = {}

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

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                print(f"Error sending message to {session_id}: {e}")
                self.disconnect(session_id)

manager = ConnectionManager()

analysis_type_modules = {
    "aq_hotspots": "aq_hotspots.py",
    "uhi_hotspots": "uhi_hotspots.py",
    "green_access": "green_access_ndvi.py"
}

def run_single_analysis(analysis_type: str, session_id: str, ee_geometry, aoi_bbox, gdf):
    """Run a single analysis in a separate thread"""
    global analysis_type_modules
    try:
        # Send starting message
        asyncio.run(manager.send_message(session_id, {
            "type": "analysis_start",
            "analysis": analysis_type,
            "message": f"Starting {analysis_type.replace('_', ' ').title()} analysis..."
        }))
        
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
        asyncio.run(manager.send_message(session_id, {
            "type": "analysis_complete",
            "analysis": analysis_type,
            "message": f"{analysis_type.replace('_', ' ').title()} analysis completed!"
        }))
        
        return True
        
    except Exception as e:
        # Send error message
        asyncio.run(manager.send_message(session_id, {
            "type": "analysis_error",
            "analysis": analysis_type,
            "message": f"Error in {analysis_type}: {str(e)}"
        }))
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
            asyncio.run(manager.send_message(session_id, {
                "type": "progress_update",
                "current": i + 1,
                "total": total_analyses,
                "message": f"Running {analysis.replace('_', ' ').title()} ({i + 1}/{total_analyses})"
            }))
            
            success = run_single_analysis(analysis, session_id, ee_geometry, aoi_bbox, gdf)
            if success:
                completed_analyses.append(analysis)
            
            # Add a small delay between analyses
            time.sleep(1)
        
        # Send final completion message
        asyncio.run(manager.send_message(session_id, {
            "type": "all_analyses_complete",
            "completed_analyses": completed_analyses,
            "message": f"All analyses completed! {len(completed_analyses)}/{total_analyses} successful."
        }))
        
        # Update global status
        analysis_status[session_id] = {
            "status": "completed",
            "completed_analyses": completed_analyses,
            "requested_analyses": analyses,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        asyncio.run(manager.send_message(session_id, {
            "type": "error",
            "message": f"Error running analyses: {str(e)}"
        }))
        analysis_status[session_id] = {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/")
async def read_root():
    return FileResponse("statics/index.html")

async def continue_websocket(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back any messages (for debugging)
            await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        print(f"WebSocket connection closed for {session_id}: {e}")
    finally:
        manager.disconnect(session_id)

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    return await continue_websocket(websocket, session_id)

@app.websocket("/ws-reconnect/{session_id}")
async def websocket_reconnect(websocket: WebSocket, session_id: str):
    print("Its here")
    if session_id not in analysis_status:
        print("Now its here")
        await websocket.accept()
        await websocket.close(code=status.WS_1012_SERVICE_RESTART, reason="Your session ID is invalid or expired.")
        print("here3")
        return
    if analysis_status[session_id]["status"] == "completed":
        print("here4")
        await websocket.accept()
        total_analyses = len(analysis_status[session_id].get("requested_analyses", []))
        completed_analyses = analysis_status[session_id].get("completed_analyses", [])
        await websocket.send_text(json.dumps( {
            "type": "all_analyses_complete",
            "completed_analyses": completed_analyses,
            "message": f"All analyses completed! {len(completed_analyses)}/{total_analyses} successful."
        }))
        await asyncio.sleep(10)  # Keep connection alive for a short while
        print("here5")
        return
    print("here6")
    return await continue_websocket(websocket, session_id)
        

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

    await asyncio.sleep(1)
    
    # Start analyses in background
    background_tasks.add_task(run_analyses_background, request.analyses, request.session_id, request.district, request.upazila)
    
    return {
        "message": "Analysis started successfully",
        "session_id": request.session_id,
        "analyses": request.analyses
    }

@app.get("/analysis-status/{session_id}")
async def get_analysis_status(session_id: str):
    """Get current status of analyses"""
    if session_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis session not found")
    
    return analysis_status[session_id]

@app.get("/results/{session_id}/{analysis_type}")
async def get_results(session_id: str, analysis_type: str):
    """Serve analysis results HTML file"""
    # Map analysis types to their HTML files
    html_files = {
        "aq_hotspots": "aq_hotspots.html",
        "uhi_hotspots": "uhi_hotspots.html", 
        "green_access": "green_access.html"
    }
    
    if analysis_type not in html_files:
        raise HTTPException(status_code=404, detail="Analysis type not found")
    
    html_file = html_files[analysis_type]
    file_path = os.path.join("web_outputs", session_id, html_file)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Results file not found. Analysis may not be completed yet.")
    
    return FileResponse(file_path, media_type='text/html')

@app.get("/results-viewer")
async def results_viewer():
    """Serve the results viewer page"""
    return FileResponse("statics/results.html")

@app.get("/available-results/{session_id}")
async def get_available_results(session_id: str):
    """Get list of available result files for a session"""
    if session_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis session not found")
    
    session_data = analysis_status[session_id]
    if session_data["status"] != "completed":
        return {"available_results": [], "status": session_data["status"]}
    
    available_results = []
    html_files = {
        "aq_hotspots": "aq_hotspots.html",
        "uhi_hotspots": "uhi_hotspots.html", 
        "green_access": "green_access.html"
    }
    
    for analysis in session_data.get("completed_analyses", []):
        if analysis in html_files:
            file_path = os.path.join("web_outputs", session_id, html_files[analysis])
            if os.path.exists(file_path):
                available_results.append({
                    "analysis_type": analysis,
                    "analysis_name": analysis.replace('_', ' ').title(),
                    "file_name": html_files[analysis],
                    "url": f"/results/{session_id}/{analysis}"
                })
    
    return {"available_results": available_results, "status": "completed"}

@app.get("/health")
async def read_health():
    return {"status": "ok"}

@app.get("/api/districts")
async def get_districts():
    """Get list of available districts"""
    try:
        districts = router.get_districts_list()
        return {"districts": districts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching districts: {str(e)}")

@app.get("/api/upazilas/{district_name}")
async def get_upazilas(district_name: str):
    """Get list of upazilas for a specific district"""
    try:
        upazilas = router.get_upazilas_by_district(district_name)
        return {"upazilas": upazilas}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching upazilas: {str(e)}")
    
@app.post('/llm-inference')
async def llm_inference(request: LLM_Inference_Request):
    prompt = md(request.prompt)
    system_prompt = request.systemPrompt or groq_api.SYSTEM_PROMPT
    if request.type:
        if "green" in request.type.lower():
            system_prompt = groq_api.SYSTEM_PROMPT_GREEN
        if "uhi" in request.type.lower():
            system_prompt = groq_api.SYSTEM_PROMPT_UHI
        if "aq" in request.type.lower():
            system_prompt = groq_api.SYSTEM_PROMPT_AQ

    content, prompt_tokens, completion_tokens, total_tokens = groq_api.call_groq_with_system_and_user(system_prompt, prompt, groq_api.MODEL)
    global tokens_used
    tokens_used += total_tokens
    print("Total tokens used so far:", tokens_used)

    markdowned = False
    if request.markdown:
        try:
            content = markdown.markdown(content, extensions=['extra', 'toc', 'tables'])
            markdowned = True
        except Exception as e:
            print(f"Error converting to markdown: {e}")
    return {
        "response": content.replace('\n', ''),
        "markdowned": markdowned
    }


@app.get("/checker")
async def checker_function():
    global signature
    global checker
    checker += 1
    print("Checker incremented to:", checker, "Its signature is: ", signature)
    return {"checker": checker}


@app.get("/how-it-works")
async def how_it_works():
    return FileResponse("statics/how-it-works.html")