from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks, status, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Any
from models.anlyzers import router_lite
from models.llms import groq_api
import markdown
from markdownify import markdownify as md
# from fastapi.middleware.cors import CORSMiddleware
import random
import requests
import dotenv

dotenv.load_dotenv()


checker = 0
signature = random.random() * 1000000
tokens_used = 0

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'anlyzers'))

app = FastAPI()
os.makedirs("web_outputs", exist_ok=True)

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*", "null"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Mount static files
app.mount("/statics", StaticFiles(directory="statics"), name="static")
app.mount("/web_outputs", StaticFiles(directory="web_outputs"), name="web_outputs")

# Store active connections and analysis status
active_connections: Dict[str, WebSocket] = {}
analysis_status: Dict[str, Dict[str, Any]] = {}

completion_status: Dict[str, List[str]] = {}

# SSE subscribers per session_id -> set of asyncio.Queue
sse_subscribers: Dict[str, List[asyncio.Queue]] = {}

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

class RemoteTaskStarted(BaseModel):
    session_id: str
    analyses_name : str

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

async def request_to_remote(request: AnalysisRequest, tried = 0):
    start_time = datetime.now().timestamp()
    end_time = start_time + 60 * 60  # 1 hour from start
    try:
        response = requests.post(f"{os.getenv('REMOTE_SERVER_URL')}/run-analysis", json=request.model_dump())
    except Exception as e:
        print(str(e))
        if tried >= 4:
            print("Exception. Sending error...")
            analysis_status[request.session_id]["status"] = "error" + str(e)
            asyncio.run(manager.send_message(request.session_id, {
                "type": "error",
                "message": "Error connecting to remote server. Please try again later." + str(e)
            }))
            return
        else:
            tried += 1
            print("Exception. Attempting retry...")
            await asyncio.sleep(30)
            await request_to_remote(request, tried)
            return
    
    if response.status_code != 200:
        if tried >= 4:
            print("Status error. Sending error...")
            analysis_status[request.session_id]["status"] = "error"
            asyncio.run(manager.send_message(request.session_id, {
                "type": "error",
                "message": f"Error running analyses: {response.text}"
            }))
            return
        else:
            tried += 1
            print("Status error. Attempting retry...")
            await asyncio.sleep(30)
            return await request_to_remote(request, tried)
    if response.status_code == 200:
        while analysis_status[request.session_id]["status"] == "running" and datetime.now().timestamp() < end_time:
            await asyncio.sleep(120)
            requests.get(f"{os.getenv('REMOTE_SERVER_URL')}")
                
        
    

@app.get("/")
async def read_root():
    return RedirectResponse("/statics/map.html")

# -----------------------------
# Server-Sent Events (SSE)
# -----------------------------

async def _ensure_session(session_id: str):
    if session_id not in analysis_status:
        analysis_status[session_id] = {
            "status": "idle",
            "requested_analyses": [],
            "completed_analyses": [],
            "data": {},
            "timestamp": datetime.now().isoformat(),
        }
    if session_id not in sse_subscribers:
        sse_subscribers[session_id] = []

async def _broadcast_event(session_id: str, event: Dict[str, Any]):
    # Fan out to all SSE queues for this session
    queues = sse_subscribers.get(session_id, [])
    # Attach server timestamp
    event = {**event, "server_ts": datetime.now().isoformat()}
    for q in list(queues):
        try:
            await q.put(event)
        except Exception:
            # best-effort
            pass

def _format_sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"

@app.get("/sse/{session_id}")
async def sse(session_id: str, request: Request):
    await _ensure_session(session_id)
    queue: asyncio.Queue = asyncio.Queue()
    sse_subscribers[session_id].append(queue)

    async def event_generator():
        # Initial hello
        yield _format_sse({"type": "connected", "message": "SSE connected"})
        try:
            # Heartbeat ticker
            last_heartbeat = datetime.now().timestamp()
            while True:
                # client disconnect?
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield _format_sse(event)
                except asyncio.TimeoutError:
                    # heartbeat to keep connection alive
                    now = datetime.now().timestamp()
                    if now - last_heartbeat >= 15:
                        last_heartbeat = now
                        yield _format_sse({"type": "heartbeat"})
        finally:
            # cleanup
            if session_id in sse_subscribers and queue in sse_subscribers[session_id]:
                sse_subscribers[session_id].remove(queue)

    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), headers=headers)

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
        "completed_analyses": [],
        "data": {},
        "timestamp": datetime.now().isoformat(),
    }

    await asyncio.sleep(1)
    
    # Start analyses in background
    background_tasks.add_task(request_to_remote, request)
    
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

@app.get("/results-data/{session_id}/{analysis_type}")
async def get_results_data(session_id: str, analysis_type: str):
    if session_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis session not found")
    data = analysis_status[session_id].get("data", {}).get(analysis_type)
    if data is None:
        raise HTTPException(status_code=404, detail="No data found for requested analysis")
    return data

@app.get("/health")
async def read_health():
    return {"status": "ok"}

@app.get("/api/districts")
async def get_districts():
    """Get list of available districts"""
    try:
        districts = router_lite.get_districts_list()
        return {"districts": districts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching districts: {str(e)}")

@app.get("/api/upazilas/{district_name}")
async def get_upazilas(district_name: str):
    """Get list of upazilas for a specific district"""
    try:
        upazilas = router_lite.get_upazilas_by_district(district_name)
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



@app.post("/xxd-completed-analysis-from-remote-server")
async def analysis_response_from_remote(files: List[UploadFile] = File(None), session_id: str = Form(...)):
    if session_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis session not found")
    os.makedirs(os.path.join("web_outputs", session_id), exist_ok=True)
    for file in files:
        file_location = os.path.join("web_outputs", session_id, file.filename)
        with open(file_location, 'wb') as out_file:
            content = await file.read()  # async read
            out_file.write(content)
    return {"message": "Files saved"}


@app.post("/xxd-push-event")
async def push_event(request: Request, session_id: Optional[str] = None, message: Optional[str] = None):
    """Remote server pushes an event that will be broadcast via SSE and used to update state.
    Accepts either JSON body {session_id, message} or query params.
    """
    # Read from body if not provided as query params
    if session_id is None or message is None:
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")
        session_id = body.get("session_id")
        event = body.get("message")
    else:
        # message passed as JSON string in query param
        try:
            event = json.loads(message)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    # If event is still a string, try to parse it; otherwise expect dict
    if isinstance(event, str):
        try:
            event = json.loads(event)
        except Exception:
            # keep as string, but wrap
            event = {"type": "message", "message": event}

    if not isinstance(event, dict):
        raise HTTPException(status_code=400, detail="message must be a JSON object or JSON string")

    await _ensure_session(session_id)

    # Update in-memory status on certain events
    etype = event.get("type")
    analysis = event.get("analysis")

    if etype == "progress_update":
        analysis_status[session_id]["status"] = "running"
    elif etype == "analysis_start" and analysis:
        analysis_status[session_id]["status"] = "running"
    elif etype == "analysis_complete" and analysis:
        analysis_status[session_id].setdefault("completed_analyses", [])
        if analysis not in analysis_status[session_id]["completed_analyses"]:
            analysis_status[session_id]["completed_analyses"].append(analysis)
        if "data" in event:
            analysis_status[session_id].setdefault("data", {})[analysis] = event["data"]
        req = set(analysis_status[session_id].get("requested_analyses", []))
        done = set(analysis_status[session_id].get("completed_analyses", []))
        if req and req.issubset(done):
            analysis_status[session_id]["status"] = "completed"
    elif etype == "all_analyses_complete":
        analysis_status[session_id]["status"] = "completed"
    elif etype == "error":
        analysis_status[session_id]["status"] = "error"

    # Broadcast to connected SSE clients
    await _broadcast_event(session_id, event)
    return {"message": "ok"}

# Backward-compat: keep old endpoint name but route to SSE
@app.post("/xxd-send-websocket-message-to-the-client")
async def send_websocket_message_to_client(message: str, session_id: str):
    return await push_event(session_id=session_id, message=message)


@app.post("/xxd-update-analysis-status")
async def update_analysis_status(session_id: str, status: str):
    if session_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis session not found")
    analysis_status[session_id] = json.loads(status)
    return {"message": "Status updated successfully"}