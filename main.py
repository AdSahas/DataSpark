import json
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.messages import AIMessage, ToolMessage

from backend.graph import build_graph

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = build_graph()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return FileResponse("frontend/app.html")


@app.get("/app")
def dashboard():
    return FileResponse("frontend/app.html")


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    import pandas as pd
    import io
    from backend.tools import set_dataframe

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    set_dataframe(df)
    return {"filename": file.filename, "rows": len(df), "columns": list(df.columns)}


@app.post("/chat")
async def chat(req: ChatRequest):
    import traceback
    try:
        result = agent.invoke({
            "messages": [("user", req.message)],
            "dataset_type": None,
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    messages = result.get("messages", [])

    # ── Collect tool calls from the message history ──────────────────────────
    tool_calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"tool": tc["name"], "args": tc["args"]})

    # ── Find the final AIMessage and parse its JSON content ──────────────────
    structured_data = {}
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content

            # Case A: content is already a dict (rare but possible)
            if isinstance(content, dict):
                structured_data = content
                break

            # Case B: content is a JSON string (our case — model_dump_json())
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    # Make sure it looks like our AgentResponse schema
                    if "summary" in parsed or "interpretation" in parsed:
                        structured_data = parsed
                        break
                except json.JSONDecodeError:
                    # Plain text fallback
                    structured_data = {
                        "summary": content,
                        "interpretation": None,
                        "statistics": {},
                        "insight": None,
                        "thinking": None,
                        "error_guidance": None,
                    }
                    break

    return {
        "thinking":       structured_data.get("thinking"),
        "summary":        structured_data.get("summary"),
        "interpretation": structured_data.get("interpretation"),
        "statistics":     structured_data.get("statistics") or {},
        "insight":        structured_data.get("insight"),
        "error_guidance": structured_data.get("error_guidance"),
        "tool_calls":     tool_calls,
    }
