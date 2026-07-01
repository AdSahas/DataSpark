import io
import json
import os

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse

from backend.graph import build_graph
from backend.tools import set_dataframe

load_dotenv()

app = FastAPI(title="DataSpark API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("OPENAI_API_KEY")
agent = build_graph(api_key=api_key)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def index():
    return FileResponse(path="frontend/home.html", media_type="text/html")


@app.get("/app")
async def launch_app():
    return FileResponse(path="frontend/app.html", media_type="text/html")


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Make dataframe available to tools
    set_dataframe(df)

    # send an OK response with the number of rows and columns
    return {
        "message": "CSV uploaded successfully.",
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        result = agent.invoke(
            {
                "messages": [("user", req.message)],
                "dataset_type": None,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    messages = result.get("messages", [])

    tool_calls = []

    # Collect tool calls
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "tool": tc["name"],
                        "args": tc["args"],
                    }
                )

    # Attach tool outputs
    for msg in messages:
        if getattr(msg, "name", None):
            for tc in reversed(tool_calls):
                if tc["tool"] == msg.name and "result" not in tc:
                    try:
                        tc["result"] = json.loads(msg.content)
                    except Exception:
                        tc["result"] = msg.content
                    break

    final_message = ""

    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai":
            final_message = msg.content
            break

    if not final_message and messages:
        final_message = messages[-1].content

    if isinstance(final_message, list):
        final_message = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in final_message
        )

    if final_message.startswith("```"):
        lines = final_message.splitlines()
        if len(lines) >= 3:
            final_message = "\n".join(lines[1:-1])

    try:
        structured = json.loads(final_message)
    except Exception:
        structured = {
            "summary": final_message,
            "interpretation": None,
            "statistics": {},
            "insight": None,
            "error_guidance": None,
        }

    return {
        "tool_calls": tool_calls,
        **structured,
    }
