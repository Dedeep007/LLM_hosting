import asyncio
import datetime
import torch

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# ---------- APP ----------
app = FastAPI()

# ---------- MODEL ----------
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    trust_remote_code=True,

    # ---- 4-bit quantization ----
    quantization="bitsandbytes",

    # ---- GPU memory controls ----
    dtype="bfloat16",
    gpu_memory_utilization=0.75,
    max_model_len=2048,
    max_num_seqs=32,
)

engine = AsyncLLMEngine.from_engine_args(engine_args)

# ---------- SCHEMA ----------
class ChatMessage(BaseModel):
    message: str
    temperature: float = 0.7
    max_tokens: int = 200
    stream: bool = False


# ---------- ENDPOINT ----------
@app.post("/chat")
async def chat(payload: ChatMessage, request: Request):
    timestamp = datetime.datetime.now().isoformat()
    client_ip = request.client.host if request.client else "unknown"

    sampling_params = SamplingParams(
        temperature=payload.temperature,
        top_p=0.9,
        max_tokens=payload.max_tokens,
    )

    request_id = f"{timestamp}-{client_ip}"

    # ---------- STREAMING ----------
    if payload.stream:
        async def token_stream():
            previous_text = ""
            async for output in engine.generate(payload.message, sampling_params, request_id):
                for completion in output.outputs:
                    # Yield only the new tokens (delta)
                    new_text = completion.text[len(previous_text):]
                    previous_text = completion.text
                    if new_text:
                        yield new_text

        return StreamingResponse(
            token_stream(),
            media_type="text/plain",
        )

    # ---------- NON-STREAM ----------
    final_output = ""
    async for output in engine.generate(payload.message, sampling_params, request_id):
        final_output = output.outputs[0].text

    return {
        "timestamp": timestamp,
        "client_ip": client_ip,
        "prompt": payload.message,
        "response": final_output,
    }
