import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.lora.request import LoRARequest,OLoRARequest

import random
import time

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

lora_path = ["../../../weights/loras/alpaca-lora-7b","../../../weights/loras/wizardLM-lora-7b"]
LoRA_id_list = [1,2]
LoRA_Path_list = lora_path
LoRA_name_list = ["alpaca-lora-7b","wizardLM-lora-7b"]
all_lora_reqs = lora_reqs= [LoRARequest("alpaca-lora-7b", 1, lora_path[0]), LoRARequest("wizardLM-lora-7b",2,lora_path[1])]

now = time.monotonic()
LOG_INTERVAL_SEC = 5

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    prefix_pos = request_dict.pop("prefix_pos", None)
    stream = request_dict.pop("stream", False)
    
    olora = request_dict.pop("olora", None)
    olora_request = None
    
    lora = request_dict.pop("lora", None)
    lora_request = None
    
    estimated_tokens = request_dict['max_tokens']
    
    if olora is not None:
        olora_request = OLoRARequest(LoRA_name_list[:-1],
                                    olora_int_ids=LoRA_id_list[:-1],
                                    lora_local_path=LoRA_Path_list[:-1],
                                    self_id = 2, 
                                    lora_reqs= all_lora_reqs[:-1])
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    
    if lora is not None:
        lora_request = all_lora_reqs[random.randint(0,1)]
        
    start = time.monotonic()
    if olora_request is not None:
        results_generator = engine.generate(prompt,
                                            sampling_params,
                                            request_id,
                                            olora_request=olora_request,
                                            prefix_pos=prefix_pos)
    elif lora_request is not None:
        results_generator = engine.generate(prompt,
                                            sampling_params,
                                            request_id,
                                            lora_request=lora_request,
                                            prefix_pos=prefix_pos)
    else:
        results_generator = engine.generate(prompt,
                                            sampling_params,
                                            request_id,
                                            prefix_pos=prefix_pos)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    
    end = time.monotonic()
    global now
    duration = time.monotonic() - now
    if duration > LOG_INTERVAL_SEC:
        print(f"Normalized Latency: {(end - start) / estimated_tokens} s/token")
        now = time.monotonic()
    return JSONResponse(ret)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    engine_args = AsyncEngineArgs(
        model="../../../weights/backbone/llama_7b_hf",
        enable_lora=True,
        enable_olora=False,
        enforce_eager=True,
        max_loras=5,
        max_lora_rank=16,
        max_cpu_loras=12,
        max_num_seqs=64,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        disable_log_requests=True,
        swap_space=16
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    uvicorn.run(app, host="127.0.0.1", port=8001, access_log=False)