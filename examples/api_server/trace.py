import requests
import os
import json
from typing import List

import asyncio
import aiohttp
import time

URL = 'http://127.0.0.1:8000/generate'


def load_promts(text_dir, prompt_num: int) -> List[str]:
    texts = []
    for file in os.listdir(text_dir):
        with open(os.path.join(text_dir, file), 'r') as f:
            data = json.load(f)
            prompts: list = data['data']
            texts.extend(prompts)
            if len(texts) >= prompt_num:
                break
            
    return texts

async def async_request_task(session, url, data):
    async with session.post(url, json=data) as response:
        return await response.json()
    
async def async_requests(avg_tokens: int, request_rate: int):
    assert avg_tokens in [25, 50, 100, 200, 250], "avg_tokens should be one of [25, 50, 100, 200, 250]"
    texts = load_promts(f"text_dataset/prompts_size50_avg{avg_tokens}", prompt_num=request_rate)
    print("finish loading")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(request_rate):
            data = {
                "prompt": texts[i],
                "max_tokens": avg_tokens + 10,
                # "olora": True
            }
            tasks.append(async_request_task(session, URL, data))
        
        print("sending")
        start = time.monotonic()
        response = await asyncio.gather(*tasks)
        end = time.monotonic()
        duration = end - start
        tokens_total_pred = (avg_tokens + 10) * request_rate
        s_per_token = duration / tokens_total_pred
        print(f"request rate: {request_rate}, Normalized Latency: {s_per_token:.6f} s/token")


if __name__ == '__main__':
    request_rate = 40
    time_gap = 0.02
    asyncio.run(async_requests(avg_tokens=25, request_rate=request_rate))
    # import time
    # time.sleep(time_gap)
    # asyncio.run(async_requests(avg_tokens=25, request_rate=request_rate))
    # time.sleep(time_gap)
    # asyncio.run(async_requests(avg_tokens=50, request_rate=request_rate))