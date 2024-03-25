```bash
cd examples/api_server
python3 server.py

# client: 
mkdir download_dataset && cd download_dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

cd ..
python3 ../../benchmarks/benchmark_serving.py \
        --backend vllm \
        --model ../../../weights/backbone/llama_7b_hf \
        --dataset ./download_dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
        --request-rate 20 \
        --save-result 
```