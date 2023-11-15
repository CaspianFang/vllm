from vllm import LLM, SamplingParams
import ray

prompts = [
    "Beijing University of Posts and Telecommunications is",
    "MIT is",
    "CMU is",
    "THU is",
    "Before climbing the mountain, we should prepare"
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=75)

# llm = LLM("/media/airaiot/Workspace/triton_env/weights_repository/llama_7b_hf", tensor_parallel_size=2)
llm = LLM("/vllm_workspace/weights/llama_7b_hf", tensor_parallel_size=2)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, \n Generated text: {generated_text!r}")
