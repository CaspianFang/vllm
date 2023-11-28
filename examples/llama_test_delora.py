from vllm import LLM, SamplingParams
import time
if __name__ == "__main__":
    # prompt = "Hello, my dog is cute"
    prompt = "I am a graduate student at Johns Hopkins."
    prompts = [prompt]
    path = "/vllm_workspace/weights/backbone/alpaca_hf_ckpt_merged"
    lora_path = "/vllm_workspace/weights/loras/alpaca-lora-7b"
    lora_path_2 = "/vllm_workspace/weights/loras/bactrian-x-llama-7b-lora"
    lora_path_3 = "/vllm_workspace/weights/loras/wizardLM-lora-7b"
    
    # TODO: 1. multi-lora loading (FIXED)   2. ray optim
    
    llm = LLM(model=path,
              trust_remote_code=True,
              lora_paths=[lora_path, lora_path_2, lora_path_3],
              adapter_names=["adapter_1", "adapter_2", "adapter_3"],
              delora_name="adapter_1")
            #   lora_paths=[lora_path],
            #   adapter_names=["adapter_1"])
    # )

    print(llm.llm_engine.workers[0].model)

    sampling_params = SamplingParams(temperature=0,
                                     top_p=1,
                                     best_of=1,
                                     top_k=-1,
                                     max_tokens=100,
                                     use_beam_search=False,
                                     lora_id="adapter_1")
    llm._add_request(prompt=prompt,
                     prompt_token_ids=None,
                     sampling_params=sampling_params)

    sampling_params = SamplingParams(temperature=0,
                                     top_p=1,
                                     best_of=1,
                                     top_k=-1,
                                     max_tokens=100,
                                     use_beam_search=False,
                                     lora_id="adapter_2")
    llm._add_request(prompt=prompt,
                     prompt_token_ids=None,
                     sampling_params=sampling_params)

    sampling_params = SamplingParams(temperature=0,
                                     top_p=1,
                                     best_of=1,
                                     top_k=-1,
                                     max_tokens=100,
                                     use_beam_search=False,
                                     lora_id="adapter_3")
    llm._add_request(prompt=prompt,
                     prompt_token_ids=None,
                     sampling_params=sampling_params)
    
    sampling_params = SamplingParams(temperature=0,
                                     top_p=1,
                                     best_of=1,
                                     top_k=-1,
                                     max_tokens=100,
                                     use_beam_search=False)
    llm._add_request(prompt=prompt,
                     prompt_token_ids=None,
                     sampling_params=sampling_params)
    start = time.time()
    outputs = llm._run_engine(use_tqdm=True)
    end = time.time()
    print(f"cost: {end - start} s")
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")