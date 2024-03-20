"""
This example shows how to use the multi-LoRA functionality for offline inference.

Requires HuggingFace credentials for access to Llama2.
"""

from typing import Optional, List, Tuple

#from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest,OLoRARequest


def create_test_prompts(lora_path: List[str]) -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters.
    
    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=1`, the expectation is that the requests
    with the second LoRA adapter will be ran after all requests with the
    first adapter have finished.
    """

    LoRA_id_list = [1,2,3]
    LoRA_Path_list = lora_path
    LoRA_name_list = ["alpaca-lora-7b","bactrian-x-llama-7b-lora","wizardLM-lora-7b"]
    all_lora_reqs = lora_reqs= [LoRARequest("alpaca-lora-7b", 1, lora_path[0]),LoRARequest("bactrian-x-llama-7b-lora",2,lora_path[1]), LoRARequest("wizardLM-lora-7b",3,lora_path[2])]

    # return [
    #     ("A robot may not injure a human being",
    #      SamplingParams(temperature=0.0,
    #                     logprobs=1,
    #                     prompt_logprobs=1,
    #                     max_tokens=128), None),
    #     ("To be or not to be,",
    #      SamplingParams(temperature=0.8,
    #                     top_k=5,
    #                     presence_penalty=0.2,
    #                     max_tokens=128), None),
    #     ("[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
    #      SamplingParams(temperature=0.0,
    #                     logprobs=1,
    #                     prompt_logprobs=1,
    #                     max_tokens=128,
    #                     stop_token_ids=[32003]),
    #      OLoRARequest(LoRA_name_list,olora_int_ids=LoRA_id_list,lora_local_path=LoRA_Path_list,self_id = 1, lora_reqs= all_lora_reqs  )),
    #     ("[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
    #      SamplingParams(n=3,
    #                     best_of=3,
    #                     use_beam_search=True,
    #                     temperature=0,
    #                     max_tokens=128,
    #                     stop_token_ids=[32003]),
    #      OLoRARequest(LoRA_name_list[:-1],olora_int_ids=LoRA_id_list[:-1],lora_local_path=LoRA_Path_list[:-1],self_id = 2, lora_reqs= all_lora_reqs[:-1])),
    #     ("[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
    #      SamplingParams(temperature=0.0,
    #                     logprobs=1,
    #                     prompt_logprobs=1,
    #                     max_tokens=128,
    #                     stop_token_ids=[32003]),
    #      OLoRARequest(LoRA_name_list[:-1],olora_int_ids=LoRA_id_list[:-1],lora_local_path=LoRA_Path_list[:-1],self_id = 2, lora_reqs= all_lora_reqs[:-1])),
    #     ("[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
    #      SamplingParams(n=3,
    #                     best_of=3,
    #                     use_beam_search=True,
    #                     temperature=0,
    #                     max_tokens=128,
    #                     stop_token_ids=[32003]),
    #      OLoRARequest(LoRA_name_list[0:1],olora_int_ids=LoRA_id_list[0:1],lora_local_path=LoRA_Path_list[0:1],self_id = 3, lora_reqs= all_lora_reqs[0:1])),
    # ]




    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128), None),
        ("To be or not to be,",
         SamplingParams(temperature=0.8,
                        top_k=5,
                        presence_penalty=0.2,
                        max_tokens=128), None),
        ("[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
         SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128,
                        stop_token_ids=[32003]),
                        None),
        ("[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
         SamplingParams(n=3,
                        best_of=3,
                        use_beam_search=True,
                        temperature=0,
                        max_tokens=128,
                        stop_token_ids=[32003]),
                        None),
        ("[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
         SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128,
                        stop_token_ids=[32003]),
                        None),
        ("[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
         SamplingParams(n=3,
                        best_of=3,
                        use_beam_search=True,
                        temperature=0,
                        max_tokens=128,
                        stop_token_ids=[32003]),
                        None),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams,
                                              Optional[OLoRARequest]]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_olora_request(str(request_id),
                               prompt,
                               sampling_params,
                               olora_request=lora_request)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output.outputs)


def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(model="../../weights/backbone/llama_7b_hf",
                             enable_lora=False,
                             enable_olora=False,
                             enforce_eager=True,
                             max_loras=1,
                             max_lora_rank=8,
                             max_cpu_loras=2,
                             max_num_seqs=256)
    return LLMEngine.from_engine_args(engine_args)


def main():
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine()
    lora_path = ["../../weights/loras/alpaca-lora-7b","../../weights/loras/bactrian-x-llama-7b-lora","../../weights/loras/wizardLM-lora-7b"]
    test_prompts = create_test_prompts(lora_path)
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    main()