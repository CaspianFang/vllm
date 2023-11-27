from vllm.model_executor.parallel_utils.layers import BLoraColumnParallelLinear, BLoraQKVColumnParallelLinear, BLoraRowParallelLinear, ColumnParallelLinear, RowParallelLinear, QKVLoraLayer
from peft.tuners.lora import LoraLayer
from peft import LoraConfig
import re
import torch
import time

WEIGHTS_NAME = "adapter_model.bin"
PREFIX = "base_model.model."
PARAMETER_PREFIX = "lora_"


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

# MODIFY
def _create_new_module(lora_config, adapter_name, target, delora_name):
    lora_alpha = lora_config.lora_alpha
    r = lora_config.r
    lora_dropout = lora_config.lora_dropout
    if isinstance(target, ColumnParallelLinear):
        new_module = BLoraQKVColumnParallelLinear(
            input_size=target.input_size,   # 4096 
            output_size=target.output_size_per_partition,   # 4096 * 3
            adapter_name=adapter_name,
            bias=target.bias,
            gather_output=target.gather_output,
            skip_bias_add=target.skip_bias_add,
            quant_config=target.quant_config,
            lora_alpha=lora_alpha,
            r=r,
            lora_dropout=lora_dropout,
            delora_name=delora_name)
        return new_module
    if isinstance(target, RowParallelLinear):
        new_module = BLoraRowParallelLinear(
            input_size=target.input_size_per_partition,
            output_size=target.output_size,
            adapter_name=adapter_name,
            bias=target.bias,
            input_is_parallel=target.input_is_parallel,
            reduce_results=target.reduce_results,
            skip_bias_add=target.skip_bias_add,
            quant_config=target.quant_config,
            lora_alpha=lora_alpha,
            r=r,
            lora_dropout=lora_dropout,
            delora_name=delora_name)
        return new_module


def _replace_module(parent, 
                    child_name,     # qkv_proj | o_proj
                    new_module,     # BLoraColumnParallelLinear() | BLoraRowParallelLinear()
                    child):         # ColumnParallelLinear() | RowParallelLinear()
    setattr(parent, child_name, new_module)
    """ state_dict: 
    new_module: BLoraColumnParallelLinear(): weight,    
                                             lora_A.adapter_1.weight, 
                                             lora_B.adapter_1.weight
                                             
    child:      ColumnParallelLinear():      weight,
    """
    new_module.weight = child.weight
    
    if getattr(child, "state", None) is not None:
        new_module.state = child.state
        new_module.to(child.weight.device)
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(child.weight.device)


def _create_and_replace(lora_config, adapter_name, 
                        target,         # ColumnParallelLinear() | RowParallelLinear()
                        target_name,    # qkv_proj | o_proj
                        parent,         # LlamaAttentio
                        delora_name):        
    if (isinstance(target, (ColumnParallelLinear, RowParallelLinear))
            and not isinstance(target, (QKVLoraLayer, LoraLayer))):
        new_module = _create_new_module(lora_config, adapter_name, target, delora_name)  # get a BLORA module, e.g. BLoraColumnParallelLinear()
        _replace_module(parent,         # LlamaAttention
                        target_name,    # qkv_proj | o_proj
                        new_module,     # BLoraColumnParallelLinear()
                        target)         # ColumnParallelLinear() | RowParallelLinear()
        
    elif isinstance(target, (QKVLoraLayer, LoraLayer)):
        target.update_layer(adapter_name, 
                            lora_config.r,
                            lora_config.lora_alpha, 
                            lora_config.lora_dropout,
                            lora_config.init_lora_weights,
                            delora_name)


def add_lora_adapter(model: torch.nn.Module, 
                     lora_path: str,
                     adapter_name: str,
                     delora_name: str = None):
    lora_config = LoraConfig.from_pretrained(lora_path,
                                             revision=None,
                                             use_auth_token=None)
    key_list = [key for key, _ in model.named_modules()]

    
    model_state_dict_be = model.state_dict()
    
    # iterate the modules of LLaMa to insert the LoRA adapter
    
    # TODO: we should re-construct the logic from here to fit LlaMa LoRA
    
    
    for key in key_list:
        if "lora_" in key:
            continue
        # ==== OLD ====
        target_module_found = any(
            re.match(f".*\\.{target_key}$", key)    
            for target_key in lora_config.target_modules) or any(
                target_key == key for target_key in lora_config.target_modules)
            
        # ==== NEW ====
        target_module_found = any(
            re.search(target_key, key) for target_key in lora_config.target_modules) or any(
                target_key == key for target_key in lora_config.target_modules)
            
        # key: qkv_proj
        # target_key: q_proj, k_proj, v_proj
            
        if not target_module_found:
            continue

        parent, target, target_name = _get_submodules(model, key)
        # parent: LlamaAttention
        # target: ColumnParallelLinear() | RowParallelLinear()
        # target_name: qkv_proj | o_proj

        print(f"target: {target!r}")
        
        # create and replace
        _create_and_replace(lora_config, 
                            adapter_name, # "adapter_1"
                            target,       # ColumnParallelLinear()
                            target_name,  # qkv_proj
                            parent,
                            delora_name)       # LlamaAttention

    adapters_weights = torch.load(f"{lora_path}/{WEIGHTS_NAME}")

    processed_adapter_state_dict = {}
    for key, value in adapters_weights.items():
        if key.startswith(PREFIX):
            new_key = key[len(PREFIX):]
        else:
            new_key = key
        processed_adapter_state_dict[new_key] = value

    state_dict = {}
    for k, v in processed_adapter_state_dict.items():
        if PARAMETER_PREFIX in k:
            suffix = k.split(PARAMETER_PREFIX)[1]
            if "." in suffix:
                to_replace = ".".join(suffix.split(".")[1:])
                k = k.replace(to_replace, f"{adapter_name}.{to_replace}")
            else:
                k = f"{k}.{adapter_name}"
        state_dict[k] = v

    # https://github.com/huggingface/peft/blob/70302d7b4fc667f6b2e80ac1b8cccde081270d1e/src/peft/peft_model.py#L555

    

    model_state_dict_af = model.state_dict()
    print(f"diff: {len(model_state_dict_af) - len(model_state_dict_be)}")
    
    # MODIFY
    # for key in model_state_dict_af.keys():
    #     if key not in model_state_dict_be.keys():
    #         print(key)
    
    # print("=== model_state_dict_af ===")

    # for key in model_state_dict_af.keys():
    #     print(key)
    # END

    model.load_lora_weights_parallel(state_dict)
    model.cuda()