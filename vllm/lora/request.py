from dataclasses import dataclass

from typing import List


@dataclass
class LoRARequest:
    """
    Request for a LoRA adapter.

    Note that this class should be be used internally. For online
    serving, it is recommended to not allow users to use this class but
    instead provide another layer of abstraction to prevent users from
    accessing unauthorized LoRA adapters.

    lora_int_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.
    """

    lora_name: str
    lora_int_id: int
    lora_local_path: str

    def __post_init__(self):
        if self.lora_int_id < 1:
            raise ValueError(
                f"lora_int_id must be > 0, got {self.lora_int_id}")

    def __eq__(self, value: object) -> bool:
        return isinstance(
            value, LoRARequest) and self.lora_int_id == value.lora_int_id

    def __hash__(self) -> int:
        return self.lora_int_id


@dataclass
class OLoRARequest:
    lora_name: List[str]
    lora_int_id: List[int]
    lora_local_path: List[str]
    
    def __post_init__(self):
        for _id in self.lora_int_id:
            if _id < 1:
                raise ValueError(
                    f"lora_int_id must be > 0, got {_id}")
            
    def __eq__(self, value: object) -> bool:
        status = True
        if isinstance(value, OLoRARequest):
            for k, v in zip(self.lora_int_id, value.lora_int_id):
                if k != v:
                    status = False
        return isinstance(
            value, OLoRARequest) and status
        
    def __hash__(self) -> int:
        # sort the list of lora_int_id
        # TODO: There may be some problems
        return hash(self.lora_int_id.sort())