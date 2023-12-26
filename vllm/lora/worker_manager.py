import logging
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, List, Optional, Set, Type, Union
from collections import Counter
import torch

from vllm.lora.models import (TARGET_MODULES_QKV, LoRAModel, LoRAModelManager,
                              LRUCacheLoRAModelManager, create_lora_adapter)
from vllm.lora.request import LoRARequest
from vllm.lora.layers import LoRAMapping
from vllm.config import LoRAConfig

logger = logging.getLogger(__name__)


class AbstractWorkerLoRAManager(ABC):
    """Abstract class for managing LoRA models on the worker side."""

    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int,
                 vocab_size: int, lora_config: LoRAConfig,
                 device: torch.device):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.vocab_size = vocab_size
        self.device = device
        self.lora_config = lora_config

    @abstractproperty
    def is_enabled(self) -> bool:
        ...

    @abstractmethod
    def create_lora_adapter(
        self,
        model: torch.nn.Module,
        target_modules: Union[str, List[str]] = TARGET_MODULES_QKV,
    ) -> Any:
        ...

    @abstractmethod
    def apply_loras(self, lora_requests: List[LoRARequest],
                    lora_mapping: LoRAMapping) -> None:
        ...

    @abstractmethod
    def add_lora(self, lora_request: LoRARequest) -> bool:
        ...

    @abstractmethod
    def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool:
        ...

    @abstractmethod
    def remove_lora(self, lora_id: int) -> bool:
        ...

    @abstractmethod
    def remove_all_loras(self) -> bool:
        ...

    @abstractmethod
    def list_loras(self) -> Set[int]:
        ...


class DisabledWorkerLoRAManager(AbstractWorkerLoRAManager):
    """WorkerLoRAManager that does nothing."""

    @property
    def is_enabled(self) -> bool:
        return False

    def create_lora_adapter(
        self,
        model: torch.nn.Module,
        target_modules: Union[str, List[str]] = TARGET_MODULES_QKV,
    ) -> Any:
        return model

    def apply_loras(self, lora_requests: List[LoRARequest],
                    lora_mapping: LoRAMapping) -> None:
        return

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return False

    def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool:
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def remove_all_loras(self) -> bool:
        return

    def list_loras(self) -> Set[int]:
        return set()


class WorkerLoRAManager(AbstractWorkerLoRAManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Every request, the requested LoRAs will be loaded (unless they are already
    loaded), and every other LoRA will be unloaded."""

    _lora_manager_cls: Type[LoRAModelManager] = LoRAModelManager

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        device: torch.device,
        lora_model_cls: Type[LoRAModel] = LoRAModel,
    ):
        self._lora_manager: Optional[LoRAModelManager] = None
        self._lora_model_cls = lora_model_cls
        super().__init__(max_num_seqs, max_num_batched_tokens, vocab_size,
                         lora_config, device)
        self.ite_lora_reqs = None

    @property
    def is_enabled(self) -> bool:
        return True

    def create_lora_adapter(
        self,
        model: torch.nn.Module,
        target_modules: Union[str, List[str]] = TARGET_MODULES_QKV,
    ) -> Any:
        lora_manager = create_lora_adapter(
            model,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            target_modules=target_modules,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            lora_manager_cls=self._lora_manager_cls,
        )
        self._lora_manager = lora_manager
        return lora_manager.model

    def apply_loras(self, lora_requests: List[LoRARequest],
                    lora_mapping: LoRAMapping) -> None:
        self._apply_loras(lora_requests)
        self._lora_manager.set_row_lora_mapping(lora_mapping)

    def _apply_loras(self, lora_requests: List[LoRARequest]) -> None:
        loras_that_exist = self.list_loras()
        loras_map = {
            lora_request.lora_int_id: lora_request
            for lora_request in lora_requests if lora_request
        }
        if len(loras_map) > self._lora_manager.max_num_seqs:
            raise RuntimeError(
                f"Number of requested LoRAs ({len(loras_map)}) is greater "
                "than the number of GPU LoRA slots "
                f"({self._lora_manager.max_num_seqs}).")

        new_loras = set(loras_map)
        loras_to_add = new_loras - loras_that_exist
        loras_to_remove = loras_that_exist - new_loras

        for lora_id in loras_to_remove:
            self.remove_lora(lora_id)   # here remove_lora is  simply put index_id dict to None, didn't unload the weight -hqf

        for lora_id in loras_to_add:
            self.add_lora(loras_map[lora_id]) # 

    def _load_lora(self, lora_request: LoRARequest) -> LoRAModel:  ## lora lora weight to cpu - hqf
        try:
            lora = self._lora_model_cls.from_local_checkpoint(
                lora_request.lora_local_path,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",
                dtype=self.lora_config.lora_dtype,
                target_embedding_padding=self.vocab_size +
                self.lora_config.lora_extra_vocab_size,
            )
        except Exception as e:
            raise RuntimeError(
                f"Loading lora {lora_request.lora_local_path} failed") from e
        if lora.rank > self.lora_config.max_lora_rank:
            raise ValueError(
                f"LoRA rank {lora.rank} is greater than max_lora_rank "
                f"{self.lora_config.max_lora_rank}.")
        return lora

    def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool:
        if lora_request.lora_int_id in self.list_loras():
            return False
        return self._lora_manager.add_lora(
            self._lora_manager.create_dummy_lora(lora_request.lora_int_id,
                                                 rank))

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if lora_request.lora_int_id in self.list_loras():
            return False
        lora = self._load_lora(lora_request)
        loaded = self._lora_manager.add_lora(lora)
        self._lora_manager.activate_lora(lora.id)
        return loaded

    def remove_lora(self, lora_id: int) -> bool:
        return self._lora_manager.remove_lora(lora_id)

    def remove_all_loras(self) -> bool:
        self._lora_manager.remove_all_loras()

    def list_loras(self) -> Set[int]:
        return set(self._lora_manager.list_loras())
    def schedule_modes(self,lora_requests:List[LoRARequest],
                      lora_counter:Counter,lora_mapping:LoRAMapping) -> None:
        most_common_count = lora_counter.most_common(1)[0][1]
        most_common_lora_id = lora_counter.most_common(1)[0][0]
        total_count = sum(lora_counter.values())
        change_to_delora = most_common_count > 0.5 * total_count
        if ( self._lora_manager.now_backbone_lora and  self._lora_manager.now_delora) : ##  these condition can't make sure all layers are delora, but at least one layer
            mode = "de-lora"
        elif (self._lora_manager.now_backbone_lora and (not self._lora_manager.now_delora) ): ## all layers are merged
            mode = "merged" 
        else:
            mode = "unmerged"                            ## all layers are unmerged
        new_loras = Set(lora_mapping.index_mapping)
        lora_num = len(new_loras)
        if mode == "merged":
            if (self.list_loras() != new_loras):
                # add backbone lora hqf 
                lora_requests.append(self.ite_lora_reqs.get(self._lora_manager.now_backbone_lora))
                self.apply_loras(lora_requests,lora_mapping)
                self._lora_manager.merged_to_delora_all(self._lora_manager.now_backbone_lora)
        elif mode == "unmerged":
            self.apply_loras(lora_requests,lora_mapping)
            if (change_to_delora):
                self._lora_manager.unmerged_to_delora_one(most_common_lora_id)
        else :
            if (change_to_delora):    
                if ( self._lora_manager.now_delora == most_common_lora_id ):            #  new_delora = old delora
                    self.apply_loras(lora_requests,lora_mapping)
                    all_delora = self._lora_manager.unmerged_to_delora_one( most_common_lora_id )
                    if (all_delora and lora_num == 1):
                        self._lora_manager.delora_to_merged_all(most_common_lora_id)
                else :
                    # should add now_backbone_lora hqf
                    lora_requests.append(self.ite_lora_reqs.get(self._lora_manager.now_backbone_lora))
                    self.apply_loras(lora_requests,lora_mapping)
                    self._lora_manager.delora_to_unmerged_one()      
            else :
                self._lora_manager.delora_to_unmerged_one()
        self.ite_lora_reqs = dict([{req.lora_int_id,req} for req in lora_requests])


class LRUCacheWorkerLoRAManager(WorkerLoRAManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Uses an LRU Cache. Every request, the requested LoRAs will be loaded
    (unless they are already loaded) and least recently used LoRAs will
    be unloaded if the cache is above capacity."""

    _lora_manager_cls: Type[
        LRUCacheLoRAModelManager] = LRUCacheLoRAModelManager

    def create_lora_adapter(
        self,
        model: torch.nn.Module,
        target_modules: Union[str, List[str]] = TARGET_MODULES_QKV,
    ) -> Any:
        lora_manager = create_lora_adapter(
            model,
            target_modules=target_modules,
            lora_manager_cls=self._lora_manager_cls,
            max_num_seqs=self.max_num_seqs,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            max_num_batched_tokens=self.max_num_batched_tokens,
        )
        self._lora_manager = lora_manager
        return lora_manager.model

    def _apply_loras(self, lora_requests: List[LoRARequest]) -> None:
        loras_map = {
            lora_request.lora_int_id: lora_request
            for lora_request in lora_requests if lora_request
        }
        if len(loras_map) > self._lora_manager.max_num_seqs:
            raise RuntimeError(
                f"Number of requested LoRAs ({len(loras_map)}) is greater "
                "than the number of GPU LoRA slots "
                f"({self._lora_manager.max_num_seqs}).")
        for lora in loras_map.values():
            self.add_lora(lora)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if lora_request.lora_int_id not in self.list_loras():
            # Remove before we load the new lora to save memory
            if len(self._lora_manager) + 1 > self._lora_manager.capacity:
                self._lora_manager.remove_oldest_lora()
            lora = self._load_lora(lora_request)
            loaded = self._lora_manager.add_lora(lora)
        else:
            # If the lora is already loaded, just touch it to
            # update its position in the caches
            loaded = self._lora_manager.get_lora(lora_request.lora_int_id)
        self._lora_manager.activate_lora(lora_request.lora_int_id)
        return loaded
