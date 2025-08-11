from typing import List


class DistributedBackend(object):

    def __init__(self, args):
        pass

    def transform_model(self, model):
        raise NotImplementedError

    def get_context_for_microstep_forward(
        self, model, microstep_idx, gradient_accumulation_steps
    ):
        raise NotImplementedError

    def is_master_process(self) -> bool:
        raise NotImplementedError

    def get_adjusted_args_for_process(self, args):
        raise NotImplementedError

    def get_raw_model(self, model):
        raise NotImplementedError

    def translate_model_parameter_name_for_node(self, parameter_name) -> List[str]:
        raise NotImplementedError

    def get_world_size(self):
        raise NotImplementedError

    def finalize(self):
        pass
