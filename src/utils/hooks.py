from collections import OrderedDict

import torch.nn as nn


def clear_runtime_hooks(module: nn.Module) -> None:
    module._backward_pre_hooks = OrderedDict()
    module._backward_hooks = OrderedDict()
    module._forward_hooks = OrderedDict()
    module._forward_hooks_with_kwargs = OrderedDict()
    module._forward_hooks_always_called = OrderedDict()
    module._forward_pre_hooks = OrderedDict()
    module._forward_pre_hooks_with_kwargs = OrderedDict()
    module._state_dict_hooks = OrderedDict()
    module._state_dict_pre_hooks = OrderedDict()
    module._load_state_dict_pre_hooks = OrderedDict()
    module._load_state_dict_post_hooks = OrderedDict()
