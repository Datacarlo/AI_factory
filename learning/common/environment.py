"""Technical training environment without host names or device indices."""
import platform
from pathlib import Path
import numpy as np
import torch
import yaml


def environment(device=None):
    cpu = platform.processor() or platform.machine()
    if Path('/proc/cpuinfo').exists():
        cpu = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                    if s.startswith('model name')), cpu)
    uses_cuda = torch.cuda.is_available() and (device is None or str(device).startswith('cuda'))
    hardware = dict(device='gpu' if uses_cuda else 'cpu', cpu_model=cpu)
    if uses_cuda:
        hardware.update(name=torch.cuda.get_device_name(),
                        compute_capability='.'.join(map(str, torch.cuda.get_device_capability())))
    return dict(hardware=hardware, software=dict(python=platform.python_version(),
        pytorch=str(torch.__version__), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), numpy=np.__version__, pyyaml=yaml.__version__))
