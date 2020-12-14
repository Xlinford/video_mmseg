from .psp_head import PSPHead
from .fcn_head import FCNHead
from .memory_head import MemoryHead
from .memory_gap_head import MemoryGAPHead
from .memory_gap_guide_head import MemoryGAPGuideHead
from .memory_downsample_head import MemoryDownSampleHead
from .memory_downsample_weight_pooling_head import MemoryDownSampWeightPoolingHead
from .memory_global_context_head import MemoryGlobalContextHead
from .memory_global_context_head2 import MemoryGlobalContextHead2
from .memory_global_context_head3 import MemoryGlobalContextHead3
from .memory_channel_head import MemoryHead2
from .memory_l8_head import MemoryL8Head
from .memory_channel_ppm_head import MemoryChannelPPMHead
from .memory_ppm_head import MemoryPPMHead
from .memory_sep_head import MemorySeparableHead
from .memory_ppm_sep_head import MemoryPPMSepHead

__all__ = [
    'PSPHead', 'FCNHead', 'memory_head', 'MemoryGAPHead', 'MemoryGAPGuideHead', 'MemoryDownSampleHead',
    'MemoryDownSampWeightPoolingHead', 'MemoryGlobalContextHead', 'MemoryGlobalContextHead2', 'MemoryGlobalContextHead3',
    'MemoryHead2', 'MemoryL8Head', 'MemoryChannelPPMHead', 'MemoryPPMHead', 'MemorySeparableHead', 'MemoryPPMSepHead'
]
