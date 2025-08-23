from ..rt_detr.modeling_rt_detr import (RTDetrConvNormLayer)
from ..d_fine.modular_d_fine import DFineSCDown,DFineCSPRepLayer,DFineRepNCSPELAN4
from ..d_fine.modeling_d_fine import DFineRepVggBlock,DFineEncoder,DFineHybridEncoder

class DEIMConvNormLayer(RTDetrConvNormLayer):
    pass

class DEIMSCDown(DFineSCDown):
    pass

class DEIMCSPRepLayer(DFineCSPRepLayer):
    pass
    
class DEIMRepNCSPELAN4(DFineRepNCSPELAN4):
    pass

class DEIMRepVggBlock(DFineRepVggBlock):
    pass

class DEIMEncoder(DFineEncoder):
    pass

class DEIMHybridEncoder(DFineHybridEncoder):
    pass


