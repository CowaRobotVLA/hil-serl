# from .continuous.bc import BCAgent
# from .continuous.sac import SACAgent
# from .continuous.sac_hybrid_single import SACAgentHybridSingleArm
from .continuous.pi05 import PI05Agent
# from .continuous.sac_hybrid_dual import SACAgentHybridDualArm

agents = {
    # "bc": BCAgent,
    # "sac": SACAgent,
    # "sac_hybrid_single": SACAgentHybridSingleArm,
    "pi05": PI05Agent,
    # "sac_hybrid_dual": SACAgentHybridDualArm,
}
