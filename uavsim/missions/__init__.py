"""Mission modules — structured flight tasks beyond simple demos."""

from uavsim.missions.transport import (
    MissionPhase,
    TransportMission,
    generate_slung_load_mjcf,
    transport_hover_gains,
)
from uavsim.missions.cooperative_transport import (
    CoopPhase,
    CooperativeTransportMission,
    MultiVehicleSimAdapter,
    cooperative_hover_gains,
    equilateral_offsets,
    generate_cooperative_mjcf,
)

__all__ = [
    "MissionPhase",
    "TransportMission",
    "generate_slung_load_mjcf",
    "transport_hover_gains",
    "CoopPhase",
    "CooperativeTransportMission",
    "MultiVehicleSimAdapter",
    "cooperative_hover_gains",
    "equilateral_offsets",
    "generate_cooperative_mjcf",
]
