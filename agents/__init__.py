from agents.flow_map_policy import FlowMapPolicy
from agents.fmq import FMQAgent

agents = dict(
    offline=FlowMapPolicy,
    fmq=FMQAgent,
)
