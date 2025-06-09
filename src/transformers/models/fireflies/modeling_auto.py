from .fireflies.modeling_fireflies import FirefliesModel
from .fireflies.configuration_fireflies import FirefliesConfig

MODEL_MAPPING.register(FirefliesConfig, FirefliesModel)
