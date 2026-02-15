from .llm import LLM
from omegaconf import OmegaConf, DictConfig


def build_llm(config: DictConfig) -> LLM:
    llm = LLM(config)
    return llm