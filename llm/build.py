from .llm import LLM
from omegaconf import OmegaConf
from pathlib import Path


def build_llm(config_file: str | Path) -> LLM:
    config = OmegaConf.load(config_file)
    llm = LLM(config)
    return llm
