# llm-template
A little project that allows building your own untrained models based on GPT architecture.

Building an architecture with your own sizes is as simple as creating your own config file.
## Example 
```Python
from llm_template.llm.build import build_llm

config_file: str | Path = ...
llm = build_llm(config_file)
```
## Functionality
At the moment the project allows to:
- Choose between different attention mechanisms (MHA, GQA, MQA)
- Create a FFN layer with any activation function that PyTorch has
- Use your own RoPE parameters

Planned work:
- Add Mixture-of-Experts support
- Add ability to load waights from Hugging Face for existing models
