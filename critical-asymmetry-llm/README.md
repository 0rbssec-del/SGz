# Critical Asymmetry LLM

Architecture pour maintenir un LLM (texte + vision) dans la zone critique
d'asymetrie informationnelle A ~ 0.5.

## Installation
    pip install -e .

## Usage
    from src.agents.multimodal_agent import MultiModalCAAAgent
    agent = MultiModalCAAAgent(encoder, A_star=0.5)
    result = agent.step(input_ids, attention_mask, pixel_values, pixel_values_private)
    print(result["A_lisse"])

## Tests
    pytest tests/ -v

## Docker
    make docker-build && make docker-up

## Licence
MIT
