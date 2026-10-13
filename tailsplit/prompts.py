"""Chat prompts of exactly 256 template tokens, built offline from each model's own tokenizer.

The same prompts are used in both conditions and by both servers. Prompt i asks for a long
sequence of integers and is padded with the word ' number' until the chat template (with the
generation prompt) is 256 tokens long.
"""
from pathlib import Path

from .spec import MODELS, PROMPT_COUNT, PROMPT_TOKENS, SCHEMA, sha, snapshot, write_json


def construct(model, hf_home, output, tokenizer=None):
    model_path = snapshot(hf_home, model)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    prompts = []
    for index in range(PROMPT_COUNT):
        prefix = f'Request {index:03d}. Write a long sequence of integers. Continue until the response limit.'
        found = None
        for n in range(300):
            candidate = [{'role': 'user', 'content': prefix + ' number' * n}]
            ids = tokenizer.apply_chat_template(candidate, tokenize=True, add_generation_prompt=True,
                                                return_dict=False)
            if len(ids) == PROMPT_TOKENS:
                found = {'messages': candidate, 'prompt_token_ids': list(ids)}
                break
        if found is None:
            raise RuntimeError(f'Cannot build a {PROMPT_TOKENS}-token prompt for {model}')
        prompts.append(found)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = {p.name: sha(p) for p in sorted(Path(model_path).iterdir())
             if p.is_file() and p.suffix in ('.json', '.jinja', '.txt')}
    path = output / f'{model}.json'
    write_json(path, {'model': model, 'repo': MODELS[model][0], 'revision': MODELS[model][1],
                      'schema': SCHEMA, 'prompts': prompts, 'tokenizer_files': files})
    return path
