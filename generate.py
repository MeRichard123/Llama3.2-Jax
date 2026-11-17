"""Load a Hugging Face Llama 3.2 checkpoint and generate a short completion.

Example: .env/bin/python generate.py --tokenizer tokenizer.model --prompt 'Once upon a time'
Add --chat for an instruction-tuned checkpoint, or JAX_PLATFORMS=cuda to require CUDA.

Python: ``from generate import generate; text = generate('Once upon a time')``
"""
import os
import argparse
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")
import jax
import jax.numpy as jnp
from flax import nnx

from model import (LLAMA32_CONFIG_1B, LLAMA32_CONFIG_3B, Llama3Model,
                   compute_rope_params, generate as generate_tokens)
from tokenizer import ChatFormat, Llama3Tokenizer
from quantisation import QUANTISATIONS 

def generate(prompt, max_new_tokens=32, *, model=None, tokenizer=None,
             weights_path=Path(__file__).with_name('model.safetensors'),
             tokenizer_path=Path(__file__).with_name('tokenizer.model'),
             context_size=None, chat=True, temperature=0.0, top_k=None,
             key=None, quantisation=None, use_cache=True):
    """Return generated text without parsing arguments or printing.

    Loads the local checkpoint/tokenizer when omitted. Pass ``model`` and
    ``tokenizer`` to reuse them across calls. Context defaults to the loaded
    model's limit, or 512 when loading a checkpoint here. Plain completions
    include the prompt; chat completions contain only the assistant response.
    Sampling (temperature > 0) requires a JAX PRNG ``key``.
    quantisation is forwarded to load_model when loading a checkpoint here;
    a supplied model retains its existing quantisation.
    use_cache enables fixed-shape prefill and KV-cached decoding by default.
    """
    if context_size is None:
        context_size = model.cfg['context_length'] if model is not None else 512
    if context_size <= 0:
        raise ValueError('context_size must be positive')
    if max_new_tokens < 0:
        raise ValueError('max_new_tokens must be nonnegative')
    if model is not None and context_size > model.cfg['context_length']:
        raise ValueError('context_size must be within the model context length')
    if tokenizer is None:
        tokenizer = Llama3Tokenizer(str(tokenizer_path))
    ids = (ChatFormat(tokenizer).encode(prompt) if chat
           else tokenizer.encode(prompt, bos=True))
    if len(ids) + max_new_tokens > context_size:
        raise ValueError('Prompt plus generated tokens exceeds context_size')
    if model is None:
        model = load_model(weights_path, context_size, quantisation=quantisation)
    stop = tokenizer.special['<|eot_id|>' if chat else '<|end_of_text|>']
    result = generate_tokens(model, jnp.array([ids], dtype=jnp.int32),
                             max_new_tokens, context_size, temperature=temperature,
                             top_k=top_k, eos_id=stop, key=key, use_cache=use_cache)
    completion = jax.device_get(result[0, len(ids):]).tolist()
    if completion and completion[-1] == stop:
        completion.pop()
    text = tokenizer.decode(completion)
    return text if chat else prompt + text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=Path(__file__).with_name('model.safetensors'))
    parser.add_argument('--tokenizer', type=Path, default=Path(__file__).with_name('tokenizer.model'))
    parser.add_argument('--prompt', default='Once upon a time')
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--context-size', type=int, default=512)
    parser.add_argument('--kv-cache', action=argparse.BooleanOptionalAction, default=True,
                        help='Use fixed-shape prefill and cached decoding (default: enabled)')
    parser.add_argument('--quantisation', choices=tuple(QUANTISATIONS), default=None,
                        help='Weight quantisation for linear layers (default: BF16)')
    parser.add_argument('--chat', action='store_true', help='Use the Llama instruction chat template')
    args = parser.parse_args()
    if not args.tokenizer.is_file():
        parser.error(f'Tokenizer vocabulary not found: {args.tokenizer}; pass --tokenizer PATH')
    if not args.weights.is_file():
        parser.error(f'Checkpoint not found: {args.weights}')
    if args.max_new_tokens < 0:
        parser.error('--max-new-tokens must be nonnegative')
    print(f'Loading {args.weights} on {jax.devices()[0]}...', flush=True)
    try:
        text = generate(args.prompt, args.max_new_tokens,
                        weights_path=args.weights, tokenizer_path=args.tokenizer,
                        context_size=args.context_size, chat=args.chat,
                        quantisation=args.quantisation, use_cache=args.kv_cache)
    except ValueError as error:
        parser.error(str(error))
    print(text)


if __name__ == '__main__':
    main()
