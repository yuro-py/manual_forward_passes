# Practicing manual forward passes on random small huggingface language models.

Use this to convert weights of a downloaded huggingface model from safetensors to JSON readable format in the current directory.

python -c "
import json
path = '~/.cache/huggingface/hub/models--{model name}/blobs/{full file name}'
with open(path, 'rb') as f:
    n = int.from_bytes(f.read(8), 'little')
    print(json.dumps(json.loads(f.read(n)), indent=2))
" > {preferable model name}.json
