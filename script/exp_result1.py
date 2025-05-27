import json
from pathlib import Path


with open(Path(__file__).resolve().parent / "../CONFIG.txt") as f:
    data = f.readlines()
d = [(i.split(': ')[0], i.split('; ')[1]) for i in data]
d = [(i, max([(idx, k) for idx, k in  enumerate(json.loads(j))], key=lambda x:x[1])) for i, j in d]
d = [(i, j) for i, j in d if not i.endswith("rand")]
print('\n'.join([f'{i}: epoch{j[0]}-{j[1]:.4f}' for i, j in d]))
