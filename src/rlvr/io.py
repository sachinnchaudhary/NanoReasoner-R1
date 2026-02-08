import json  
import random
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional  


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line) 


def reservoir_sample_jsonl(path: Path, k: int, seed: int = 0) -> List[Dict[str, Any]]:
    """
    Uniform sample of k examples from a JSONL file in one pass.
    """
    rng = random.Random(seed)
    sample: List[Dict[str, Any]] = []
    n = 0
    for ex in iter_jsonl(path):
        n += 1
        if len(sample) < k:
            sample.append(ex)
        else:
            j = rng.randrange(n)
            if j < k:
                sample[j] = ex
    return sample 


