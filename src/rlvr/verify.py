import re
from typing import Dict, Any, Tuple

INT_RE = re.compile(r"^[+-]?\d+$")   


def _norm_ws(s: str) -> str:
    return " ".join((s or "").strip().split()) 

def _extract_answer(text: str) -> str:
    lower = text.lower()
    start = lower.rfind("<answer>")
    end = lower.rfind("</answer>")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start + len("<answer>") : end].strip() 

def verify_numeric(ex: Dict[str, Any], model_out: str) -> Tuple[float, bool]:
    pred = _norm_ws(_extract_answer(model_out)).replace(",", "")
    gold = str(ex["answer"]).strip().replace(",", "")
    if not INT_RE.fullmatch(pred):
        return 0.0, False
    return (1.0, True) if pred == gold else (0.0, True)  

def verify_mcq(ex: Dict[str, Any], model_out: str) -> Tuple[float, bool]:
    pred = _norm_ws(_extract_answer(model_out)).upper()
    gold = str(ex["answer"]).strip().upper()
    if pred not in {"A","B","C","D"}:
        return 0.0, False
    return (1.0, True) if pred == gold else (0.0, True) 


def _norm_fact(s: str) -> str:
    s = _norm_ws(s).lower()
    s = s.rstrip(".").strip()
    return s 

def verify_short_text(ex: Dict[str, Any], model_out: str) -> Tuple[float, bool]:
    pred = _norm_fact(_extract_answer(model_out))
    gold = ex.get("target_norm") or _norm_fact(ex.get("target", ""))
    if not pred:
        return 0.0, False
    return (1.0, True) if pred == gold else (0.0, True)  
    