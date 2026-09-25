"""
LLM-editable term vocabulary: custom features as sandboxed math expressions.

Fixes failure mechanism F1 (closed hypothesis vocabulary) without giving the
LLM code execution. A custom term is a string expression over:

    d['key']     design values          e['key']    equipment values (str)
    env['key']   environment values     numbers, + - * / ( )
    min max abs exp log sqrt sigmoid    step(x) = 1 if x >= 0 else 0
    comparisons (a >= b etc.) evaluate to 1.0 / 0.0

Example: "sigmoid((d['camera_def'] - 15) / 3) * step(d['engine_def'] - 10)"

Compilation walks the AST against a strict whitelist (no attributes, no
names besides d/e/env, no calls besides the table above). A compiled term is
then probed on sample points: every value must be finite and |v| <= 10, else
it is rejected — the same normalisation contract every built-in term obeys.
"""

import ast
import math
from typing import Any, Callable, Dict, List, Sequence

_FUNCS: Dict[str, Callable] = {
    "min": min, "max": max, "abs": abs,
    "exp": lambda x: math.exp(min(x, 60.0)),
    "log": lambda x: math.log(max(x, 1e-9)),
    "sqrt": lambda x: math.sqrt(max(x, 0.0)),
    "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-max(min(x, 60.0), -60.0))),
    "step": lambda x: 1.0 if x >= 0 else 0.0,
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Call,
    ast.Subscript, ast.Name, ast.Load, ast.Compare, ast.IfExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
    ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
    ast.Index,  # py<3.9 compat in some environments
)


class ExprError(ValueError):
    pass


def _validate(node: ast.AST) -> None:
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_NODES):
            raise ExprError(f"disallowed syntax: {type(child).__name__}")
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name) or child.func.id not in _FUNCS:
                raise ExprError("only min/max/abs/exp/log/sqrt/sigmoid/step calls allowed")
            if child.keywords:
                raise ExprError("keyword arguments not allowed")
        if isinstance(child, ast.Name) and child.id not in ("d", "e", "env", *_FUNCS):
            raise ExprError(f"unknown name {child.id!r}")
        if isinstance(child, ast.Subscript):
            if not (isinstance(child.value, ast.Name)
                    and child.value.id in ("d", "e", "env")):
                raise ExprError("subscripts only on d/e/env")
        if isinstance(child, ast.Constant) and not isinstance(child.value, (int, float, str)):
            raise ExprError("only number/string constants allowed")


def compile_expr(expr: str) -> Callable[[Dict, Dict], float]:
    """Compile a custom-term expression to a (design, equipment) -> float."""
    if len(expr) > 400:
        raise ExprError("expression too long")
    tree = ast.parse(expr, mode="eval")
    _validate(tree)
    code = compile(tree, "<custom-term>", "eval")

    class _NumDict(dict):
        def __missing__(self, key):
            return 0.0

    def fn(d: Dict[str, Any], e: Dict[str, Any]) -> float:
        env = _NumDict({k[4:]: v for k, v in e.items()
                        if k.startswith("env_") and isinstance(v, (int, float))})
        dd = _NumDict({k: float(v) for k, v in d.items()
                       if isinstance(v, (int, float))})
        # equipment compares as strings; unknown keys -> ""
        class _StrDict(dict):
            def __missing__(self, key):
                return ""
        ee = _StrDict({k: str(v) for k, v in e.items()
                       if not k.startswith("env_")})
        try:
            out = eval(code, {"__builtins__": {}}, {"d": dd, "e": ee, "env": env,
                                                    **_FUNCS})
        except Exception as exc:
            raise ExprError(f"evaluation failed: {exc}")
        if isinstance(out, bool):
            return 1.0 if out else 0.0
        return float(out)

    return fn


def check_bounded(fn: Callable, probes: Sequence[tuple], limit: float = 10.0) -> None:
    """Reject expressions that blow up on probe (design, equipment) points."""
    for d, e in probes:
        v = fn(d, e)
        if not math.isfinite(v) or abs(v) > limit:
            raise ExprError(f"unbounded value {v!r} on probe point - normalise "
                            f"the expression to roughly [-1, 1]")
