"""Evaluate DAG predicates, and check them against the schema before running.

    Predicate("workload.p99_input_tokens > 1024").evaluate(ctx)   ->  True
    Predicate("workload.p99_input").check(node_ids)               ->  ["...no field 'p99_input'..."]

Two jobs, and the second is the one that matters. A predicate is a string in a
JSON file; a typo in it does not crash, it evaluates falsy, and a node quietly
never runs for the rest of the project. `check()` resolves every dotted path
against the pydantic schema WITHOUT needing a live fingerprint, so
validate_dag.py catches that at parse time instead of at hour two.

Restricted AST, not eval: comparisons, boolean ops, arithmetic, literals,
attribute paths, and a short allowlist of functions. Anything else -- calls,
subscripts, comprehensions, lambdas, imports -- is a parse error rather than
something clever.

HISTORY

  A typo used to disable a node silently. An unknown attribute path evaluated
  falsy, so `fingerprint.model.has_fp8_checkpiont` simply meant "this node never
  runs" -- for the rest of the project, with no error. Every path is now
  resolved against the pydantic schema at VALIDATION time, so a typo fails at
  parse.

  The evaluator is a restricted AST walker rather than eval(). It admits
  comparisons, boolean and arithmetic operators, attribute paths and a short
  function allowlist. It rejects __import__, comprehensions, calls to anything
  unlisted, and references to node ids the DAG does not define.
"""

from __future__ import annotations

import ast
import math
import typing
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from fingerprint import Context, NodeMeasurement

ALLOWED_FUNCS: dict[str, Any] = {
    "ceil": lambda x, to=1: math.ceil(x / to) * to,
    "floor": lambda x, to=1: math.floor(x / to) * to,
    "min": min, "max": max, "abs": abs, "round": round,
    "int": int, "float": float, "len": len,
}

ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Constant, ast.Name, ast.Attribute, ast.Call, ast.Load, ast.Tuple, ast.List,
    ast.IfExp,
)

# Roots a path may start from. Everything reachable from Context.
ROOTS = ("fingerprint", "workload", "slo", "measurements", "preconditions",
         "incumbent", "accept_band")


class PredicateError(ValueError):
    pass


# --------------------------------------------------------------------------
# static path checking against the pydantic schema
# --------------------------------------------------------------------------

def _unwrap_optional(tp: Any) -> Any:
    """`float | None` -> `float`. Optionality is not a path concern."""
    if get_origin(tp) in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _member_type(cls: Any, name: str) -> Any:
    if isinstance(cls, type) and issubclass(cls, BaseModel):
        if name in cls.model_fields:
            return _unwrap_optional(cls.model_fields[name].annotation)
        if name in cls.model_computed_fields:
            return _unwrap_optional(cls.model_computed_fields[name].return_type)
        known = sorted(set(cls.model_fields) | set(cls.model_computed_fields))
        raise PredicateError(f"{cls.__name__} has no field {name!r}. Known: {', '.join(known)}")
    raise PredicateError(f"cannot descend into {getattr(cls, '__name__', cls)} looking for {name!r}")


def resolve_path_type(path: str, node_ids: set[str] | None = None) -> Any:
    """Walk a dotted path through the schema. Raises PredicateError with the
    field that failed and what was available instead."""
    parts = path.split(".")
    if parts[0] not in ROOTS:
        raise PredicateError(f"{path}: unknown root {parts[0]!r}. Must start with one of {', '.join(ROOTS)}")

    cur: Any = _member_type(Context, parts[0])
    for i, part in enumerate(parts[1:], start=1):
        origin = get_origin(cur)
        if origin is dict:
            key_is_node = get_args(cur)[1] is NodeMeasurement
            if key_is_node and node_ids is not None and part not in node_ids:
                raise PredicateError(
                    f"{path}: measurements refers to {part!r}, which is not a node in this DAG"
                )
            cur = _unwrap_optional(get_args(cur)[1])
            continue
        if cur is Any:
            return Any            # incumbent.* and preconditions.<free-form>.*
        try:
            cur = _member_type(cur, part)
        except PredicateError as e:
            raise PredicateError(f"{path}: {e}") from None
    return cur


# --------------------------------------------------------------------------
# the predicate itself
# --------------------------------------------------------------------------

class Predicate:
    def __init__(self, expr: str):
        self.expr = expr
        try:
            self.tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise PredicateError(f"{expr!r}: {e.msg}") from None
        for n in ast.walk(self.tree):
            if not isinstance(n, ALLOWED_NODES):
                raise PredicateError(
                    f"{expr!r}: {type(n).__name__} is not permitted in a predicate"
                )
            if isinstance(n, ast.Call):
                if not isinstance(n.func, ast.Name) or n.func.id not in ALLOWED_FUNCS:
                    fn = getattr(n.func, "id", ast.dump(n.func))
                    raise PredicateError(
                        f"{expr!r}: {fn}() is not on the allowlist ({', '.join(sorted(ALLOWED_FUNCS))})"
                    )

    def paths(self) -> set[str]:
        """Dotted paths referenced, longest form only (`a.b.c`, not `a` and `a.b`)."""
        found: set[str] = set()

        def flatten(node: ast.AST) -> str | None:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                base = flatten(node.value)
                return f"{base}.{node.attr}" if base else None
            return None

        for n in ast.walk(self.tree):
            if isinstance(n, ast.Attribute):
                p = flatten(n)
                if p and p.split(".")[0] in ROOTS:
                    found.add(p)
            elif isinstance(n, ast.Name) and n.id in ROOTS:
                found.add(n.id)
        # drop prefixes that are strictly contained in a longer path
        return {p for p in found if not any(q != p and q.startswith(p + ".") for q in found)}

    def check(self, node_ids: set[str] | None = None) -> list[str]:
        errs = []
        for p in sorted(self.paths()):
            try:
                resolve_path_type(p, node_ids)
            except PredicateError as e:
                errs.append(str(e))
        return errs

    def evaluate(self, ctx: Context) -> Any:
        return _Eval(ctx).visit(self.tree.body)

    def __repr__(self) -> str:
        return f"Predicate({self.expr!r})"


class _Eval:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    def visit(self, n: ast.AST) -> Any:
        m = getattr(self, f"v_{type(n).__name__}", None)
        if m is None:
            raise PredicateError(f"cannot evaluate {type(n).__name__}")
        return m(n)

    def v_Constant(self, n): return n.value
    def v_Tuple(self, n): return tuple(self.visit(e) for e in n.elts)
    def v_List(self, n): return [self.visit(e) for e in n.elts]

    def v_Name(self, n):
        if n.id not in ROOTS:
            raise PredicateError(f"unknown name {n.id!r}")
        return getattr(self.ctx, n.id)

    def v_Attribute(self, n):
        base = self.visit(n.value)
        if isinstance(base, dict):
            if n.attr not in base:
                raise PredicateError(f"no key {n.attr!r} in {list(base)[:6]}")
            return base[n.attr]
        if not hasattr(base, n.attr):
            raise PredicateError(f"{type(base).__name__} has no field {n.attr!r}")
        return getattr(base, n.attr)

    def v_BoolOp(self, n):
        vals = (self.visit(v) for v in n.values)
        return all(vals) if isinstance(n.op, ast.And) else any(vals)

    def v_UnaryOp(self, n):
        v = self.visit(n.operand)
        return (not v) if isinstance(n.op, ast.Not) else -v

    def v_BinOp(self, n):
        a, b = self.visit(n.left), self.visit(n.right)
        return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                ast.Div: lambda: a / b, ast.FloorDiv: lambda: a // b,
                ast.Mod: lambda: a % b, ast.Pow: lambda: a ** b}[type(n.op)]()

    def v_Compare(self, n):
        left = self.visit(n.left)
        for op, comp in zip(n.ops, n.comparators):
            right = self.visit(comp)
            ok = {ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right,
                  ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right,
                  ast.Gt: lambda: left > right, ast.GtE: lambda: left >= right,
                  ast.In: lambda: left in right, ast.NotIn: lambda: left not in right}[type(op)]()
            if not ok:
                return False
            left = right
        return True

    def v_IfExp(self, n):
        return self.visit(n.body) if self.visit(n.test) else self.visit(n.orelse)

    def v_Call(self, n):
        return ALLOWED_FUNCS[n.func.id](*(self.visit(a) for a in n.args))
