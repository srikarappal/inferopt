"""Check a technique DAG is well-formed before anything tries to traverse it.

    python validate_dag.py dag/llm.json

pydantic validates each node's fields; networkx validates the graph properties
the traversal depends on. Both are already standard, in-memory, and local -- no
scheduler, no server, no FSM framework. The orchestrator that consumes this is a
plain loop: apply, measure, keep or revert, follow the edge.

HISTORY

  This file exists because a typo used to be silent. A predicate referencing a
  misspelled attribute evaluated falsy, which reads as "this node does not
  apply" -- so the node never ran, for the rest of the project, with no error
  anywhere. Every expression is now parsed AND resolved against the pydantic
  schema here, before a single GPU-minute is spent.

  The worst-path launch count is checked against the DAG's own budget guard for
  the same reason: a DAG that can exceed its guard will stop mid-traversal and
  emit a partial frontier, which looks like a completed run.
"""

from __future__ import annotations

import json
import sys
from typing import Literal

import networkx as nx
from predicates import Predicate, PredicateError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

NodeClass = Literal["root", "lossless", "lossy", "checkpoint", "terminal"]


class Node(BaseModel):
    # extra="allow" so prose fields (rationale, note, todo, diagnostics) ride
    # along without needing to be declared -- they are for humans, not the
    # traversal, but they belong in the same file as the thing they explain.
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    node_class: NodeClass = Field(alias="class")
    status: Literal["active", "todo"] = "active"
    title: str | None = None
    requires: list[str] = []
    applicable_when: str | None = None
    action: dict | None = None
    sweep: list[dict] | None = None
    probes: list[str] = []
    quality_benchmarks: list[str] = []
    cost_launches: int = 0
    scenario: str | None = None
    on_keep: list[str] = []
    on_revert: list[str] = []


def build(dag: dict) -> tuple[dict[str, Node], nx.DiGraph, list[str]]:
    errs: list[str] = []
    nodes: dict[str, Node] = {}
    for raw in dag["nodes"]:
        try:
            n = Node(**raw)
            nodes[n.id] = n
        except ValidationError as e:
            for d in e.errors():
                errs.append(f"{raw.get('id', '?')}: {'.'.join(map(str, d['loc']))} — {d['msg']}")

    g = nx.DiGraph()
    for nid, n in nodes.items():
        g.add_node(nid, node=n)
    for nid, n in nodes.items():
        for succ in n.on_keep + n.on_revert:
            if succ not in nodes:
                errs.append(f"{nid}: edge to unknown node {succ!r}")
            else:
                g.add_edge(nid, succ)
        for req in n.requires:
            if req not in nodes:
                errs.append(f"{nid}: requires unknown node {req!r}")
    return nodes, g, errs


def main(path: str) -> int:
    dag = json.loads(open(path).read())
    nodes, g, errs = build(dag)
    warns: list[str] = []

    roots = [n.id for n in nodes.values() if n.node_class == "root"]
    terminals = [n.id for n in nodes.values() if n.node_class == "terminal"]
    if len(roots) != 1:
        errs.append(f"expected exactly one root node, found {roots}")
    if not terminals:
        errs.append("no terminal node — traversal has nowhere to end")

    # --- graph properties (networkx does the work) ---
    if not nx.is_directed_acyclic_graph(g):
        for cyc in list(nx.simple_cycles(g))[:5]:
            errs.append(f"cycle: {' -> '.join(cyc + [cyc[0]])}")

    root = roots[0] if roots else None
    if root and nx.is_directed_acyclic_graph(g):
        reachable = nx.descendants(g, root) | {root}
        for nid in set(g.nodes) - reachable:
            msg = f"{nid}: unreachable from {root}"
            (warns if nodes[nid].status == "todo" else errs).append(msg)

        # Fan-in > 1 needs no warning: a greedy walk takes one edge per node,
        # so a diamond (keep and revert rejoining) can never visit a node twice.
        # What CAN strand the traversal is a node with no way out.
        for nid, n in nodes.items():
            if n.node_class in ("terminal",):
                continue
            if not (n.on_keep or n.on_revert):
                errs.append(f"{nid}: no outgoing edge — traversal dead-ends here")
            if n.status == "active" and not n.on_keep:
                errs.append(f"{nid}: active node with no on_keep edge")

        # THE budget number. A greedy traversal follows ONE path root->terminal,
        # taking on_keep or on_revert at each node -- it never visits every node.
        # Summing all nodes (what a naive check does) overstates the cost.
        # Cost is reported per customer shape. A non-LoRA run never executes the
        # multi_lora nodes, so charging them to every customer overstates the
        # common case and hides that multi-LoRA is genuinely more expensive.
        active_guard = {n.id for n in nodes.values() if n.status == "active"}
        scenarios = {
            "default": lambda n: n.scenario is None,
            "multi_lora": lambda n: True,
        }
        worst: dict[str, tuple] = {}
        for sname, keep in scenarios.items():
            wp, wc = None, 0
            for term in terminals:
                for pth in nx.all_simple_paths(g, root, term):
                    cost = sum(nodes[x].cost_launches for x in pth
                               if x in active_guard and keep(nodes[x]))
                    if cost > wc:
                        wp, wc = pth, cost
            worst[sname] = (wp, wc)
        worst_path, worst_cost = worst["multi_lora"]
    else:
        worst_path, worst_cost = None, 0

    # --- predicates: parse and resolve every path against the schema ---
    # This is the check that stops a typo from silently disabling a node. A bad
    # expression must fail here, not evaluate falsy at hour two of a run.
    node_ids = set(nodes)
    n_exprs = 0
    for n in nodes.values():
        exprs = []
        if n.applicable_when:
            exprs.append(("applicable_when", n.applicable_when))
        for k, v in ((n.action or {}).get("set") or {}).items():
            if isinstance(v, str) and not v.replace("_", "").isalnum():
                exprs.append((f"action.set.{k}", v))
        for i, sw in enumerate(n.sweep or []):
            for k, v in sw.items():
                if isinstance(v, str) and any(c in v for c in "+-*/.()"):
                    exprs.append((f"sweep[{i}].{k}", v))
        for where, e in exprs:
            n_exprs += 1
            try:
                for bad in Predicate(e).check(node_ids):
                    errs.append(f"{n.id}.{where}: {bad}")
            except PredicateError as ex:
                errs.append(f"{n.id}.{where}: {ex}")

    # --- probe discipline ---
    for n in nodes.values():
        if n.status != "active":
            continue
        if n.node_class == "lossless" and "equivalence" not in n.probes:
            errs.append(f"{n.id}: lossless node without an equivalence probe")
        if n.node_class == "lossy":
            if "quality" not in n.probes:
                errs.append(f"{n.id}: lossy node without a quality probe")
            if not n.quality_benchmarks:
                errs.append(f"{n.id}: lossy node names no quality benchmarks")
        if n.node_class == "lossless" and "quality" in n.probes:
            warns.append(f"{n.id}: lossless node runs a quality benchmark — "
                         f"equivalence is stronger and far cheaper")
        if n.node_class in ("lossless", "lossy") and "goodput" not in n.probes:
            errs.append(f"{n.id}: no goodput probe — nothing to decide keep/revert on")

    # --- report ---
    active = [n for n in nodes.values() if n.status == "active"]
    todo = [n for n in nodes.values() if n.status == "todo"]
    lossy = [n for n in active if n.node_class == "lossy"]
    guard = dag["traversal"]["budget_guard"]["max_launches"]

    print(f"{path}   modality={dag['modality']}  v{dag['version']}\n")
    print(f"  nodes            {len(nodes)}  ({len(active)} active, {len(todo)} todo)")
    print(f"  edges            {g.number_of_edges()}")
    print(f"  acyclic          {nx.is_directed_acyclic_graph(g)}")
    print(f"  expressions      {n_exprs} parsed and schema-checked")
    if worst_path:
        for sname, (wp, wc) in worst.items():
            mins = wc * 8 + len(lossy) * 5
            lim = guard[sname] if isinstance(guard, dict) else guard
            flag = "" if wc <= lim else f"  << EXCEEDS {lim}"
            print(f"  worst path [{sname:10s}] {wc:2d} launches  "
                  f"~{mins:3d} min (~{mins/60:.1f}h)   guard {lim}{flag}")
            if wc > lim:
                errs.append(f"{sname}: worst path {wc} launches exceeds guard {lim}")
        print(f"\n  longest route    {' -> '.join(worst_path)}")
    print()

    for w in warns:
        print(f"  WARN  {w}")
    for e in errs:
        print(f"  FAIL  {e}")
    print()
    print("  DAG is well-formed." if not errs else f"  {len(errs)} problem(s).")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "dag/llm.json"))
