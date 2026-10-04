"""What a run's cards sit on, to read a measurement by: how the cards reach
each other, the PCIe link each one has, the power it may draw, the driver and
the CPU beside them.

Recorded, never decided on. Tensor parallel exchanges every layer's results
between the cards, so a frontier measured over NVLink and one measured
through the CPU's host bridge without peer to peer are different machines
however alike the cards are; before this every host with two or more cards
was written down as NVLink (4 Oct 2026). Each reading is nvidia-smi's own,
and a part it will not answer is None, not a guess.
"""

import re
import subprocess

ansi = re.compile(r"\x1b\[[0-9;]*m")
gpu_label = re.compile(r"GPU(\d+)")
card_fields = ("index", "pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current",
               "pcie.link.width.max", "power.limit", "power.max_limit", "driver_version")


def smi(*arguments) -> str:
    """nvidia-smi's answer, or empty when it will not give one."""
    try:
        done = subprocess.run(["nvidia-smi", *arguments], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout if done.returncode == 0 else ""


def matrix(text: str) -> dict[tuple[int, int], str]:
    """{(row card, column card): cell} from a topo matrix: the header is the
    line whose first two words are card labels, a row starts with one card
    label followed by a cell."""
    rows = [ansi.sub("", line).split() for line in text.splitlines() if line.strip()]
    header = next((tokens for tokens in rows if len(tokens) > 1
                   and gpu_label.fullmatch(tokens[0]) and gpu_label.fullmatch(tokens[1])), None)
    if header is None:
        return {}
    columns = [int(gpu_label.fullmatch(token).group(1)) for token in header if gpu_label.fullmatch(token)]
    cells = {}
    for tokens in rows:
        row = gpu_label.fullmatch(tokens[0]) if len(tokens) > 1 and not gpu_label.fullmatch(tokens[1]) else None
        if row is None:
            continue
        for column, cell in zip(columns, tokens[1:1 + len(columns)]):
            cells[(int(row.group(1)), column)] = cell
    return cells


def pairs(cards: list[int]) -> list[tuple[int, int]]:
    return [(a, b) for position, a in enumerate(cards) for b in cards[position + 1:]]


def number(text: str):
    """An int or float from a csv cell, None for [N/A] and the like."""
    text = text.strip()
    try:
        value = float(text)
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def card_links(cards: list[int]) -> tuple[list[dict] | None, str | None]:
    """(per card PCIe generation and width now and at most, power limit and its
    maximum; the driver), for the cards the run holds."""
    text = smi(f"--query-gpu={','.join(card_fields)}", "--format=csv,noheader,nounits")
    rows = [[cell.strip() for cell in line.split(",")] for line in text.splitlines() if line.strip()]
    rows = [row for row in rows if len(row) == len(card_fields) and number(row[0]) in cards]
    if not rows:
        return None, None
    links = [{"index": number(row[0]), "gen": number(row[1]), "gen_max": number(row[2]),
              "width": number(row[3]), "width_max": number(row[4]),
              "power_w": number(row[5]), "power_max_w": number(row[6])} for row in rows]
    return links, rows[0][7] or None


def cpu_model() -> str | None:
    """The host CPU's name: /proc/cpuinfo's on x86, lscpu's cores on Arm,
    where cpuinfo names only an implementer code (the GB10's read 0x41)."""
    try:
        with open("/proc/cpuinfo") as handle:
            found = re.search(r"^model name\s*:\s*(.+)$", handle.read(), re.M)
        if found:
            return found.group(1).strip()
    except OSError:
        pass
    try:
        text = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    names = list(dict.fromkeys(re.findall(r"^Model name:\s*(.+)$", text, re.M)))
    return " + ".join(name.strip() for name in names) or None


def read(cards: list[int]) -> dict:
    """The fingerprint fields for these cards (the indexes nvidia-smi shows):
    interconnect, gpu_paths, p2p, card_links, driver_version, cpu_model."""
    links, driver = card_links(cards)
    found = {"card_links": links, "driver_version": driver, "cpu_model": cpu_model(),
             "interconnect": None, "gpu_paths": None, "p2p": None}
    if len(cards) < 2:
        return found
    paths = matrix(smi("topo", "-m"))
    wanted = pairs(cards)
    if all(pair in paths for pair in wanted):
        found["gpu_paths"] = {f"{a}-{b}": paths[(a, b)] for a, b in wanted}
        found["interconnect"] = "nvlink" if all(paths[pair].startswith("NV") for pair in wanted) else "pcie"
    reads = matrix(smi("topo", "-p2p", "r"))
    if all(pair in reads for pair in wanted):
        found["p2p"] = all(reads[pair] == "OK" for pair in wanted)
    return found


def describe(hw) -> str:
    """One line for the run's log: the cards and what joins them."""
    parts = [f"{hw.gpu_count} x {hw.gpu_name}"]
    if hw.gpu_count > 1:
        paths = sorted(set((hw.gpu_paths or {}).values()))
        parts.append(f"over {hw.interconnect or 'an unknown link'}" + (f" ({', '.join(paths)})" if paths else ""))
        parts.append("peer to peer " + {True: "on", False: "off", None: "unknown"}[hw.p2p])
    # A unified memory part's card hangs off the CPU by NVLink-C2C, and the
    # PCIe link nvidia-smi gives it (x1, Gen1 on the GB10) is a placeholder:
    # kept in the record, left out of the line.
    links = [] if getattr(hw, "unified_memory", False) else (hw.card_links or [])
    if links:
        widths = sorted({f"x{link['width']} of x{link['width_max']}" for link in links if link.get("width")})
        gens = sorted({f"Gen{link['gen_max']}" for link in links if link.get("gen_max")})
        if widths:
            parts.append(f"PCIe {', '.join(widths)}" + (f" {', '.join(gens)}" if gens else ""))
        powers = sorted({f"{link['power_w']:g} W of {link['power_max_w']:g} W" for link in links
                         if link.get("power_w") and link.get("power_max_w")})
        if powers:
            parts.append(", ".join(powers))
    if hw.driver_version:
        parts.append(f"driver {hw.driver_version}")
    if hw.cpu_model:
        parts.append(hw.cpu_model)
    return "cards     " + ", ".join(parts)
