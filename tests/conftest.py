"""Put src/ on the path so the tests import the package under development.

Without this a test run picks up whatever `inferopt` is installed in the
environment, which is the wrong thing to test and fails confusingly when
nothing is installed at all. An editable install would also work; this makes
the checkout self-sufficient so `python tests/test_dag_unit.py` needs no setup.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
