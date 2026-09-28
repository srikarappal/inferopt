"""Write the estimate-only AIConfigurator systems from inferopt.cards.CARDS.

    python tools/write_aic_systems.py        -> src/inferopt/aic_systems/*.yaml

Run after changing a card; tests/test_cards.py fails while the files and the
table disagree.
"""

from inferopt import cards

if __name__ == "__main__":
    written = cards.write_systems()
    print(f"{len(written)} systems in {cards.systems_dir}")
