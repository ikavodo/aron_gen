from pathlib import Path

from .core.AronsonSet import AronsonSet, Direction

BASE_DIR = Path(__file__).resolve().parents[2]  # parent of parent
DATA_DIR = BASE_DIR / "data"


def run_generation(n):
    dir_name = DATA_DIR / f"iters_{n}"
    dir_name.mkdir(parents=True, exist_ok=True)

    directions = [
        (Direction.FORWARD, 'forward'),
        (Direction.BACKWARD, 'backward')
    ]

    elements = {'forward': [], 'backward': [], 'intersect': []}
    stats = {'forward': {}, 'backward': {}, 'intersect': {}}

    # Collect sequences per direction
    seq_sets = {}
    for direction, dir_key in directions:
        aset = AronsonSet('t', direction)
        try:
            # faster than not checking semantics
            aset.generate_full(n)
        except Exception as e:
            print(f"Warning: Generation stopped due to {e}")

        dir_elements = []
        iter_stats = {}
        for iter_num in range(n + 1):
            sorted_seqs = sorted([list(seq) for seq in aset[iter_num]])
            dir_elements.extend(sorted_seqs)
            iter_stats[iter_num] = len(aset[iter_num]) + (
                0 if not iter_num else iter_stats[iter_num - 1]
            )
        elements[dir_key] = dir_elements
        stats[dir_key] = iter_stats
        seq_sets[dir_key] = [set(tuple(seq) for seq in aset[i]) for i in range(n + 1)]

    # Compute intersection per iteration
    intersect_elements = []
    intersect_stats_iter = {}
    for i in range(n + 1):
        intersect_i = seq_sets['forward'][i] & seq_sets['backward'][i]
        intersect_elements.extend([list(seq) for seq in sorted(intersect_i)])
        intersect_stats_iter[i] = len(intersect_i)
    elements['intersect'] = intersect_elements
    stats['intersect'] = intersect_stats_iter

    # Write sequences to seqs.py
    with open(dir_name / 'seqs.py', 'w') as f:
        f.write(f"# Elements corresponding to all Aronson sequences of length up to n={n}\n")
        for key in ['forward', 'backward', 'intersect']:
            f.write(f"{key}_elems = [\n")
            for elem in elements[key]:
                f.write(f"    {elem},\n")
            f.write("]\n\n")

    # Write stats to stats.py
    with open(dir_name / 'stats.py', 'w') as f:
        f.write("# Ground truth for number of sets per iteration\n")
        for key in ['forward', 'backward', 'intersect']:
            f.write(f"{key}_stats = {stats[key]}\n")
