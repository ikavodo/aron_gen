import os
from .core.AronsonSet import AronsonSet, Direction


def run_generation(n):
    dir_name = f"aron_gen_data_iters={n}"
    os.makedirs(dir_name, exist_ok=True)

    directions = [
        (Direction.FORWARD, 'forward'),
        (Direction.BACKWARD, 'backward')
    ]

    elements = {'forward': [], 'backward': []}
    stats = {'forward': {}, 'backward': {}}

    for direction, dir_key in directions:
        aset = AronsonSet('t', direction)
        try:
            aset.generate_full(n)
        except Exception as e:
            print(f"Warning: Generation stopped due to {e}")

        dir_elements = []
        for iter_num in range(n + 1):
            sorted_seqs = sorted([list(seq) for seq in aset[iter_num]])
            dir_elements.extend(sorted_seqs)
            elements[dir_key] = dir_elements
            stats[dir_key][iter_num] = len(aset[iter_num]) + (
                0 if not iter_num else stats[dir_key][iter_num - 1]
            )

    with open(f'{dir_name}/ground_truth_seqs.py', 'w') as f:
        f.write(f"# Elements corresponding to all Aronson sequences of length up to n={n}\n")
        f.write(f"forward_elems = [\n")
        for elem in elements['forward']:
            f.write(f"    {elem},\n")
        f.write("]\n\n")
        f.write(f"backward_elems = [\n")
        for elem in elements['backward']:
            f.write(f"    {elem},\n")
        f.write("]\n")

    with open(f'{dir_name}/ground_truth_stats.py', 'w') as f:
        f.write("# Ground truth for number of sets per iteration\n")
        f.write(f"forward_stats = {stats['forward']}\n")
        f.write(f"backward_stats = {stats['backward']}\n")
