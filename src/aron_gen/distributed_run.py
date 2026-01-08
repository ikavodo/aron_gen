#!/usr/bin/env python3
import os
import argparse
import multiprocessing
from math import ceil, log2

from aron_gen.cli import run_generation
from aron_gen.core.AronsonSequence import VerificationError
from aron_gen.core.AronsonSet import AronsonSequence, AronsonSet, PRUNE_THRESH, ORD_INITIAL, ORD_TABLE

# Determine number of worker processes from Slurm or default to all CPUs
default_procs = multiprocessing.cpu_count()
n_procs = int(os.getenv('SLURM_CPUS_PER_TASK', default_procs))


# --------------------------------------------------
# Top-level helper functions for multiprocessing
# --------------------------------------------------

def is_valid_extension(elem, non_elements, current_perm):
    return not (
            elem in non_elements or
            (elem - 1 in current_perm and elem - 2 in current_perm) or
            (elem - 1 in current_perm and elem + 1 in current_perm) or  # middle
            (elem + 1 in current_perm and elem + 2 in current_perm)
    )

def generate_valid_seqs(elem, initial_remaining, iteration, cur_ord_key, non_elements,
                        letter, direction, error_rate):
    if not is_valid_extension(elem, non_elements, []):
        return
    current = [elem]
    remaining = set(initial_remaining)
    remaining.remove(elem)
    current_sum = elem

    def rec(current_sum, remaining):
        if len(current) == iteration:
            if iteration <= PRUNE_THRESH:
                mean = current_sum / iteration
                metric = max(x - mean for x in current)
                upper_bound = ceil(log2(iteration) * ORD_TABLE[cur_ord_key]) + 1
                if metric > (1 - error_rate) * upper_bound:
                    # sequence statistically incorrect
                    return

            try:
                yield AronsonSequence(letter, current, direction, check_semantics=True)
            except VerificationError:
                return
            return

        for e in list(remaining):
            if is_valid_extension(e, non_elements, current):
                current.append(e)
                remaining.remove(e)
                yield from rec(current_sum + e, remaining)
                remaining.add(e)
                current.pop()

    yield from rec(current_sum, remaining)

def worker_task(args):
    (elem, initial_remaining, iteration, cur_ord_key,
     non_elements, letter, direction, error_rate) = args
    return list(generate_valid_seqs(elem, initial_remaining, iteration, cur_ord_key,
                                    non_elements, letter, direction, error_rate))


# --------------------------------------------------
# Parallel generate_full implementation
# --------------------------------------------------

def generate_full_parallel(self, n_iterations: int, error_rate: float = 0.0):
    if n_iterations <= 0:
        return

    cur_ord_key = ORD_INITIAL
    while self.cur_iter < n_iterations:
        self.cur_iter += 1
        iteration = self.cur_iter
        upper_bound = iteration * ORD_TABLE[cur_ord_key] + 2 * 2 * self.get_prefix_idx()
        if upper_bound >= 10 ** (cur_ord_key + 1):
            cur_ord_key += 1

        initial_remaining = {x for x in range(1, upper_bound) if x not in self.non_elements}
        non_elements = set(self.non_elements)
        top_elems = sorted(initial_remaining)

        # Prepare args for workers
        common = (
            initial_remaining,
            iteration,
            cur_ord_key,
            non_elements,
            self.letter,
            self.direction,
            error_rate
        )
        tasks = [(elem, *common) for elem in top_elems]

        # Dispatch
        with multiprocessing.Pool(processes=n_procs) as pool:

            results = pool.map(worker_task, tasks)

        # Aggregate
        all_seqs = set()
        for seq_list in results:
            all_seqs.update(seq_list)

        self._update_iter(all_seqs)


# Patch the method on AronsonSet
AronsonSet.generate_full = generate_full_parallel


def main():
    parser = argparse.ArgumentParser(description='Distributed Aronson sequence generator.')
    parser.add_argument('n', type=int, nargs='?', default=4,
                        help='Number of iterations to generate (default: 4)')
    args = parser.parse_args()
    run_generation(args.n)
    print('Done.')


if __name__ == '__main__':
    main()
