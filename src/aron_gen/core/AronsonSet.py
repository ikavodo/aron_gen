from __future__ import annotations

from itertools import islice, combinations, permutations
from math import log2, ceil
from .AronsonSequence import AronsonSequence, Direction, Refer, REPR_PREFIX, REPR_SUFFIX, VerificationError
from collections import defaultdict, Counter
from typing import Callable, Literal
from contextlib import suppress

# global: dictionary with maximum ordinal lengths per number of bits in decimal representation
ORD_TABLE = {i + 1: j for i, j in enumerate([7, 14, 26, 39, 45, 56, 69, 75, 87, 99])}
# initial key for maximum ordinal length
ORD_INITIAL = 2
# for n_iters > 4 don't prune, as don't know enough at this point
PRUNE_THRESH = 4


# Exception classes


class GenError(Exception):
    """
    Custom exception raised when generating from the input sequence is impossible.

    :param message: The error message to be shown.
    :param n: Number of iterations or elements (depending on mode).
    :param mode: "iter" if referring to iterations, "elem" if to elements.
    """

    def __init__(self, n: int, message: str = "Generating failed", mode: Literal["iter", "elem"] = "iter"):
        self.n = n
        self.message = message
        self.mode = mode
        super().__init__(self.message)

    def __str__(self):
        if self.mode == "iter":
            return f"{self.message}: converged after {self.n} iterations"
        else:
            return f"{self.message}: stopped after {self.n} elements"


# This class holds and generates correct Aronson sequences
class AronsonSet:
    """
    Class for generating correct AronsonSequence objects of a given letter and direction,
    see https://ikavodo.github.io/aronson-1/
    :param letter: The letter used for generating sequences.
    :param direction: sequences generation direction.
    """

    def __init__(self, letter: str, direction: Direction = Direction.FORWARD):
        AronsonSequence.check_letter(letter)
        AronsonSequence.check_direction(direction)
        # Prune generation search space: these ordinals appear in no correct sequences in a given direction
        self.non_elements = {2, 3, 5, 6, 8, 9} if direction == Direction.FORWARD else {1, 2, 5, 6, 9, 10}
        self.letter = letter.lower()  # Letter used for generating sequences
        self.direction = direction  # Sequence direction
        self.iter_dict = defaultdict(set)  # init empty dictionary
        self.cur_iter = 0  # no generating iterations yet
        self.seen_seqs = {AronsonSequence(self.letter, [], self.direction)}  # every set contains the empty sequence
        self.iter_dict[self.cur_iter] = self.seen_seqs.copy()  # initial set
        self.subset_dict = defaultdict(set)  # used for fast generation

    @property
    def display_letter(self):
        """ Display in uppercase"""
        return self.letter.upper()

    # constructors
    @classmethod
    def from_sequence(cls, seq: AronsonSequence = None, error_check: bool = True):
        """
        constructor of singleton set (holding a single non-empty AronsonSequence instance).
        :param seq: AronsonSequence
        :param error_check: error checking
        :return: instance
        """
        seq = seq if seq is not None else AronsonSequence('t')
        if error_check and not seq.is_correct():
            # sequence must be correct to generate a set instance
            raise VerificationError(input_seq=seq)

        obj = cls(seq.get_letter(), seq.get_direction())
        obj._update_iter({seq})  # update seen_seqs and iter_dict
        return obj

    @classmethod
    def from_dict(cls, iter_dict: dict[int, set[AronsonSequence]] = None, error_check: bool = True):
        """
        Constructor from set of AronsonSequence instances.
        :param iter_dict: Dictionary of iterations
        :param error_check: over sequences
        :return: Class instance
        """
        if not iter_dict or iter_dict is None:
            return AronsonSet('t')  # default

        seqs = next((v for v in iter_dict.values() if v), set())

        if not seqs:
            # Safety measure
            raise ValueError("Dictionary must hold non-empty value")

        # error checking
        field_set = {(s.get_letter(), s.get_direction()) for sets in iter_dict.values() for s in sets}

        if len(field_set) > 1:
            raise ValueError("All sequences must have letter and direction of current set")

        if error_check:
            for seq_set in iter_dict.values():
                for s in seq_set:
                    if not s.is_correct():
                        raise VerificationError(input_seq=s)

        # look into dictionary for letter and direction fields
        seq = next(iter(seqs))
        new_instance = AronsonSet(seq.letter, seq.direction)
        emp_seq = AronsonSequence(seq.letter, [], seq.direction)

        # Every set must hold empty sequence
        if not any(emp_seq in v for v in iter_dict.values()):
            # no side effects
            iter_dict = iter_dict.copy()
            iter_dict[0].add(emp_seq)

        new_instance._set_iter_dict(iter_dict)
        return new_instance

    @classmethod
    def from_set(
            cls,
            seqs: set[AronsonSequence] | None = None,
            error_check: bool = True,
    ) -> "AronsonSet":
        """
        Construct an AronsonSet from an existing set of AronsonSequence instances.
        All sequences are assigned to iteration 0. No generation is performed.
        """
        if seqs is None:
            seqs = {AronsonSequence('t')}

        # Build iteration dictionary directly
        iter_dict = defaultdict(set)
        iter_dict[0] = set(seqs)

        # Delegate to fast internal constructor
        return cls.from_dict(iter_dict, error_check=error_check)

    def is_correct(self, seq: AronsonSequence):
        """
        Is given sequence correct with regard to class instance
        :param seq: AronsonSequence to be verified
        :return: True/False
        """

        return (
                seq.get_letter() == self.display_letter and
                seq.get_direction() == self.direction and seq.is_correct()
        )

    def is_complete(self, seq: AronsonSequence):
        """
        If given sequence is complete with regard to class instance
        :param seq: AronsonSequence to be checked
        :return: True/False
        """
        return (
                seq.get_letter() == self.display_letter and
                seq.get_direction() == self.direction and seq.is_complete()
        )

    def _update_iter(self, seqs: set[AronsonSequence]):
        """
        Update iteration dictionary and seen sequences
        :param seqs: for updating
        :return: None
        """
        self.iter_dict[self.cur_iter].update(seqs)
        self.seen_seqs.update(seqs)

    @staticmethod
    def backward_search(seq: AronsonSequence):
        """
        Find all occurrences missing in sentence prefix
        :param seq: for which to find occurrences
        :return: set of new sequences with appended missing occurrences
        """
        occurrences = seq.get_prefix_missing()
        # using __add__() operator of AronsonSequence class over individual elements
        return {seq.copy() + occ for occ in occurrences}

    def get_prefix_idx(self):
        return len(REPR_PREFIX.replace(" ", "")) + 1 if self.direction == Direction.FORWARD else len(
            REPR_SUFFIX.replace(" ", ""))

    def _agen(self, seq: AronsonSequence):
        """
        Internal generator for backwards-based generation.
        :param seq: An optional AronsonSequence to generate from.
        :return: A generator yielding new indices for the sequence.
        """
        idx, s = seq.get_prefix(), seq.get_sentence()
        pref_idx = self.get_prefix_idx()
        s = s[idx:-pref_idx] if self.direction == Direction.FORWARD else s[pref_idx: -idx if idx else None][::-1]
        while True:
            idx_rel = 1 + s.find(self.letter)  # Find the relative position of the letter
            if idx_rel <= 0:  # Letter not found in string buffer, raise StopIteration
                break

            idx += idx_rel
            yield idx
            ordinal = seq.n2w(idx)
            s = s[idx_rel:] + (ordinal if self.direction == Direction.FORWARD else ordinal[::-1])

    def backward_generate(self, n: int, seq: AronsonSequence = None):
        """
        Generate new sequences from well-behaved sequences (with backward/self-referring elements only)
        :param n: num of elements to generate
        :param seq: for generation, optional
        :return: generated sequence with n new elements
        """
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        if n < 0:
            raise ValueError(" n must be non negative")

        new_elements = list(islice(self._agen(seq=seq), n))
        # __add__() operator
        new_seq = seq + new_elements
        if len(new_seq) < len(seq) + n:
            # could not extend sequence up to desired length, print appropriate error message
            raise GenError(len(seq) + n, 'elem')

        return {new_seq}

    def generate_aronson(self, n: int) -> set[AronsonSequence]:
        """
        Wrapper for backward_generate() method with default arguments
        :param n: of elements to generate
        :return: singleton set holding an AronsonSequence instance
        """
        return self.backward_generate(n)

    def swap(self, seq: AronsonSequence = None):
        """
        Swap two elements if the positions to which all elements point are unaffected by the swap
        :param seq: AronsonSequence object
        :return: set of newly generated sets in which swapping is legal
        """
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        new_sets = set()
        for i, j in combinations(range(len(seq)), 2):
            ranges = [seq.get_reference(seq[x])[0] for x in (i, j)]
            range_of_effect = range(min(r.start for r in ranges), max(r.stop for r in ranges))
            if all(x - 1 not in range_of_effect for x in seq.get_elements()):
                # all elements unaffected
                swapped = seq.get_elements().copy()
                swapped[i], swapped[j] = swapped[j], swapped[i]
                new_sets.add(AronsonSequence(self.letter, swapped, self.direction))
        return new_sets

    @staticmethod
    def _subset_to_index_pairs(seq_length):
        """
        Map each index pair (i, j) to all non-empty subsets of indices strictly in between.
        If i and j are adjacent, also include each individually as possible subsets.
        :param seq_length: for which to find index pairs
        :return: relevant pairs
        """
        index_pair_subsets = defaultdict(list)
        for i in range(seq_length):
            for j in range(i + 1, seq_length):
                mid = list(range(i + 1, j))  # indices strictly in between
                if not mid:
                    # i and j are adjacent → allow deleting i or j individually
                    [index_pair_subsets[(i, j)].append((x,)) for x in (i, j)]
                else:
                    for r in range(1, len(mid) + 1):
                        for sub in combinations(mid, r):
                            index_pair_subsets[(i, j)].append(sub)
        return index_pair_subsets

    def subset(self, seq: AronsonSequence = None):
        """
        Take subset if other elements aren't affected.
        :param self:
        :param seq: AronsonSequence object
        :return: set of newly generated sets in which swapping is legal
        """

        new_sets = set()
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        if len(seq) > self.cur_iter:
            self.subset_dict = self._subset_to_index_pairs(len(seq))
        # pick arbitrary pairs and delete every subset in between
        for i, j in combinations(range(len(seq)), 2):
            for sub in self.subset_dict[(i, j)]:
                ranges = [seq.get_reference(seq[x])[0] for x in sub]
                # anything after first discarded element is affected
                range_min = min(r.start for r in ranges)
                if all(x - 1 < range_min for x in seq.get_elements()):
                    # All elements refer backwards before index of omitted element
                    sub_seq = seq.get_elements().copy()
                    [sub_seq.remove(seq[s]) for s in sub]
                    new_sets.add(AronsonSequence(self.letter, sub_seq, self.direction))
        return new_sets

    def forward_generate(self, seq: AronsonSequence = None):
        """
        Generate by searching within a bounded search space for valid ordinals
        :param seq: to append to
        :return: generated sequence
        """
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        sentence_len = len(seq.get_sentence())
        # generate all singletons if no input
        lower_bound = 1 if seq.is_empty() else sentence_len - (
            self.get_prefix_idx()) - 1
        new_seqs = set()
        # digits in sentence length
        ord_key = len(str(sentence_len))
        if seq.is_empty():
            upper_bound = ORD_TABLE[ord_key] + sentence_len
        else:
            upper_bound = len(seq) * ORD_TABLE[ord_key] + sentence_len

        for elem in range(lower_bound, upper_bound + 1):
            # exhaustive search
            if elem in self.non_elements:
                # skip
                continue
            # __add__() operator
            candidate = seq + elem
            if self.is_correct(candidate):
                new_seqs.add(candidate)
        return new_seqs

    def generate_singletons(self):
        """
        Wrapper for generating set of singleton AronsonSequences
        :return: set of correct singleton AronsonSequences
        """
        return self.forward_generate(AronsonSequence(self.letter, [], self.direction))

    def is_valid_extension(self, elem, current_perm):
        """
        helper for checking validity of new sequence elements
        :param elem: to add
        :param current_perm: which elem is added to
        :return: True or False
        """
        if elem in self.non_elements or (elem - 1 in current_perm and elem - 2 in current_perm) or \
                (elem + 1 in current_perm and elem + 2 in current_perm):
            return False
        return True

    # Currently infeasible from n > 4
    def generate_full(self, n_iterations: int, error_rate: float = 0., check_semantics=False):
        """
        Exhaustive generation of all Aronson sequences up to given length
        :param error_rate: degree of search precision, with 0. corresponding to no error and 1 to no sequences found.
        Is pessimistic- (sequences_found/total_sequences) >= (1 - error_rate) for n_iterations <= 4
        :param n_iterations: max length of generated sequences
        :param check_semantics: set to True for large n_iterations
        :return: None
        """

        def backtrack(current_perm, current_sum, remaining, max_len):
            if len(current_perm) == max_len:
                if max_len > PRUNE_THRESH:
                    # don't use metric for n>4 as we can't guarantee full generation
                    yield current_perm.copy()
                    return

                mean = current_sum / max_len
                metric = max(x - mean for x in current_perm)
                upper_metric_bound = ceil(log2(max_len) * ORD_TABLE[cur_ord_key]) + 1

                if metric <= (1 - error_rate) * upper_metric_bound:
                    yield current_perm.copy()
                return

            for elem in set(remaining):
                if self.is_valid_extension(elem, current_perm):
                    yield from backtrack(
                        current_perm + [elem],
                        current_sum + elem,
                        remaining - {elem},
                        max_len
                    )

        if n_iterations <= 0:
            return

        # generation engine
        cur_ord_key = ORD_INITIAL

        while self.cur_iter < n_iterations:
            self.cur_iter += 1
            upper_bound = self.cur_iter * ORD_TABLE[cur_ord_key] + 2 * self.get_prefix_idx()
            if upper_bound >= 10 ** (cur_ord_key + 1):
                cur_ord_key += 1

            initial_remaining = {x for x in range(1, upper_bound) if x not in self.non_elements}

            cur_seqs = set()
            for perm in backtrack([], 0, initial_remaining, self.cur_iter):
                if check_semantics:
                    # Faster for n_iterations>=4
                    try:
                        seq = AronsonSequence(self.letter, perm, self.direction, check_semantics=check_semantics)
                        cur_seqs.add(seq)
                    except VerificationError:
                        continue
                else:
                    seq = AronsonSequence(self.letter, perm, self.direction)
                    if seq.is_correct():
                        cur_seqs.add(seq)

            self._update_iter(cur_seqs)

    def generate_fast(self, n_iterations: int, forward_generate=False):
        """
        Optimized generation using swap/subset operations
        :param forward_generate: more computationally expensive, generates more sequences
        :param n_iterations: to generate for
        :return: None
        """
        if n_iterations < 0:
            raise ValueError("Num of iterations must be non-negative")
        if n_iterations == 0:
            return

        while self.cur_iter < n_iterations:

            prev_seqs = self.iter_dict[self.cur_iter]
            self.cur_iter += 1

            if self.cur_iter == 1:
                cur_seqs = self.generate_singletons()
            else:
                # update subset_dict at iteration start
                self.subset_dict = self._subset_to_index_pairs(self.cur_iter)
                cur_seqs = set()
                for seq in prev_seqs:
                    cur_seqs.update(self.swap(seq))
                    cur_seqs.update(self.subset(seq))
                    insert_idx = seq.get_reference(seq[-1])[0].stop
                    if not any(elem > insert_idx for elem in seq):
                        # no element points to after last one, meaning can generate from rules
                        cur_seqs.update(self._handle_backward_rules(seq))
                        if forward_generate:
                            # Becomes computationally expensive as upper_bound grows
                            cur_seqs.update(self.forward_generate(seq))

            filtered = {seq for seq in cur_seqs if seq not in self.seen_seqs}
            if not filtered:
                # converged
                raise GenError(self.cur_iter)
            self._update_iter(filtered)

    def _handle_backward_rules(self, seq):
        """
        Handle backward generation rules for a given sequence.
        """
        results = self.backward_search(seq) if not seq.is_prefix_complete() else set()
        with suppress(GenError):
            results.update(self.backward_generate(1, seq))
        return results

    # Filters
    def filter_elements(self, elems):
        """
        return all seen sequences containing elements
        :param elems: for sequences
        :return: set of seen sequences including elems

        """
        if isinstance(elems, int):
            elems = {elems}
        elif isinstance(elems, list):
            elems = set(elems)
        elif not isinstance(elems, set):
            raise ValueError("Input must be an integer, list, or set")
        filtered = {
            n_iter: {seq for seq in seqs if elems.issubset(seq)}
            for n_iter, seqs in self.iter_dict.items()
        }
        return AronsonSet.from_dict(filtered)

    def filter_symmetric(self, seq_len=0):
        """
        return all sequences for which all permutations are also in set
        :param seq_len: to start filtering from
        :return: set of such sequences
        """
        filtered: dict[int: set[AronsonSequence]] = defaultdict(set[AronsonSequence])
        for n_iter, seqs in self.iter_dict.items():
            for seq in seqs:
                seq_perm = {AronsonSequence(self.letter, list(perm), self.direction) for perm in
                            permutations(seq, len(seq))}
                if all(perm in self.seen_seqs for perm in seq_perm) and len(seq) >= seq_len:
                    # Perhaps not all permutations generated within same iteration
                    filtered[n_iter].add(seq)
        return AronsonSet.from_dict(filtered)

    def filter_refs(self, refs):
        """
        return all seen sequences including elements with reference pointers in the set refs
        :param refs: pointers for elements in sequence
        :return: set of seen sequences including refs

        """
        if isinstance(refs, list):
            refs = set(refs)
        elif isinstance(refs, Refer):
            refs = {refs}
        elif not isinstance(refs, set):
            raise ValueError("input argument must be iterable or a Direction type")

        filtered = defaultdict(set)

        for n_iter, seqs in self.iter_dict.items():
            for seq in seqs:
                current_refs = {ref[1] for ref in seq.get_refer_dict().values()}
                if refs.issubset(current_refs):
                    filtered[n_iter].add(seq)

        return AronsonSet.from_dict(filtered)

    def filter_monotonic(self, ascending=True):
        """
        Filter out of instance only those sets which are monotonic with respect to ascending/descending order
        :param ascending: True by default
        :return: monotonic sets
        """
        filtered = set()
        condition = 1 if ascending else -1
        for seq in self.seen_seqs:
            tuple_mono = seq.is_monotonic()
            # Take trivially monotonic sequences in either case
            if tuple_mono[0] and (tuple_mono[1] is None or tuple_mono[1] == condition):
                filtered.add(seq)
        return AronsonSet.from_set(filtered)

    def find_non_elements(self, n_iter=None):
        """
        Find all elements up to maximum element in set which do not appear in any sequence
        :param n_iter: to look within
        :return: elements
        """
        if n_iter is not None and (n_iter < 0 or n_iter > self.cur_iter):
            raise ValueError("must choose correct iteration")

        if len(self.seen_seqs) == 1:
            # trivially empty
            return set()

        search_set = self.iter_dict[n_iter] if n_iter is not None else self.seen_seqs
        seen_elems = {elem for seq in search_set for elem in seq}
        # set has at least one non-empty sequence-> max is defined
        return set(range(1, self.max)) - seen_elems

    def get_elements(self):
        """ get all elements appearing within a given sequence within the set"""
        return {elem for seq in self.seen_seqs for elem in seq}

    def get_unique_element_sequences(self, other: 'AronsonSet'):
        """ Get all sequences holding elements not appearing in any sequence in another set"""
        missing_elements = self.get_elements() - other.get_elements()
        return {seq for seq in self.seen_seqs if any(elem in seq for elem in missing_elements)}

    def is_empty(self):
        return len(self.seen_seqs) == 1 and self.seen_seqs == {AronsonSequence(self.get_letter(), [],
                                                                               self.direction)}

    # Utility methods
    def copy(self):
        """ shallow copy for new instance"""
        new_set = AronsonSet(self.letter, self.direction)
        new_set._set_iter_dict(self.iter_dict)
        return new_set

    def clear(self):
        """ wrapper for clearing a set"""
        self._set_iter_dict({})

    # Setters
    def _set_iter_dict(self, new_dict):
        """ main setter, of iteration dictionary and relevant set of seen sequences"""
        emp_seq = {AronsonSequence(self.letter, [], self.direction)}
        new_dict = new_dict or {0: emp_seq}
        self.iter_dict = new_dict
        self.seen_seqs = set(seq for seqs in new_dict.values() for seq in seqs)
        self.cur_iter = max(new_dict.keys())

    def flip_direction(self):
        """ Flip direction of an empty set"""
        if len(self.seen_seqs) > 1:
            raise ValueError("Can't flip direction of non-default AronsonSet instance.")
        self.direction = self.direction.flip
        for seq in self.seen_seqs:
            # flip empty seq direction
            seq.flip_direction()

    def _set_operation_core(self: 'AronsonSet', other: 'AronsonSet', set_op: Callable[[set, set], set], n: int = 0,
                            error_rate=0.0):
        """
        Core logic for set operations on AronsonSets.
        """

        def first_seen(seq, aset):
            """ Helper for looking up first appearance iteration"""
            return next((i for i, s in aset.iter_dict.items() if seq in s), float('inf'))

        if self.letter != other.letter:
            raise ValueError("Mismatched letters: sets must use the same letter.")

        # Create independent working copies and generate up to n iterations
        set1, set2 = self.copy(), other.copy()
        set1.generate_full(n, error_rate)
        set2.generate_full(n, error_rate)

        # Apply set operation on tuple representations (AronsonSequence equality depends on direction)
        op_result = set_op(
            {tuple(seq.get_elements()) for seq in set1},
            {tuple(seq.get_elements()) for seq in set2}
        )

        new_iter_dict = defaultdict(set)
        new_iter_dict[0].add(AronsonSequence(self.letter, [], self.direction))

        for elements in op_result:
            seq = AronsonSequence(self.letter, list(elements), self.direction)
            # write sequence in set corresponding to minimum generation iteration
            gen_iter = min(first_seen(seq, set1), first_seen(seq, set2))
            new_iter_dict[gen_iter].add(seq)

        return AronsonSet.from_dict(new_iter_dict)

    # getters
    def get_seen_seqs(self):
        return self.seen_seqs

    def get_iter_dict(self):
        return self.iter_dict

    def get_letter(self):
        """
        :return: letter in upper case
        """
        return self.display_letter

    def get_n_iterations(self):
        return self.cur_iter

    def get_direction(self):
        return self.direction

    def peek(self):
        """ Recover an arbitrary sequence from seen sequences"""
        return next(iter(self.seen_seqs))

    def discard(self, seq: AronsonSequence):
        """
        Discard a sequence from a set.
        :param seq: to be discarded
        :return: None
        """
        # no error-checking: discard does nothing if seq is misaligned w.r.t. direction or letter
        for seqs in self.iter_dict.values():
            seqs.discard(seq)
        self.get_seen_seqs().discard(seq)

    @property
    def max(self):
        """ maximum element seen in some sequence in the set"""
        if len(self.seen_seqs) == 1:
            raise ValueError("Set contains only the empty sequence")
        return max(seq.get_prefix() for seq in self.seen_seqs)

    def get_len_dict(self, generated_full=False):
        """
        get dictionary mapping lengths of sequences to number of sequences in set.
        :return:
        """
        if generated_full:
            # iteration dictionary is partitioned by sequence length
            return {key: len(val) for key, val in self.iter_dict.items()}
        # compute a new dictionary
        return Counter(len(seq) for seq in self.seen_seqs)

    # operator overloading
    def __and__(self, other: 'AronsonSet', n: int = 0, error_rate: float = 0.):
        """
        & operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :return: set instance
        """
        return self._set_operation_core(other, set.intersection, n, error_rate)

    def __iand__(self, other, n: int = 0, error_rate: float = 0.):
        """
        &= operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :return: set instance
        """
        result = self._set_operation_core(other, set.intersection, n, error_rate)
        self.cur_iter = n
        self._set_iter_dict(result.get_iter_dict())
        return self

    def __or__(self, other: 'AronsonSet', n: int = 0, error_rate: float = 0.):
        """
        | operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :return: set instance
        """
        if self.direction != other.direction:
            raise ValueError("sets must have same direction")
        return self._set_operation_core(other, set.union, n, error_rate)

    def __ior__(self, other: 'AronsonSet', n: int = 0, error_rate: float = 0.):
        """
        |= operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :return: set instance
        """
        if self.direction != other.direction:
            raise ValueError("sets must have same direction")
        result = self._set_operation_core(other, set.union, n, error_rate)
        self.cur_iter = n
        self._set_iter_dict(result.get_iter_dict())
        return self

    def __sub__(self, other: 'AronsonSet', n: int = 0, error_rate: float = 0.):
        """
        - operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :return: set instance
        """
        if self.direction == other.direction:
            return self._set_operation_core(other, set.difference, n, error_rate)

        # Otherwise need to compute intersection first if directions are not aligned
        return self._set_operation_core(self & other, set.difference, n, error_rate)

    def __isub__(self, other: 'AronsonSet', n: int = 0, error_rate: float = 0.):
        """
        -= operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :return: set instance
        """
        # take difference with intersection (which has same direction) if directions not the same
        if self.direction == other.direction:
            result = self._set_operation_core(other, set.difference, n, error_rate)
        else:
            result = self._set_operation_core(self & other, set.difference, n, error_rate)

        self.cur_iter = n
        self._set_iter_dict(result.get_iter_dict())
        return self

    def __getitem__(self, index: int):
        """
        [] operator, returns the generated sequences at specified iteration
        :param index: of iterations for which AronsonSequences are to be retrieved.
        :return: The relevant generated sets
        """
        return self.iter_dict[index]

    def __iter__(self):
        """
        for loop operator, returns an iterator over the seen sequences.
        :return: An iterator for the sequences.
        """
        return iter(self.seen_seqs)

    def __len__(self):
        """
        len() operator
        :return: The length of the Aronson sequence.
        """
        return len(self.seen_seqs)

    def __contains__(self, item):
        """ in operator, returns True or False"""
        return item in self.seen_seqs

    @property
    def _hashable_iter_dict(self):
        """ Returns a frozenset representation of iter_dict for hashing"""
        return frozenset(
            (i, frozenset(seq_set)) for i, seq_set in self.iter_dict.items()
        )

    def __eq__(self, other):
        """ = operator"""
        if not isinstance(other, AronsonSet):
            return NotImplemented
        return (
                self.letter == other.letter and
                self.direction == other.direction and
                # create immutable data-structure from iteration dictionaries
                self._hashable_iter_dict == other._hashable_iter_dict
        )

    def __hash__(self):
        """ hash() operator"""
        return hash((
            self.letter,
            self.direction,
            self._hashable_iter_dict
        ))

    def __repr__(self):
        """ repr() operator, prints repr() method of sequences by iteration (starting with empty sequence)"""
        return "\n".join(repr(seq) for i in sorted(self.iter_dict) for seq in self.iter_dict[i])
