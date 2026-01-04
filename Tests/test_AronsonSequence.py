import unittest
from itertools import permutations

from num2words import num2words
from AronsonSequence import AronsonSequence, REPR_PREFIX, REPR_SUFFIX, Refer, Direction, REPR_FORWARD, \
    REPR_BACKWARD

# take into account letter at beginning of sentence
LEN_REPR_PREFIX = len(REPR_PREFIX.replace(" ", "")) + 1
LEN_REPR_SUFFIX = len(REPR_SUFFIX.replace(" ", ""))

class AronsonSequenceTests(unittest.TestCase):
    """ unit test for AronsonSet"""

    def test_construction(self):
        # test default values
        default = AronsonSequence('t')
        self.assertEqual(default.is_empty(), True)
        self.assertEqual(default.get_direction(), Direction.FORWARD)
        # Normal construction
        seq = AronsonSequence('t', [1, 2, 3], Direction.FORWARD)
        # check is converted to lower_case
        self.assertEqual(seq.letter, 't')
        self.assertEqual(seq.elements, [1, 2, 3])
        self.assertTrue(seq.direction)

        # Invalid letters
        for invalid_letter in ['tt', '!', '']:
            with self.subTest(letter=invalid_letter):
                with self.assertRaises(ValueError):
                    AronsonSequence(invalid_letter, [1], Direction.FORWARD)

        # Invalid elements
        invalid_elements_list = [
            [1, 'x'],  # contains non-int
            [0, 1],  # contains non-positive
        ]
        for invalid_elements in invalid_elements_list:
            with self.subTest(elements=invalid_elements):
                with self.assertRaises(ValueError):
                    AronsonSequence('t', invalid_elements, Direction.FORWARD)

    def test_construction_duplicate_elements(self):
        duplicate_elements_list = [
            ([1, 2, 1], [1, 2]),
            ([1, 1, 2], [1, 2]),
            ([2, 1, 2], [2, 1]),
            ([2, 1, 1], [2, 1])
        ]

        for elements, expected in duplicate_elements_list:
            with self.subTest(elements=elements):
                seq = AronsonSequence('t', elements, Direction.FORWARD)
                self.assertEqual(seq.get_elements(), expected)

    def test_construction_empty(self):
        seq1 = AronsonSequence('t')
        seq2 = AronsonSequence('t', [], Direction.BACKWARD)
        self.assertEqual(seq1.elements, seq2.elements)

    def test_repr(self):
        # test that human representation is correct
        seq = AronsonSequence('t', [25], Direction.FORWARD)
        self.assertTrue(repr(seq)[0].isupper())
        self.assertIn(REPR_PREFIX + num2words(25, ordinal=True) + REPR_SUFFIX, repr(seq))
        self.assertIn(REPR_FORWARD, repr(seq))
        self.assertEqual(str(seq), repr(seq))
        seq = AronsonSequence('t', [100, 101], Direction.BACKWARD)
        for elem in seq.get_elements():
            self.assertIn(num2words(elem, ordinal=True), str(seq))
        self.assertIn(REPR_BACKWARD, repr(seq))

    def test_repr_empty(self):
        seq1 = AronsonSequence('t')
        seq2 = AronsonSequence('t', [], Direction.BACKWARD)
        # same representation
        self.assertEqual(repr(seq1), repr(seq2))

    def test_sentence(self):
        # test that the internal sentence is correct, use something with hyphen
        seq = AronsonSequence('t')
        suffix = REPR_PREFIX.replace(" ", "") + seq.n2w(25) + REPR_SUFFIX.replace(" ", "")
        elems = [3, 4]
        test_sequences = [
            (AronsonSequence('t', [25], Direction.FORWARD), suffix),
            (AronsonSequence('t', elems, Direction.BACKWARD), ''.join([seq.n2w(i) for i in elems[::-1]]))
        ]

        for seq, part in test_sequences:
            with self.subTest(seq=seq):
                self.assertIn(part, seq.get_sentence())

    def test_has_forward_refer(self):
        # check that has_forward_ref() method works correctly
        test_cases = [
            # backwards referring in both directions
            (AronsonSequence('t', [1, 4, 11], Direction.FORWARD), False),
            (AronsonSequence('t', [3, 4], Direction.BACKWARD), False),

            # self referring in both directions
            (AronsonSequence('t', [10, 12], Direction.FORWARD), False),
            (AronsonSequence('t', [8, 14], Direction.BACKWARD), False),

            # forward referring in both directions
            (AronsonSequence('t', [19, 100], Direction.FORWARD), True),
            (AronsonSequence('t', [19, 100], Direction.BACKWARD), True),
            # trivially correct
            (AronsonSequence('t'), False),
            (AronsonSequence('t', [], Direction.BACKWARD), False),
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(seq.has_forward_ref(), expected)

    def test_all_refers(self):
        # check that refer_dictionaries are correct
        test_cases = [
            # all referring types- BACKWARD, SELF, BACKWARD, SELF, FORWARD
            (AronsonSequence('t', [1, 12, 17, 30, 100], Direction.FORWARD), True),
            (AronsonSequence('t', [3, 13, 21, 34, 100], Direction.BACKWARD), True)
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                refs = set(ref for _, ref in seq.get_refer_dict().values())
                self.assertEqual(refs, set(Refer))

    def test_ref_dict_range(self):
        # check that refer_dictionaries are correct
        test_cases = [
            # all referring types- BACKWARD, SELF, BACKWARD, SELF, FORWARD
            (AronsonSequence('t', [1], Direction.FORWARD), True),
            (AronsonSequence('t', [1], Direction.BACKWARD), True)
        ]
        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                rel_pos = LEN_REPR_PREFIX if seq.get_direction() == Direction.FORWARD \
                    else LEN_REPR_SUFFIX
                for idx_range, _ in seq.get_refer_dict().values():
                    self.assertIn(rel_pos, idx_range)

    def test_refer_empty(self):
        # check that refer_dictionaries are empty
        test_cases = [
            # all referring types- BACKWARD, SELF, BACKWARD, SELF, FORWARD
            (AronsonSequence('t'), {}),
            (AronsonSequence('t', [], Direction.BACKWARD), {})
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                ref = seq.get_refer_dict()
                self.assertEqual(ref, expected)

    def test_is_empty(self):
        test_cases = [
            # all referring types- BACKWARD, SELF, BACKWARD, SELF, FORWARD
            (AronsonSequence('t'), True),
            (AronsonSequence('t', [], Direction.BACKWARD), True),
            (AronsonSequence('t', [1], Direction.FORWARD), False),
            (AronsonSequence('t', [1], Direction.BACKWARD), False)
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(seq.is_empty(), expected)

    def test_is_correct(self):
        test_cases = [
            # all of these are correct sentences
            # backward referring
            (AronsonSequence('t', [1, 4, 11], Direction.FORWARD), True),
            (AronsonSequence('t', [3, 4, 11], Direction.BACKWARD), True),
            # self referring
            (AronsonSequence('t', [10, 12], Direction.FORWARD), True),
            (AronsonSequence('t', [8, 14], Direction.BACKWARD), True),
            # forward referring
            (AronsonSequence('t', [19], Direction.FORWARD), True),
            (AronsonSequence('t', [19], Direction.BACKWARD), True),
            # all referring
            (AronsonSequence('t', [1, 12, 17, 30], Direction.FORWARD), True),
            (AronsonSequence('t', [3, 13, 21, 34], Direction.BACKWARD), True),
            # false
            (AronsonSequence('t', [19, 100], Direction.FORWARD), False),
            (AronsonSequence('t', [19, 100], Direction.BACKWARD), False),
            # trivial
            (AronsonSequence('t'), True),
            (AronsonSequence('t', [], Direction.BACKWARD), True),
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(seq.is_correct(), expected)

    def test_clear(self):
        empty = AronsonSequence('t')
        emp_cpy = empty.copy()
        emp_cpy.clear()
        self.assertEqual(emp_cpy, empty)
        seq = AronsonSequence('t', [25], Direction.FORWARD)
        seq.clear()
        self.assertEqual(seq, empty)

    def test_add(self):

        empty = AronsonSequence('t')
        self.assertEqual(empty + empty, empty)

        # mismatched letter
        with self.assertRaises(ValueError):
            _ = empty + AronsonSequence('x', [2])

        # mismatched direction
        with self.assertRaises(ValueError):
            _ = empty + AronsonSequence('t', [2], Direction.BACKWARD)

        # invalid element (e.g., 0)
        with self.assertRaises(ValueError):
            _ = empty + [0, 2]

        # good construction
        seq1 = AronsonSequence('t', [1])
        seq2 = AronsonSequence('t', [4])
        expected = AronsonSequence('t', [1, 4])
        self.assertEqual(seq1 + seq2, expected)

        # Test adding int
        self.assertEqual(seq1 + 5, AronsonSequence('t', [1, 5]))

        # Test adding list of ints
        self.assertEqual(seq1 + [6, 7], AronsonSequence('t', [1, 6, 7]))

    def test_iadd(self):
        emp_seq = AronsonSequence('t')
        seq = AronsonSequence('t', [2])
        emp_seq += seq
        self.assertEqual(emp_seq, seq)

        # mismatched letter
        with self.assertRaises(ValueError):
            seq += AronsonSequence('x', [2])

        # mismatched direction
        with self.assertRaises(ValueError):
            seq += AronsonSequence('t', [2], Direction.BACKWARD)

        # invalid element in list
        with self.assertRaises(ValueError):
            seq += [3, 0]

        seq += 3
        seq2 = AronsonSequence('t', [2, 3])
        self.assertEqual(seq, seq2)
        elems = [4, 5]
        seq += elems
        self.assertEqual(seq, seq2 + AronsonSequence('t', elems))

    def test_complete(self):
        test_cases = [
            (AronsonSequence('l', [1, 23], Direction.FORWARD), True),
            (AronsonSequence('l', [23, 1], Direction.FORWARD), True),
            (AronsonSequence('l', [1], Direction.FORWARD), False),
            (AronsonSequence('j', [24], Direction.BACKWARD), True),
            (AronsonSequence('l', [6], Direction.BACKWARD), False),
            # trivial
            (AronsonSequence('t'), False),
            (AronsonSequence('t', [], Direction.BACKWARD), False),
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(seq.is_complete(), expected)

    def test_prefix(self):
        test_cases = [
            (AronsonSequence('t'), 0),
            (AronsonSequence('t', [], Direction.BACKWARD), 0),
            (AronsonSequence('t', [1, 11], Direction.FORWARD), 11),
            (AronsonSequence('t', [3, 11], Direction.BACKWARD), 11),
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(seq.get_prefix(), expected)

    def test_prefix_complete(self):
        test_cases = [
            (AronsonSequence('t'), True),
            (AronsonSequence('t', [], Direction.BACKWARD), True),
            (AronsonSequence('t', [1, 4], Direction.FORWARD), True),
            (AronsonSequence('t', [1, 11], Direction.FORWARD), False),
            (AronsonSequence('t', [3, 4], Direction.BACKWARD), True),
            (AronsonSequence('t', [3, 11], Direction.BACKWARD), False),
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(seq.is_prefix_complete(), expected)

    def test_get_prefix_missing(self):
        test_cases = [
            (AronsonSequence('t', [1, 11], Direction.FORWARD), {4}),
            (AronsonSequence('t', [3, 11], Direction.BACKWARD), {4}),
            (AronsonSequence('t'), set()),
            (AronsonSequence('t', [], Direction.BACKWARD), set()),
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(seq.get_prefix_missing(), expected)

    def test_set_elements_append(self):
        for direction in Direction:
            suffix_ind = len(REPR_SUFFIX) + len(REPR_FORWARD) if direction == Direction.FORWARD else len(REPR_BACKWARD)

            for append_flag in [True, False]:
                seq = AronsonSequence('t', [1, 2], direction)
                seq_repr = str(seq)
                prev_elements = seq.get_elements().copy()
                # Set elements with the given append flag
                seq.set_elements([3, 4], append=append_flag)

                if append_flag:
                    for elem in prev_elements:
                        self.assertIn(elem, seq.get_elements())
                    # When append=True, ensure the original elements are retained
                    self.assertIn(seq_repr[:-suffix_ind], str(seq))
                    self.assertEqual(seq.get_prefix(), max([3, 4]))
                else:
                    for elem in prev_elements:
                        self.assertNotIn(elem, seq.get_elements())
                    # When append=False, the previous elements should be removed
                    sliced = seq_repr[:-suffix_ind] if direction == Direction.FORWARD else seq_repr[suffix_ind:]
                    self.assertNotIn(sliced, str(seq))

    def test_set_elements_duplicates(self):
        # Test appending only duplicate elements
        seq = AronsonSequence('t', [1, 2], Direction.FORWARD)
        prev_elements = seq.get_elements().copy()
        seq.set_elements([1, 2], append=True)
        # Ensure no new elements are added
        self.assertEqual(seq.get_elements(), prev_elements)
        duplicates = [5, 1, 2, 6]
        # Test appending new elements with some duplicates
        seq.set_elements(duplicates, append=True)
        # Ensure original elements (1, 2) remain and new elements (5, 6) are added in correct order
        self.assertEqual(sorted(duplicates), seq.get_elements())

    def test_set_elements_empty_append(self):
        seq = AronsonSequence('t', [1, 2], Direction.FORWARD)
        seq.set_elements([], append=True)
        # does nothing
        self.assertEqual(seq.get_elements(), [1, 2])
        seq.set_elements(None, append=True)
        self.assertEqual(seq.get_elements(), [1, 2])

        seq.set_elements([], append=False)
        self.assertEqual(seq.get_elements(), [])
        # now check starting with empty list

    def test_set_elements_empty_start(self):
        seq1 = AronsonSequence('t')
        seq1.set_elements([], append=True)
        # does nothing
        self.assertEqual(seq1.get_elements(), [])
        # appending to empty list is equivalent to setting without append
        seq1.set_elements([1, 2], append=True)
        seq2 = AronsonSequence('t', [1, 2, 4], Direction.FORWARD)
        seq2.set_elements(seq2.get_elements()[:-1], append=False)
        self.assertEqual(seq1.get_elements(), seq2.get_elements())

    def test_get_range(self):
        seq = AronsonSequence('t', [1], Direction.FORWARD)
        r, _ = seq.get_reference(1)
        self.assertEqual(r, range(LEN_REPR_PREFIX, len(seq.get_sentence()) - LEN_REPR_SUFFIX))
        seq.flip_direction()
        self.assertEqual(r, range(LEN_REPR_SUFFIX, len(seq.get_sentence()) - LEN_REPR_PREFIX))

    def test_get_reference(self):
        for direction in Direction:
            seq = AronsonSequence('t', [], direction)
            with self.assertRaises(ValueError):
                seq.get_reference(1)
            seq.append_elements([1])
            _, ref = seq.get_reference(1)
            self.assertEqual(ref, Refer.BACKWARD)

    def test_cpy(self):
        seq = AronsonSequence('t', [1, 2], Direction.FORWARD)
        seq_cpy = seq.copy()
        self.assertEqual(seq, seq_cpy)
        seq.append_elements([3])
        # elements are mutable
        self.assertNotEqual(seq, seq_cpy)

    def test_contains(self):
        elems = [1, 2]
        seq = AronsonSequence('t', elems, Direction.FORWARD)
        for e in elems:
            self.assertIn(e, seq)
        emp_seq = AronsonSequence('t')
        for e in elems:
            self.assertNotIn(e, emp_seq)

    def test_append_element_wrapper(self):
        seq = AronsonSequence('t', [1, 2], Direction.FORWARD)
        # can use copy now
        seq_cpy = seq.copy()
        seq.set_elements([3, 4], append=True)
        # check these two do the same thing
        seq_cpy.append_elements([3, 4])
        self.assertEqual(seq, seq_cpy)

    def test_iter_and_getitem(self):
        seq = AronsonSequence('t', [5, 6, 7], Direction.FORWARD)
        self.assertEqual(list(seq), seq.get_elements())
        self.assertEqual(seq[1], 6)
        seq.set_elements([], append=False)
        self.assertEqual(list(seq), seq.get_elements())
        with self.assertRaises(IndexError):
            _ = seq[0]

    def test_len(self):
        seq = AronsonSequence('x', [10, 20], Direction.FORWARD)
        self.assertEqual(len(seq), 2)
        seq.set_elements([], append=False)
        self.assertEqual(len(seq), 0)

    def test_hash(self):
        seq = AronsonSequence('t')
        for other in {AronsonSequence('t', [], Direction.BACKWARD), AronsonSequence('t', [1]), AronsonSequence('d')}:
            self.assertEqual(hash(seq) == hash(other), seq == other)

    def test_large_sequence(self):
        # Create a very large sequence
        seq = AronsonSequence('t', list(range(1, 1001)), Direction.FORWARD)
        # Test some basic properties
        self.assertEqual(len(seq), 1000)
        self.assertEqual(seq[999], 1000)
        self.assertEqual(seq[0], 1)

    def test_flip_direction(self):
        seq = AronsonSequence('t', [1, 2, 3], Direction.FORWARD)
        seq.set_elements([4, 5, 6], append=True)

        # First set to other direction
        seq.flip_direction()
        seq.set_elements([7, 7, 8], append=True)
        last_elements = seq.get_elements()[-2:][::-1]
        self.assertIn(''.join(seq.n2w(elem) for elem in last_elements), seq.get_sentence())
        # Now check if the order of elements is consistent
        self.assertEqual(seq.get_sentence(),
                         AronsonSequence('t', [1, 2, 3, 4, 5, 6, 7, 8], Direction.BACKWARD).get_sentence())

    def test_is_hashable(self):
        seq = AronsonSequence('t', [1, 2, 3], Direction.FORWARD)
        s = set()
        # seq is hashable
        s.add(seq)
        self.assertIn(seq, s)

    def test_get_letter(self):
        seq = AronsonSequence('t', [1, 2, 3], Direction.FORWARD)
        # make sure display_letter() method works as expected
        self.assertNotEqual(seq.get_letter(), 't')

    def test_get_occurrences(self):
        ground_truths = [{1, 4, 9, 10}, {3, 4, 9, 12}]
        for truth, direction in zip(ground_truths, Direction):
            emp_seq = AronsonSequence('t', [], direction)
            self.assertEqual(emp_seq.get_occurrences(), truth)
            seq = AronsonSequence('t', [1], direction)
            target = {1} if direction == Direction.FORWARD else set()
            self.assertEqual(seq.get_occurrences(1), target)

    def test_index(self):
        for direction, elems in zip(Direction, [[1, 4, 11], [3, 4, 11]]):
            seq = AronsonSequence('t', [], direction)
            with self.assertRaises(ValueError):
                seq.__index__(1)
            seq = AronsonSequence('t', elems, direction)
            for i, elem in enumerate(seq):
                self.assertEqual(seq.__index__(elem), i)

    def test_is_permutation(self):
        elems = [1, 2]
        for direction in Direction:
            seq1 = AronsonSequence('t', [], direction)
            seq2 = AronsonSequence('t', [], direction.flip)
            with self.assertRaises(ValueError):
                # not allowed to compare sequences of conflicting direction
                AronsonSequence.is_permutation(seq1, seq2)
            seq1, seq2 = tuple(AronsonSequence('t', elems[::(1 if i else -1)], direction) for i in range(2))
            self.assertTrue(AronsonSequence.is_permutation(seq1, seq2))
            seq1.append_elements([3])
            # no longer permutation, but rather permuted subset
            self.assertFalse(AronsonSequence.is_permutation(seq1, seq2))

    def test_is_monotonic(self):
        all_elements = [([1], True), ([1, 2], True), ([1, 2, 3], False)]
        for elements, expected in all_elements:
            perms = {AronsonSequence('t', list(perm)) for perm in permutations(elements, len(elements))}
            # is_monotonic() must be of form (True, X), in third case some permutation is non-monotonic
            self.assertEqual(all(any(perm.is_monotonic()) for perm in perms), expected)

    if __name__ == '__main__':
        unittest.main()
