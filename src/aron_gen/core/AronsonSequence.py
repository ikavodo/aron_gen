import operator
from collections import defaultdict
from enum import Enum
from num2words import num2words
from typing import Optional

# global strings
REPR_PREFIX = " is the "
REPR_SUFFIX = " letter"

REPR_FORWARD = " in this sentence, not counting commas and spaces"
REPR_BACKWARD = "Not counting commas and spaces, in this sentence backwards "


class Refer(Enum):
    """
    Enum for keeping track of where elements refer to relatively to their ordinal representation.

    BACKWARD: Refers to a previous position.
    SELF: Refers to its own position.
    FORWARD: Refers to a later position.
    """
    BACKWARD = 1
    SELF = 2
    FORWARD = 3


class Direction(Enum):
    """
    Direction by which ordinals refer to indices in the string representation of the sequence
    FORWARD: left to right
    BACKWARD: right to left
    """
    FORWARD = 1
    BACKWARD = 2

    @property
    def flip(self):
        return Direction.BACKWARD if self == Direction.FORWARD else Direction.FORWARD


class VerificationError(Exception):
    """
    Custom exception raised when AronsonSequence verification (is_correct()) fails.
    :param message: The error message to be shown.
    :param input_seq: The input data that caused the failure.
    """

    def __init__(self, message="Verifier failed", input_seq=None):
        self.message = message
        self.input_seq = input_seq
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}: {self.input_seq}"


class AronsonSequence:
    """
    Represents an Aronson sequence, which is a sequence of elements with each element referring to an ordinal appearing
    within a generated sentence of form:
    See https://ikavodo.github.io/aronson-1/ for more information
    """
    # ordinal representation dictionary for memoization
    num2words_dict = defaultdict(tuple)

    @classmethod
    def n2w(cls, n, stripped=True):
        """
        Memoized conversion of a number to its ordinal word representation (stripped of spaces, commas, etc).
        """
        if n not in cls.num2words_dict:
            os = num2words(n, ordinal=True)
            # tuple: (original, stripped)
            cls.num2words_dict[n] = (os, os.replace(" and", "").replace(", ", "").replace(" ", "").replace("-", ""))
        return cls.num2words_dict[n][0] if not stripped else cls.num2words_dict[n][1]

    def _verifier(self):

        def _check_element(elem):
            elem_idx = elem - 1
            if s_buffer[elem_idx] != self.letter:
                raise VerificationError("Generated sentence is semantically incorrect")

        prefix = self.letter + REPR_PREFIX.replace(" ", "")
        suffix = REPR_SUFFIX.replace(" ", "")
        s_buffer = prefix if self.direction == Direction.FORWARD else suffix[::-1]
        forward_idxs = []  # stack for forward references to resolve
        for elem in self.elements:
            ordinal_str = self.n2w(elem, stripped=True)
            s_buffer += ordinal_str if self.direction == Direction.FORWARD else ordinal_str[::-1]
            if elem > len(s_buffer):
                forward_idxs.append(elem)
                continue
            else:
                _check_element(elem)

        # add last chunk and check forward referring elements
        s_buffer += suffix if self.direction == Direction.FORWARD else prefix[::-1]
        for elem in forward_idxs:
            if elem > len(s_buffer):
                # string complete, index over-reach
                raise VerificationError("Generated sentence is semantically incorrect")
            _check_element(elem)

    def __init__(self, letter: str, elements: Optional[list[int]] = None, direction: Direction = Direction.FORWARD,
                 *, check_semantics: bool = False):
        """
        Initializes the AronsonSequence with the given letter, elements, and direction.

        :param letter: The letter used for the sequence.
        :param elements: A list of elements (positive integers) in the sequence.
        :param direction: of the sequence
        :param check_semantics: of correctness
        :param
        """
        elements = elements if elements is not None else []
        # error checking
        self.check_letter(letter)
        self.check_direction(direction)
        self.letter = letter.lower()
        self.direction = direction
        self.check_elements(elements)

        # Deduplicate while preserving order
        seen = set()
        self.elements = [x for x in elements if not (x in seen or seen.add(x))]
        self.prefix = max(self.elements) if self.elements else 0

        if check_semantics:
            try:
                # get full sentence with no spaces or commas
                self._verifier()
            except VerificationError:
                raise

        # build internal fields relating to sentence representation
        self._update_sentence()

    def _update_sentence(self, elements=None):
        """
        Updates the string representation of the sequence, its sanitized form, occurrence set,
        and optionally builds or updates the referral dictionary.
        :param elements: Optional iterable of new elements to update in refer_dict;
                         if None, rebuild refer_dict from scratch using self.elements.
        """
        self.sentence_repr = self._build_sentence_repr()
        self.sentence = self.sentence_repr.replace(", ", "").replace(" ", "").replace("-", "")
        s = self.sentence if self.direction == Direction.FORWARD else self.sentence[::-1]
        self.occurrences = {i + 1 for i, char in enumerate(s) if char == self.letter}

        target_elements = self.elements if elements is None else elements
        updates = {x: self._set_refer_val(x) for x in target_elements}
        if elements is None:
            self.refer_dict = updates
        else:
            self.refer_dict.update(updates)

    def _build_sentence_repr(self):
        """
        Returns the human-readable string representation of the AronsonSequence.
        :return: The string representation of the sequence.
        """
        # reverse order if backwards
        elem_ord = self.elements if self.direction == Direction.FORWARD else self.elements[::-1]
        str_body = ', '.join(self.n2w(i, stripped=False) for i in elem_ord)
        return f"{self.letter + REPR_PREFIX}{str_body}{REPR_SUFFIX}".replace("  ", " ")

    def _set_refer_val(self, elem):
        """
        Determines the referral type for a specific index in the sequence.
        :param elem: The index of the element in the sequence.
        :return: pos: position of ordinal representation within string
                ref: where element points to within the string compared to the position
        """
        target_elem = elem - 1
        rep = self.n2w(elem, stripped=True)
        pos = self.sentence.find(rep) if self.direction == Direction.FORWARD else self.sentence[::-1].find(
            rep[::-1])
        end = pos + len(rep)
        if target_elem < pos:
            ref = Refer.BACKWARD
        elif pos <= target_elem < end:
            ref = Refer.SELF
        else:
            ref = Refer.FORWARD
        # return position of ordinal within string representation, and referral type as a tuple.
        return range(pos, end), ref

    @property
    def display_letter(self):
        """
        uppercase letter for representation
        :return: uppercase
        """

        return self.letter.upper()

    @staticmethod
    def check_direction(direction: Direction):
        """
        Make sure direction is a Direction Enum Type
        :param direction: of sequence
        """
        if not isinstance(direction, Direction):
            raise ValueError(f"Invalid direction value: {direction}. Must be a Direction.")

    @staticmethod
    def check_elements(elements):
        """
        Make sure elements is a list with positive integers. Duplicates are allowed but filtered out
        :param elements: of sequence
        """

        if not isinstance(elements, list) or not all(isinstance(i, int) and i > 0 for i in elements):
            raise ValueError(f"Invalid elements: {elements}. Must be a list of positive integers.")

    @staticmethod
    def check_letter(letter):
        """
        check letter input is single alpha character
        :param letter:
        """
        if not isinstance(letter, str) or len(letter) != 1 or not letter.isalpha():
            raise ValueError(f"Invalid letter: {letter!r}. Must be a single alphabetic character.")

    def has_forward_ref(self):
        """
        Checks if there are any forward referring elements in the sequence.
        :return: True if there are forward referring elements, False otherwise.
        """
        return any(ref == Refer.FORWARD for _, ref in self.refer_dict.values())

    def get_occurrences(self, idx=None):
        """
        Returns the 1-based positions of `self.letter` in the sentence, respecting direction.
        :return: A list of positions where the letter occurs.
        """
        return {i for i in self.occurrences if i <= idx} if idx is not None else self.occurrences

    @classmethod
    def is_permutation(cls, seq1: 'AronsonSequence', seq2: 'AronsonSequence'):
        """ """
        if seq1.display_letter != seq2.get_letter() or seq1.direction != seq2.get_direction():
            raise ValueError("Letter and direction must be the same")
        return set(seq1.get_elements()) == set(seq2.get_elements())

    def is_complete(self):
        """
        Checks if the sequence is self-contained, i.e., the positions of the letter in the sentence match the elements.
        :return: True if the sequence is self-contained, False otherwise.
        """
        return self.get_occurrences() == set(self.elements)

    def is_prefix_complete(self):
        """
        Checks if substring up to the largest ordinal is complete, i.e.,
        the positions of the letter in the sentence match the elements.
        :return: True if the sequence is prefix-complete, False otherwise.
        """
        return self.get_occurrences(idx=self.prefix) == set(self.elements)

    def get_prefix_missing(self):
        """
        Used for finding missing occurrences of the letter backwards from maximum index
        :return: True if the sequence is prefix-complete, False otherwise.
        """
        return self.get_occurrences(idx=self.prefix).difference(set(self.elements))

    def is_empty(self):
        """ if is empty instance"""
        return not self.elements

    def is_correct(self):
        """
        Verifies if the sequence is valid by checking if all elements occur at the correct positions.
        :return: True if the sequence is valid, False otherwise.
        """
        return all(ind in self.get_occurrences() for ind in self.elements)

    def is_monotonic(self):
        """
        Check if sequence is monotonically increasing or decreasing.
        :return: (True, X) if the sequence is monotonic, else (False, None).
        More specifically: X = +/- 1 depending on direction
        """
        if len(self.elements) < 2:
            # None is trivial monotonic type
            return True, None
        op_dict = {operator.lt: 1, operator.gt: -1}
        for op, direction in op_dict.items():
            if all(op(a, b) for a, b in zip(self.elements, self.elements[1:])):
                # keep track of monotonicity direction
                return True, direction
        return False, None

    # setters
    def set_elements(self, new_elements: list[int] = None, append=False):
        """
        Setter for the elements of the sequence. Updates the sentence, sentence_repr, and refer_dict.
        :param new_elements: The new elements for the sequence.
        :param append: Whether to append or replace elements.
        """
        new_elements = new_elements if new_elements is not None else []
        self.check_elements(new_elements)
        seen = set()

        if append:
            # ignore repeating elements
            filtered = [x for x in new_elements if not (x in seen or seen.add(x) or x in self.refer_dict.keys())]
            if not filtered:
                # do nothing
                return
            self.elements.extend(filtered)

        else:
            # Replace elements
            self.elements = [x for x in new_elements if not (x in seen or seen.add(x))]

        # Reduced computation if appending
        self._update_sentence(new_elements if append else None)
        self.prefix = max(self.elements) if self.elements else 0

    def append_elements(self, new_elements: list[int]):
        """
        Wrapper function to append new elements to the sequence.
        :param new_elements: A list of new elements to append.
        """
        self.set_elements(new_elements, append=True)

    def clear(self):
        """ wrapper for clearing the sequence of elements"""
        self.set_elements([])

    def set_letter(self, letter):
        """
        Sets the letter for the sequence and updates the sentence accordingly.
        :param letter: The new letter to set for the sequence.
        """
        self.check_letter(letter)
        self.letter = letter.lower()
        # need to update sentence-related fields
        self._update_sentence()

    def flip_direction(self):
        """
        Flips the direction of the sequence
        Updates the sentence accordingly.
        """
        self.direction = self.direction.flip
        # need to update sentence-related fields
        self._update_sentence()

    # getters

    def get_reference(self, elem):
        """
        given an element, if it is within the sequence, return the reference type
        :param elem: to be checked
        :return: tuple of types (range, refer type)
        """
        if elem not in self.refer_dict.keys():
            raise ValueError("Element not in sequence")
        return self.refer_dict[elem]

    def get_letter(self):
        """
        :return: letter in upper case
        """
        return self.display_letter

    def get_sentence(self):
        """
        Getter for the string representation of the sequence.
        :return: The string representation of the sequence.
        """
        return self.sentence

    def get_direction(self):
        """
        Getter for the direction of the sequence
        :return: Direction.
        """
        return self.direction

    def get_elements(self):
        """
        Getter for the elements of the sequence.
        :return: The elements of the sequence.
        """
        return self.elements

    def get_refer_dict(self):
        """
        Getter for the referral dictionary.
        :return: The referral dictionary.
        """
        return self.refer_dict

    def get_prefix(self):
        """
        Getter for the prefix.
        :return: The referral dictionary.
        """
        return self.prefix

    # operator overloading
    def __add__(self, other):
        """
        + operator: returns a new sequence with appended elements.
        Supports AronsonSequence, int, or list[int].
        """
        new_seq = self.copy()
        new_seq._append_flexible(other)
        return new_seq

    def __iadd__(self, other):
        """
        += operator: appends elements to the current sequence.
        """
        self._append_flexible(other)
        return self

    def _append_flexible(self, other):
        """
        Internal method to append various types to the sequence.
        """
        if isinstance(other, AronsonSequence):
            if self.display_letter != other.get_letter() or self.direction != other.get_direction():
                raise ValueError("Letter and direction must be the same")
            self.append_elements(other.get_elements())

        elif isinstance(other, int):
            self.append_elements([other])

        elif isinstance(other, list):
            self.check_elements(other)
            self.append_elements(other)

        else:
            raise TypeError("Unsupported operand type for +")

    def __repr__(self):
        """
        repr()/str() operator
        :return: The human-readable string representation of the sequence.
        """

        # return upper case
        s = self.display_letter + self.sentence_repr[1:]
        if not self.elements:
            # The empty sequence is the same in both directions
            return s
        return s + REPR_FORWARD if self.direction == Direction.FORWARD else REPR_BACKWARD + s

    def __eq__(self, other):
        """
        = operator, Compares two Aronson sequences for equality based on letter, elements, and direction.
        :param other: The other AronsonSequence to compare with.
        :return: True if the sequences are equal, False otherwise.
        """
        return isinstance(other, AronsonSequence) and self.elements == other.get_elements() and \
            self.display_letter == other.get_letter() and self.direction == other.get_direction()

    def copy(self):
        """
        .copy() operator
        :return: identical AronsonSequence instance
        """
        return AronsonSequence(
            self.letter,
            self.elements.copy(),  # avoid sharing mutable state
            self.direction
        )

    def __hash__(self):
        """
        hash() operator, returns a hash value for the AronsonSequence object based on its letter, elements,
        and direction.
        :return: A hash value for the sequence.
        """
        return hash((tuple(self.elements), self.letter, self.direction))

    def __iter__(self):
        """
        for operator, returns an iterator over the elements of the sequence.
        :return: An iterator for the elements.
        """
        return iter(self.elements)

    def __contains__(self, item):
        """ in operator, return True if element in sequence otherwise False"""
        return item in self.elements

    def __len__(self):
        """
        len() operator, returns the length of the Aronson sequence (i.e., the number of elements).
        :return: The length of the Aronson sequence.
        """
        return len(self.elements)

    def __index__(self, val):
        """ find index of element in elements"""
        try:
            return self.elements.index(val)
        except ValueError:
            raise ValueError("Element is not in sequence")

    def __getitem__(self, index: int):
        """
        [] operator, returns the element at a specified position in the Aronson sequence.
        :param index: The index position to retrieve.
        :return: The element at the specified position.
        """
        return self.elements[index]
