# Aronson Sequence Generator

![Tests](https://img.shields.io/badge/tests-95%25%20coverage-green)

A Python implementation of two classes representing self-referential sentences (`AronsonSequence`), and collections thereof (`AronsonSet`). 

The prototype which inspired the `AronsonSequence` class is [Aronson's sequence](https://oeis.org/A005224), first coined by J. K. Aronson, and quoted by D. R. Hofstadter in his book "*Methamagical Themas*" (1983). The `AronsonSequence` class takes this idea further by generalizing over all sentences of the form "Ω is the X, Y, Z... letter", where Ω $\in$ Σ is a letter in the alphabet and X, Y, Z are ordinals.

The  `AronsonSet` class constitutes a collection over `AronsonSequence` instances, where these are constrainted to be semantically correct (meaning ordinals X, Y, Z map to occurrences of the letter Ω in the sentence representation) with respect to the same letter Ω and scanning direction (left-to-right or right-to-left).  

See [this blogpost](https://ikavodo.github.io/aronson-1/) for more details.


---

## 🔧 Installation

### Clone and install in editable mode:

```bash
git clone https://github.com/ikavodo/aron_gen.git
cd aron_gen
pip install -e .
```

or as a standalone package via

```bash
pip install aron-gen
```

## Features

### `AronsonSequence` Class
Models self-referential sentences with:
- Positional reference tracking (forward/backward/self)
- Automatic sentence generation & validation
- Element manipulation (append/swap/clear)
- Direction flipping (forward ↔ backward)
- Comprehensive correctness checks

```python
# Create and validate sequence
from aron_gen.core import AronsonSequence
letter = 't'
aronson_initial = [1, 4, 11] # first three terms in Aronson's sequence
seq1 = AronsonSequence(letter, aronson_initial) # Forward
print(seq1)  # "T is the first, fourth, eleventh letter in this sentence, not counting commas and spaces"
seq1.is_correct()  # True
{seq1.get_ref(elem) for elem in aronson_initial} # {Refer.BACKWARD}
seq.is_prefix_complete() # True
seq.is_complete() # False
seq1.append_elements([16]) # Next element in Aronson
seq1.is_correct() # True
```

### `AronsonSet` Class
Manages collections of valid sequences with:

- Multiple generation strategies (pruned brute-force/fast rule-based)
- Set operations (union/intersection/difference)
- Filter operations (by element/reference/symmetry)

```python
from aron_gen.core import AronsonSet
# Generate and analyze sequences
aset1 = AronsonSet('t', Direction.BACKWARD) # Backward
empty_seq = aset1.peek() 
print(empty_seq) # "T is the letter"
seq1 = aset1.generate_aronson(3).pop() # AronsonSequence('t', [3, 4, 11], Direction.BACKWARD)
aset1.is_correct(seq1) # True
aset2 = AronsonSet('t') # Forward
aset2.is_correct(seq1) # False, sequence is incorrect w.r.t. set direction
```

## Advanced Usage
### Hybrid Generation
```python
aset = AronsonSet('t', Direction.BACKWARD)
aset.generate_full(2) # Exhaustively generate all correct AronsonSequences up to length 2
len(aset) # 67
aset_cpy = aset.copy()
aset.generate_fast(3, forward_generate=True)  # Optimized continuation to sequences of length 3
len(aset) # 198
aset_cpy.generate_full(3, error_rate = 0.25) # Find at least 75% of all correct sequences
len(aset_cpy) # 843 (out of 955)
```

### Set Operations
```python
# Combine sequence sets
seq1 = AronsonSequence('t', [1, 4, 11])
seq2 = AronsonSequence('t', [10, 12])
set1 = AronsonSet.from_sequence(seq1)
set2 = AronsonSet.from_sequence(seq2) 

# set operators |, &, -
union_set = set1 | set2
assert(union_set == AronsonSet.from_set({seq1, seq2})) # same as from_set() constructor 
intersection_set = set1 & set2 
assert(intersection_set == AronsonSet('t')) # intersection is empty forward set
difference_set = set1 - set2 
assert(difference_set == set1) # sets are complementary
```

### Filter Operations
```python
# Extract AronsonSequence instances from an AronsonSet instance via filtering
aset = AronsonSet('t')
n_iters = 2
aset.generate_full(n_iters)
len(aset) # 73 
filter1 = aset.filter_symmetric(n_iters) # get length-2 sequences for which all permutations also in set
filter2 = filter1.filter_elements(filter1.max) # get all such sequences containing maximum element
[seq for seq in filter2 if not seq.is_empty()]
# ["T is the thirty-second, thirty-third letter in this sentence, not counting commas and spaces",
# "T is the thirty-third, thirty-second letter in this sentence, not counting commas and spaces"]
```

### Testing Framework

Comprehensive test suite covering:

- Sequence validation and reference resolution
- Set operation correctness
- Edge case handling
- Performance benchmarks

Run tests with:
```bash
python -m unittest test_AronsonSet.py test_AronsonSequence.py
```
For running an optional, slower test regarding performance benchmarks run
```bash
RUN_OPTIONAL_TEST=True python -m unittest test_AronsonSet.py test_AronsonSequence.py
```

