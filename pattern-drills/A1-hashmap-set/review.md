# A1 Review: Hash Map / Set for Counting & Lookup ✅

**Status**: Completed
**Date**: 2025-12-10

---

## Pattern Definition

**Core concept**: Use hash-based data structures (dict, set, Counter) for O(1) lookup/insertion.

**When to use**:
- Checking existence ("have I seen this?")
- Counting frequencies
- Storing metadata (indices, positions, mappings)
- Fast lookups while iterating

**Complexity**: Usually O(n) time, O(n) space

---

## Problems Solved (6/6)

### Easy 1: Contains Duplicate ✓
**Problem**: Check if array has duplicates
**Solution**: `len(set(nums)) < len(nums)`
**Key insight**: Set automatically deduplicates; size comparison reveals duplicates

---

### Easy 2: Character Frequency Match (Anagrams) ✓
**Problem**: Check if two strings have same character frequencies
**Solution**: `Counter(s) == Counter(t)`
**Key insight**: Counter comparison checks exact frequency match

**Alternative manual approach**:
```python
freq = {}
for ch in s:
    freq[ch] = freq.get(ch, 0) + 1
# Then decrement for t and check for negatives
```

---

### Easy 3: First Unique Character ✓
**Problem**: Return index of first character appearing exactly once
**Solution**:
```python
c = Counter(s)
for i, char in enumerate(s):
    if c[char] == 1:
        return i
return -1
```
**Key insight**: Two passes - count first, then find first unique by iterating the *string* (not the Counter)

**Pitfall avoided**: Iterating `Counter.items()` instead of the string loses index information

---

### Medium 1: Group Anagrams ✓
**Problem**: Group strings that are anagrams of each other
**Solution**:
```python
from collections import defaultdict
d = defaultdict(list)
for w in words:
    sig = ''.join(sorted(w))
    d[sig].append(w)
return list(d.values())
```
**Key insight**: Use sorted string as signature (canonical form); group by signature

**Pitfall encountered**: `list.append()` returns `None`, not the list
- ❌ `d[s] = d.get(s, []).append(w)`
- ✓ Use `defaultdict(list)` or two-line pattern

---

### Medium 2: Two Sum ✓
**Problem**: Find two indices where nums[i] + nums[j] = target
**Solution**:
```python
d = {}
for i, n in enumerate(nums):
    if target - n in d:
        return [d[target - n], i]
    d[n] = i
```
**Key insight**: Pattern evolution - store **indices**, not counts; lookup **complement** (target - n)

**Why check before adding**: Prevents using same element twice

---

### Medium 3: Longest Substring Without Repeating Characters ✓
**Problem**: Find length of longest substring with all distinct chars
**Solution**:
```python
d = {}
l = 0
longest = 0
for i, char in enumerate(s):
    if char in d:
        l = max(l, d[char] + 1)  # Move window start forward
    longest = max(longest, i - l + 1)
    d[char] = i
return longest
```
**Key insight**: **Hybrid pattern** - hash map + sliding window
- Hash map tracks last seen index of each char
- Sliding window: `[l, i]` expands right, contracts left when duplicate found

**Critical detail**: `l = max(l, d[char] + 1)` prevents moving window start backwards

**Not DP**: This is greedy sliding window, not dynamic programming (no recurrence relation)

---

## Pattern Evolution Observed

1. **Existence checking** (Easy 1): `set`
2. **Frequency counting** (Easy 2, 3): `Counter` or `dict` with counts
3. **Grouping by signature** (Medium 1): `defaultdict(list)` with computed key
4. **Complement lookup** (Medium 2): `dict` storing indices, checking `target - n`
5. **Position tracking + window** (Medium 3): `dict` storing last index + two-pointer window

---

## Common Pitfalls & Solutions

| Pitfall | Solution |
|---------|----------|
| `KeyError` when accessing missing key | Use `.get(key, default)` or `defaultdict` |
| `list.append()` returns `None` | Don't assign result of `.append()` |
| Dict iteration order (pre-3.7) | Iterate source data, not dict, when order matters |
| Counter not sorted | Use `sorted(string)` as signature for anagrams |
| Window start moving backwards | `l = max(l, new_position)` in sliding window |

---

## Key Takeaways

1. **Set vs Dict vs Counter**:
   - `set`: existence only
   - `dict`: custom key-value mapping
   - `Counter`: frequency counting (subclass of dict)

2. **The `.get(key, default)` pattern**:
   ```python
   freq[ch] = freq.get(ch, 0) + 1
   ```

3. **defaultdict for grouping**:
   ```python
   from collections import defaultdict
   groups = defaultdict(list)
   groups[key].append(value)  # Auto-creates empty list
   ```

4. **Hash map evolves**:
   - Count → Index → Position tracking
   - Standalone → Combined with other patterns (sliding window)

5. **Pattern recognition**:
   - "Find pair/complement" → Hash map for lookup
   - "Group by property" → Hash map with computed key
   - "No repeating elements" → Hash map + sliding window

---

## Next Pattern

**A2: Sorting + Two Pointers** - Moving from hash-based O(1) lookup to sorted O(log n) or O(n) scan patterns.
