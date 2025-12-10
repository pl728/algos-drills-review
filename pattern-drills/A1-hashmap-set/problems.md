# Pattern A1: Hash Map / Set for Counting & Lookup 🟥

**Priority**: Must know (very likely to be asked)

**When to use**: Lookups, counts, existence checks, frequency tracking, mapping relationships

**Target complexity**: Usually O(n) time, O(n) space

---

## Easy Problems

### Easy 1: Contains Duplicate
```
Given: array of integers `nums`
Return: True if any value appears at least twice, False otherwise

Example:
  nums = [1,2,3,1] → True
  nums = [1,2,3,4] → False

Constraints: 1 ≤ len(nums) ≤ 10^5
Target: O(n) time, O(n) space
```

---

### Easy 2: Character Frequency Match
```
Given: two strings `s` and `t`
Return: True if they have the same character frequencies, False otherwise

Example:
  s = "listen", t = "silent" → True
  s = "hello", t = "world" → False

Constraints: 0 ≤ len(s), len(t) ≤ 10^4
Target: O(n) time, O(1) space (assuming fixed alphabet)
```

---

### Easy 3: First Unique Character
```
Given: string `s`
Return: index of first character that appears exactly once, or -1 if none

Example:
  s = "leetcode" → 0  (because 'l' appears once)
  s = "loveleetcode" → 2  (because 'v' appears once)
  s = "aabb" → -1

Constraints: 1 ≤ len(s) ≤ 10^5
Target: O(n) time, O(1) space
```

---

## Medium Problems

### Medium 1: Group Items by Frequency Signature
```
Given: array of strings `words`
Return: groups of strings that are anagrams of each other (any order)

Example:
  words = ["eat","tea","tan","ate","nat","bat"]
  → [["eat","tea","ate"], ["tan","nat"], ["bat"]]

Constraints: 1 ≤ len(words) ≤ 10^4, 1 ≤ len(word) ≤ 100
Target: O(n*k) time where k = max word length
```

---

### Medium 2: Two Sum - Return Indices
```
Given: array of integers `nums`, integer `target`
Return: indices of two numbers that add up to `target` (exactly one solution exists)

Example:
  nums = [2,7,11,15], target = 9 → [0,1]
  nums = [3,2,4], target = 6 → [1,2]

Constraints: 2 ≤ len(nums) ≤ 10^4, exactly one solution
Target: O(n) time, O(n) space
```

---

### Medium 3: Longest Substring Without Repeating Characters
```
Given: string `s`
Return: length of longest substring with all distinct characters

Example:
  s = "abcabcbb" → 3  (substring "abc")
  s = "bbbbb" → 1  (substring "b")
  s = "pwwkew" → 3  (substring "wke")

Constraints: 0 ≤ len(s) ≤ 5*10^4
Target: O(n) time, O(min(n, alphabet_size)) space
```

---

## Your Task

1. Try solving these problems in order (Easy → Medium)
2. Write your solutions in Python files in this folder
3. When ready, ask for the canonical solutions and explanations

**Key Questions to Ask Yourself:**
- What am I tracking? (existence, count, index, mapping?)
- Do I need a set or dict?
- Single pass or multiple passes?
- What's the lookup condition?

---

## Next Steps

After mastering this pattern, you'll move to:
- A2: Sorting + scan / two pointers
- A3: Two pointers (converging or same-direction)
- A4: Sliding window (fixed & variable)
- A5: Prefix sums
