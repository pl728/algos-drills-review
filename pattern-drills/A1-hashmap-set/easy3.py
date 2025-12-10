"""
Easy 3: First Unique Character

Given: string `s`
Return: index of first character that appears exactly once, or -1 if none

Example:
  s = "leetcode" → 0  (because 'l' appears once)
  s = "loveleetcode" → 2  (because 'v' appears once)
  s = "aabb" → -1

Constraints: 1 ≤ len(s) ≤ 10^5
Target: O(n) time, O(1) space
"""

from collections import Counter

def first_unique_char(s):
    """
    Your solution here.
    Hint: Two passes - one to count, one to find first unique
    """
    c = Counter(s)
    for i, char in enumerate(s):
        if c[char] == 1:
            return i
    return -1
      


# Test cases
if __name__ == "__main__":
    assert first_unique_char("leetcode") == 0
    assert first_unique_char("loveleetcode") == 2
    assert first_unique_char("aabb") == -1
    assert first_unique_char("z") == 0
    print("All tests passed!")
