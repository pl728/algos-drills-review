"""
Easy 2: Character Frequency Match

Given: two strings `s` and `t`
Return: True if they have the same character frequencies, False otherwise

Example:
  s = "listen", t = "silent" → True
  s = "hello", t = "world" → False

Constraints: 0 ≤ len(s), len(t) ≤ 10^4
Target: O(n) time, O(1) space (assuming fixed alphabet)
"""
from collections import Counter

def are_anagrams(s, t):
    """
    Your solution here.
    Hint: How do you compare character frequencies?
    """
    return Counter(s) == Counter(t)


# Test cases
if __name__ == "__main__":
    assert are_anagrams("listen", "silent") == True
    assert are_anagrams("hello", "world") == False
    assert are_anagrams("anagram", "nagaram") == True
    assert are_anagrams("rat", "car") == False
    assert are_anagrams("", "") == True
    print("All tests passed!")
