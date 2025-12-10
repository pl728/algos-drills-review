"""
Medium 3: Longest Substring Without Repeating Characters

Given: string `s`
Return: length of longest substring with all distinct characters

Example:
  s = "abcabcbb" → 3  (substring "abc")
  s = "bbbbb" → 1  (substring "b")
  s = "pwwkew" → 3  (substring "wke")

Constraints: 0 ≤ len(s) ≤ 5*10^4
Target: O(n) time, O(min(n, alphabet_size)) space
"""

def longest_substring_without_repeating(s):
    """
    Your solution here.
    Hint: This combines hash map with sliding window.
          Track the last seen index of each character.
          When you see a repeat, where should you move your window start?
    """
    d = {}
    l = 0
    r = 0
    longest = 0
    for i, char in enumerate(s):

      if char in d:
        l = max(l, d[char] + 1)
      
      r += 1
      longest = max(longest, r - l)
      d[char] = i
    return longest





# Test cases
if __name__ == "__main__":
    assert longest_substring_without_repeating("abcabcbb") == 3
    assert longest_substring_without_repeating("bbbbb") == 1
    assert longest_substring_without_repeating("pwwkew") == 3
    assert longest_substring_without_repeating("") == 0
    assert longest_substring_without_repeating("dvdf") == 3
    print("All tests passed!")
