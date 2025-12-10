"""
Medium 1: Group Items by Frequency Signature

Given: array of strings `words`
Return: groups of strings that are anagrams of each other (any order)

Example:
  words = ["eat","tea","tan","ate","nat","bat"]
  → [["eat","tea","ate"], ["tan","nat"], ["bat"]]

Constraints: 1 ≤ len(words) ≤ 10^4, 1 ≤ len(word) ≤ 100
Target: O(n*k) time where k = max word length
"""
import collections

def group_anagrams(words):
    """
    Your solution here.
    Hint: What can you use as a "signature" for anagrams?
          How do you group items by that signature?
    """
    d = {}
    for w in words:
        s = "".join(sorted(w))
        if s not in d:
            d[s] = []
        d[s].append(w)
    ans = []
    for k, l in d.items():
        ans.append(l)
    return ans

        

# Test cases
if __name__ == "__main__":
    result = group_anagrams(["eat","tea","tan","ate","nat","bat"])
    # Sort each group and the list of groups for consistent comparison
    result_sorted = sorted([sorted(group) for group in result])
    expected_sorted = sorted([sorted(["eat","tea","ate"]), sorted(["tan","nat"]), sorted(["bat"])])
    assert result_sorted == expected_sorted

    result2 = group_anagrams([""])
    assert result2 == [[""]]

    result3 = group_anagrams(["a"])
    assert result3 == [["a"]]

    print("All tests passed!")
