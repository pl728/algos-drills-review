"""
Medium 2: Two Sum - Return Indices

Given: array of integers `nums`, integer `target`
Return: indices of two numbers that add up to `target` (exactly one solution exists)

Example:
  nums = [2,7,11,15], target = 9 → [0,1]
  nums = [3,2,4], target = 6 → [1,2]

Constraints: 2 ≤ len(nums) ≤ 10^4, exactly one solution
Target: O(n) time, O(n) space
"""

def two_sum(nums, target):
    """
    Your solution here.
    Hint: For each number x, what are you looking for?
          How can you check if you've seen it already?
    """
    d = {}
    for i, n in enumerate(nums):
      if target - n in d:
        return [d[target-n], i]
      
      d[n] = i


# Test cases
if __name__ == "__main__":
    assert two_sum([2,7,11,15], 9) == [0,1]
    assert two_sum([3,2,4], 6) == [1,2]
    assert two_sum([3,3], 6) == [0,1]
    print("All tests passed!")
