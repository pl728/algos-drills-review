"""
Easy 1: Two Sum II - Sorted Array

Given: sorted array of integers `nums`, integer `target`
Return: indices (1-indexed) of two numbers that add up to `target`

Example:
  nums = [2,7,11,15], target = 9 → [1,2]
  nums = [2,3,4], target = 6 → [1,3]
  nums = [-1,0], target = -1 → [1,2]

Constraints:
  - 2 ≤ len(nums) ≤ 3*10^4
  - Array is sorted in non-decreasing order
  - Exactly one solution exists

Target: O(n) time, O(1) space
"""

def two_sum(nums, target):
    """
    Find two indices (1-indexed) that sum to target.

    Args:
        nums: sorted list of integers
        target: integer target sum

    Returns:
        list of two indices (1-indexed)
    """
    l, r = 1, len(nums)
    while l < r:
      if nums[l-1] + nums[r-1] < target:
        l += 1
      elif nums[l-1] + nums[r-1] > target:
        r -=1 
      else:
        return [l, r]


# Test cases
if __name__ == "__main__":
    assert two_sum([2,7,11,15], 9) == [1,2]
    assert two_sum([2,3,4], 6) == [1,3]
    assert two_sum([-1,0], -1) == [1,2]
    print("All tests passed!")
