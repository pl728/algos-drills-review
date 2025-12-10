"""
Easy 3: Remove Duplicates from Sorted Array

Given: sorted array of integers `nums`
Return: k = number of unique elements
        (modify nums in-place so first k elements are the unique elements in order)

Example:
  nums = [1,1,2] → k=2, nums = [1,2,_]
  nums = [0,0,1,1,1,2,2,3,3,4] → k=5, nums = [0,1,2,3,4,_,_,_,_,_]

Constraints:
  - 1 ≤ len(nums) ≤ 3*10^4
  - Array is sorted in non-decreasing order

Target: O(n) time, O(1) space
"""

def remove_duplicates(nums):
    """
    Remove duplicates in-place and return count of unique elements.

    Args:
        nums: sorted list of integers (modified in-place)

    Returns:
        int: number of unique elements
    """
    i = 0
    j = 1
    while j < len(nums):
      if nums[i] != nums[j]:
        i += 1
        nums[i] = nums[j]
      
      j += 1
    return i + 1


# Test cases
if __name__ == "__main__":
    nums = [1,1,2]
    k = remove_duplicates(nums)
    assert k == 2
    assert nums[:k] == [1,2]

    nums = [0,0,1,1,1,2,2,3,3,4]
    k = remove_duplicates(nums)
    assert k == 5
    assert nums[:k] == [0,1,2,3,4]

    nums = [1,1,1,1]
    k = remove_duplicates(nums)
    assert k == 1
    assert nums[:k] == [1]

    print("All tests passed!")
