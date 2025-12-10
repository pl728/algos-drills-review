"""
Easy 2: Merge Sorted Array

Given: two sorted arrays `nums1` (has extra space at end) and `nums2`
       integers `m` and `n` (actual elements in nums1 and nums2)
Return: nothing (modify nums1 in-place to contain merged sorted result)

Example:
  nums1 = [1,2,3,0,0,0], m = 3
  nums2 = [2,5,6], n = 3
  → nums1 becomes [1,2,2,3,5,6]

  nums1 = [1], m = 1, nums2 = [], n = 0
  → nums1 stays [1]

Constraints:
  - nums1.length == m + n
  - 0 ≤ m, n ≤ 200
  - Both arrays are sorted in non-decreasing order

Target: O(m+n) time, O(1) space
"""

def merge(nums1, m, nums2, n):
    """
    Merge nums2 into nums1 in-place.

    Args:
        nums1: sorted list with extra space at end
        m: number of actual elements in nums1
        nums2: sorted list
        n: number of elements in nums2

    Returns:
        None (modifies nums1 in-place)
    """
    i, j = m-1, n-1
    w = len(nums1) - 1
    while i >= 0 and j >= 0:
      if nums1[i] >= nums2[j]:
        nums1[w] = nums1[i]
        i -= 1
      else:
        nums1[w] = nums2[j]
        j -= 1
      w -= 1
    while j >= 0:
      nums1[w] = nums2[j]
      w -= 1
      j -= 1

# Test cases
if __name__ == "__main__":
    nums1 = [1,2,3,0,0,0]
    merge(nums1, 3, [2,5,6], 3)
    assert nums1 == [1,2,2,3,5,6]

    nums1 = [1]
    merge(nums1, 1, [], 0)
    assert nums1 == [1]

    nums1 = [0]
    merge(nums1, 0, [1], 1)
    assert nums1 == [1]

    print("All tests passed!")
