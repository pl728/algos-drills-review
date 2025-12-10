"""
Medium 1: Three Sum

Given: array of integers `nums`
Return: all unique triplets [nums[i], nums[j], nums[k]] where i≠j≠k and sum = 0

Example:
  nums = [-1,0,1,2,-1,-4]
  → [[-1,-1,2], [-1,0,1]]

  nums = [0,1,1]
  → []

  nums = [0,0,0]
  → [[0,0,0]]

Constraints:
  - 3 ≤ len(nums) ≤ 3000
  - Result must not contain duplicate triplets

Target: O(n²) time, O(1) extra space (output doesn't count)
"""

def three_sum(nums):
    """
    Find all unique triplets that sum to zero.

    Args:
        nums: list of integers

    Returns:
        list of lists: all unique triplets [a,b,c] where a+b+c=0
    """
    pass


# Test cases
if __name__ == "__main__":
    result = three_sum([-1,0,1,2,-1,-4])
    result_sorted = [sorted(t) for t in result]
    result_sorted.sort()
    expected = [[-1,-1,2], [-1,0,1]]
    assert result_sorted == expected

    assert three_sum([0,1,1]) == []
    assert three_sum([0,0,0]) == [[0,0,0]]

    print("All tests passed!")
