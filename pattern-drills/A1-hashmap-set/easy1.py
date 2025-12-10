"""
Easy 1: Contains Duplicate

Given: array of integers `nums`
Return: True if any value appears at least twice, False otherwise

Example:
  nums = [1,2,3,1] → True
  nums = [1,2,3,4] → False

Constraints: 1 ≤ len(nums) ≤ 10^5
Target: O(n) time, O(n) space
"""

def contains_duplicate(nums):
    """
    Your solution here.
    Hint: What data structure is good for checking "have I seen this before?"
    """
    return len(set(nums)) < len(nums)


# Test cases
if __name__ == "__main__":
    assert contains_duplicate([1,2,3,1]) == True
    assert contains_duplicate([1,2,3,4]) == False
    assert contains_duplicate([1,1,1,3,3,4,3,2,4,2]) == True
    print("All tests passed!")
