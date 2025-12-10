"""
Medium 3: Container With Most Water

Given: array of integers `height` where height[i] = height of line at position i
Return: maximum area of water that can be contained between two lines

Note: The width between lines i and j is (j - i)
      The height is min(height[i], height[j])
      Area = width × height

Example:
  height = [1,8,6,2,5,4,8,3,7]
  → 49
  (Between index 1 (height 8) and index 8 (height 7): width=7, height=7, area=49)

  height = [1,1]
  → 1

Constraints:
  - 2 ≤ len(height) ≤ 10^5
  - 0 ≤ height[i] ≤ 10^4

Target: O(n) time, O(1) space
"""

def max_area(height):
    """
    Find maximum area of water container.

    Args:
        height: list of integers representing line heights

    Returns:
        int: maximum area
    """
    pass


# Test cases
if __name__ == "__main__":
    assert max_area([1,8,6,2,5,4,8,3,7]) == 49
    assert max_area([1,1]) == 1
    assert max_area([4,3,2,1,4]) == 16
    assert max_area([1,2,1]) == 2

    print("All tests passed!")
