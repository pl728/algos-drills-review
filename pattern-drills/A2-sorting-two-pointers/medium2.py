"""
Medium 2: Merge Intervals

Given: array of intervals where intervals[i] = [start_i, end_i]
Return: array of merged overlapping intervals

Example:
  intervals = [[1,3],[2,6],[8,10],[15,18]]
  → [[1,6],[8,10],[15,18]]

  intervals = [[1,4],[4,5]]
  → [[1,5]]

  intervals = [[1,4],[0,4]]
  → [[0,4]]

Constraints:
  - 1 ≤ len(intervals) ≤ 10^4
  - intervals[i].length == 2
  - Intervals may be given in any order

Target: O(n log n) time, O(n) space
"""

def merge_intervals(intervals):
    """
    Merge all overlapping intervals.

    Args:
        intervals: list of [start, end] pairs

    Returns:
        list of merged intervals
    """
    pass


# Test cases
if __name__ == "__main__":
    assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
    assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
    assert merge_intervals([[1,4],[0,4]]) == [[0,4]]
    assert merge_intervals([[1,4],[2,3]]) == [[1,4]]

    print("All tests passed!")
