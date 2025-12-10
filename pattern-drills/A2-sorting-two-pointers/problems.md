# Pattern A2: Sorting + Scan / Two Pointers 🟥

**Priority**: Must know (very likely to be asked)

**When to use**: Problems involving order, proximity, pairs/triplets with sum constraints, interval overlaps, or merging

**Key insight**: Sorting often reveals structure that makes O(n) scanning or two-pointer approach possible

**Target complexity**: O(n log n) for sort + O(n) for scan = O(n log n) total

---

## Easy Problems

### Easy 1: Two Sum II - Sorted Array
```
Given: sorted array of integers `nums`, integer `target`
Return: indices (1-indexed) of two numbers that add up to `target`
        (exactly one solution exists, cannot use same element twice)

Example:
  nums = [2,7,11,15], target = 9 → [1,2]
  nums = [2,3,4], target = 6 → [1,3]
  nums = [-1,0], target = -1 → [1,2]

Constraints:
  - 2 ≤ len(nums) ≤ 3*10^4
  - Array is sorted in non-decreasing order
  - Exactly one solution exists

Target: O(n) time, O(1) space
```

---

### Easy 2: Merge Sorted Array
```
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
```

---

### Easy 3: Remove Duplicates from Sorted Array
```
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
```

---

## Medium Problems

### Medium 1: Three Sum
```
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
```

---

### Medium 2: Merge Intervals
```
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
```

---

### Medium 3: Container With Most Water
```
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
```

---

## Your Task

1. Solve these problems in order (Easy → Medium)
2. Write your solutions in Python files in this folder
3. When ready, ask for canonical solutions and pattern analysis

**Key Questions to Ask Yourself:**
- Should I sort first? What do I gain from sorting?
- Can I use two pointers (converging from ends, or same direction)?
- What's my invariant? (What stays true as pointers move?)
- How do I handle duplicates?
- Am I modifying in-place or creating new output?

---

## Pattern Variations You'll See

1. **Two pointers on already-sorted data** (Easy 1)
2. **Merging two sorted sequences** (Easy 2)
3. **Deduplication via write pointer** (Easy 3)
4. **Sort + fix one element + two pointers** (Medium 1: Three Sum)
5. **Sort + scan for overlap** (Medium 2: Intervals)
6. **Two pointers without sorting** (Medium 3: max area between boundaries)

---

## Next Steps

After solving, you'll see how A2 connects to:
- **A3**: Two pointers (pure, no sorting needed)
- **A4**: Sliding window (when you need contiguous subarrays/substrings)
- **A6**: Binary search (when sorted data allows log n search)
