I’ll treat this like building you a personal “algorithm playbook” for interviews.

I can’t literally cover *every* imaginable trick, but I *can* give you a taxonomy of patterns that, in practice, covers well over 90% of junior–mid interview problems if you really own them.

---

## 1. Master list of patterns (with priority for junior roles)

**Legend**

- **Priority**
    - 🟥 Must know (very likely to be asked)
    - 🟧 Good to know (sometimes; more often at mid-level / harder rounds)
    - 🟨 Rare for junior (only at algorithm-heavy companies / later rounds)

### Arrays & Strings

| # | Pattern | Priority | Typical problems |
| --- | --- | --- | --- |
| A1 | Hash map / set for counting & lookup | 🟥 | Two-sum, anagrams, frequency, dedupe |
| A2 | Sorting + scan / two pointers | 🟥 | 2-sum variants, intervals, dedup, merging |
| A3 | Two pointers (converging or same-direction) | 🟥 | Reverse, partition, remove/skip elements |
| A4 | Sliding window (fixed & variable) | 🟥 | Subarray/string with condition: max length, min length, count of distinct |
| A5 | Prefix sums (and sometimes hashmap with prefix) | 🟥 | Subarray sums, ranges, “number of subarrays with sum=k” |
| A6 | Binary search on sorted data | 🟥 | Search insert, first/last occurrence, boundaries |
| A7 | Binary search on answer (search space) | 🟧 | Min capacity, min/max feasible value problems |
| A8 | In-place partition / quickselect | 🟧 | Kth largest, top-K, quickselect style |
| A9 | String building with stack / list | 🟥 | Simplify path, backspace string compare |

### Linked Lists

| # | Pattern | Priority | Typical problems |
| --- | --- | --- | --- |
| L1 | Dummy head + single pass | 🟥 | Remove Nth from end, merge, delete nodes |
| L2 | Fast & slow pointers (Floyd’s) | 🟥 | Middle node, cycle detection, cycle start |
| L3 | Reverse linked list (iterative) | 🟥 | Reverse list/portion, k-group |
| L4 | Merge two sorted lists | 🟥 | MergeK via heap or divide & conquer |
| L5 | Reorder, partition, odd-even | 🟧 | Slightly more advanced pointer juggling |

### Stacks, Queues & Deques

| # | Pattern | Priority | Typical problems |
| --- | --- | --- | --- |
| S1 | Stack for parentheses / expression checking | 🟥 | Valid parentheses, min remove, RPN |
| S2 | Monotonic stack (increasing/decreasing) | 🟧 | Next greater element, daily temps, histogram |
| S3 | Deque as queue (BFS, sliding window max) | 🟥 | BFS, sliding window max/min |

### Trees (binary tree, BST)

| # | Pattern | Priority | Typical problems |
| --- | --- | --- | --- |
| T1 | Recursive DFS (pre/in/postorder) | 🟥 | Traversals, sum paths, validation |
| T2 | BFS (level order) | 🟥 | Level sums, connect siblings, zigzag |
| T3 | DFS with return value (height, balanced, etc.) | 🟥 | Balanced tree, diameter, path sums |
| T4 | BST properties + inorder traversal | 🟥 | Validate BST, kth smallest, LCA in BST |
| T5 | DFS with “carry info down and back up” | 🟧 | Path sum III, longest path with constraints |

### Graphs & Grids

| # | Pattern | Priority | Typical problems |
| --- | --- | --- | --- |
| G1 | Graph / grid BFS | 🟥 | Shortest path in unweighted graph, islands, rotten oranges |
| G2 | Graph / grid DFS | 🟥 | Count components, flood fill |
| G3 | Topological sort (Kahn or DFS) | 🟧 | Course schedule, ordering constraints |
| G4 | Dijkstra (priority queue shortest path) | 🟨 | Weighted shortest path |
| G5 | Union-Find (Disjoint Set Union) | 🟧 | Kruskal MST, number of components, connectivity |

### Recursion & Backtracking

| # | Pattern | Priority | Typical problems |
| --- | --- | --- | --- |
| B1 | Subsets / combinations | 🟥 | Subsets, combination sum, generate combos |
| B2 | Permutations | 🟥 | Permute array/unique permutations |
| B3 | Backtracking with constraints & pruning | 🟧 | N-Queens, word search, sudoku |
| B4 | “Pick/not pick” recursion | 🟥 | Many subset/combo variations |

### Dynamic Programming

| # | Pattern | Priority | Typical problems |
| --- | --- | --- | --- |
| D1 | 1D DP on sequence (linear DP) | 🟥 | Climbing stairs, house robber, max subarray |
| D2 | 2D DP on grid | 🟥 | Unique paths, min path sum, obstacles |
| D3 | 2D DP on strings | 🟥 | Edit distance, LCS, palindrome partitioning |
| D4 | Knapsack-style DP (0/1 & unbounded) | 🟧 | Coin change, subset sum |
| D5 | DP on trees / postorder DP | 🟧 | House robber III, tree DP variants |
| D6 | Bitmask DP / state compression | 🟨 | Traveling salesman, small-set problems |

### Greedy & Heaps

| # | Pattern | Priority | Typical problems |
| --- | --- | --- | --- |
| H1 | Greedy after sorting | 🟥 | Interval scheduling, meeting rooms, merging |
| H2 | Min-heap / max-heap (priority queue) | 🟥 | Top-K, K closest, merging sorted lists |
| H3 | Greedy with heap | 🟧 | Task scheduling, “pick next best feasible” |

### Math & Bit Manipulation

| # | Pattern | Priority | Typical problems |
| --- | --- | --- | --- |
| M1 | Simple math & counting (div/mod, sums) | 🟥 | FizzBuzz, digit sums, parity |
| M2 | Bit operations for sets / flags | 🟧 | Single number, subsets via bitmask |
| M3 | GCD / LCM, prime sieve | 🟨 | Rational arithmetic, primes problems |

### Data-structure heavy (often rare for junior)

| # | Pattern | Priority | Typical problems |
| --- | --- | --- | --- |
| X1 | Segment tree / Fenwick (BIT) | 🟨 | Range queries with updates |
| X2 | Trie | 🟧 | Word search, prefix matching |
| X3 | LRU/LFU cache implementation | 🟧 | List + hashmap combo |

**For a junior role, your main grind should be all 🟥 plus a decent amount of 🟧.**

🟨 is nice-to-have but usually not required.

---

## 2. Core templates in Python (+ how to tweak)

I’ll go through the **must-know** patterns and a subset of important 🟧 ones, each with:

- When to use
- Code template
- Tweaks for variations

I’ll keep the templates fairly generic so you can plug in problem logic.

---

### A1. Hash map / hash set

**When**: Lookups, counts, existence, frequency, mapping one thing to another.

```python
from collections import Counter, defaultdict

# Frequency of items
def freq(arr):
    count = Counter(arr)   # or defaultdict(int)
    # use count[x] as needed
    return count

# Checking if two strings are anagrams
def are_anagrams(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    count = Counter(s)
    for ch in t:
        count[ch] -= 1
        if count[ch] < 0:
            return False
    return True

# Generic "seen" set use
def has_duplicate(nums):
    seen = set()
    for x in nums:
        if x in seen:
            return True
        seen.add(x)
    return False

```

**Tweaks**

- Replace `Counter` with manual `defaultdict(int)` if you need more control.
- Use a `dict` where values are indices, counts, or any metadata.

---

### A2. Sorting + scan / two pointers

**When**: You care about order, proximity, or interval overlaps.

```python
def two_sum_sorted(nums, target):
    nums.sort()
    l, r = 0, len(nums) - 1
    while l < r:
        s = nums[l] + nums[r]
        if s == target:
            return True
        if s < target:
            l += 1
        else:
            r -= 1
    return False

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])
    return merged

```

**Tweaks**

- Change comparator (`key`) to sort by end time, length, etc.
- For k-sum problems, often: sort + fix some elements + two pointers.

---

### A3. Two pointers (same direction or converging)

**When**: Compressing array, removing elements in-place, partitioning.

```python
# Remove all instances of val in-place and return new length
def remove_element(nums, val):
    write = 0
    for read in range(len(nums)):
        if nums[read] != val:
            nums[write] = nums[read]
            write += 1
    return write

# Reverse array in-place
def reverse_array(nums):
    l, r = 0, len(nums) - 1
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1

```

**Tweaks**

- Change the condition deciding whether to advance one pointer, the other, or both.
- Use two pointers on strings (palindromes, skip non-alphanumeric, etc.).

---

### A4. Sliding window (fixed and variable)

**When**: Subarray / substring problems with constraints like size, sum, distinct count, etc.

```python
# Fixed-size window: max sum of subarray of size k
def max_sum_subarray(nums, k):
    window_sum = sum(nums[:k])
    best = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        best = max(best, window_sum)
    return best

# Variable-size window: longest substring with at most k distinct chars
def longest_k_distinct(s, k):
    from collections import defaultdict
    freq = defaultdict(int)
    left = 0
    best = 0

    for right, ch in enumerate(s):
        freq[ch] += 1
        while len(freq) > k:
            left_ch = s[left]
            freq[left_ch] -= 1
            if freq[left_ch] == 0:
                del freq[left_ch]
            left += 1
        best = max(best, right - left + 1)
    return best

```

**Tweaks**

- Fixed-size: when length is given; just shift and update.
- Variable-size: `while` condition changes to your constraint (sum > target, count > k, etc.).

---

### A5. Prefix sums (+ hashmap)

**When**: Subarray sums, range queries, count of subarrays with given property.

```python
# Count subarrays with sum == k
def subarray_sum(nums, k):
    from collections import defaultdict
    prefix = 0
    count = 0
    seen = defaultdict(int)
    seen[0] = 1

    for x in nums:
        prefix += x
        count += seen[prefix - k]
        seen[prefix] += 1
    return count

# Prefix sums for range queries
def build_prefix(nums):
    prefix = [0]
    for x in nums:
        prefix.append(prefix[-1] + x)
    return prefix

def range_sum(prefix, l, r):
    # sum nums[l:r+1]
    return prefix[r + 1] - prefix[l]

```

**Tweaks**

- Replace `k` with any condition expressible on prefix sums (e.g., mod, parity).
- Works in 2D as well: build prefix matrix.

---

### A6. Binary search on sorted array

**When**: Searching in sorted/monotonic data.

```python
def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# First index >= target
def lower_bound(nums, target):
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo  # position to insert

```

**Tweaks**

- Variants: first > target, last < target, first equal, etc.
- In rotated sorted array, binary search with extra logic for which side is sorted.

---

### A7. Binary search on answer (search space)

**When**: Answer is a number; condition is “can we achieve this?” and is monotonic in that number.

```python
def binary_search_answer(min_val, max_val, can):
    # can(x) returns True if it's feasible at value x
    lo, hi = min_val, max_val
    ans = max_val
    while lo <= hi:
        mid = (lo + hi) // 2
        if can(mid):
            ans = mid
            hi = mid - 1  # looking for minimal feasible
        else:
            lo = mid + 1
    return ans

```

**Tweaks**

- For “maximal feasible”, flip the direction (move `lo` when feasible).
- Common in: min capacity to ship packages, min time, split array largest sum, etc.

---

### L1 & L2. Linked list basics + fast/slow pointers

**When**: Any linked list manipulation or cycle detection.

```python
class ListNode:
    def __init__(self, val=0, nxt=None):
        self.val = val
        self.next = nxt

# Reverse list
def reverse_list(head):
    prev = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev

# Detect cycle
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# Remove Nth from end using dummy + fast/slow
def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)
    slow = fast = dummy
    for _ in range(n):
        fast = fast.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next
    slow.next = slow.next.next
    return dummy.next

```

**Tweaks**

- Fast/slow to find middle (`while fast and fast.next:`).
- Combine `reverse_list` with other ops (e.g., reorder list).

---

### S1. Stack for parentheses / expression

**When**: Matching pairs, nested structure, undo operations.

```python
def is_valid_parentheses(s):
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for ch in s:
        if ch in pairs.values():
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack

```

**Tweaks**

- Store indices instead of chars when you need positions.
- Use stack for any “last unmatched thing”.

---

### S2. Monotonic stack (increasing/decreasing)

**When**: Next greater element, daily temperatures, stock span, largest rectangle.

```python
def next_greater_elements(nums):
    res = [-1] * len(nums)
    stack = []  # indices of nums, stack is decreasing by value
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            idx = stack.pop()
            res[idx] = x
        stack.append(i)
    return res

```

**Tweaks**

- Use `<=` or `>=` depending on strict vs non-strict.
- For circular arrays, iterate twice with `i % n`.

---

### T1 & T2. Tree DFS & BFS

**When**: Almost any tree problem.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# DFS preorder
def preorder(root):
    res = []
    def dfs(node):
        if not node:
            return
        res.append(node.val)
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return res

# BFS level order
from collections import deque

def level_order(root):
    if not root:
        return []
    res = []
    q = deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(level)
    return res

```

**Tweaks**

- For DFS: modify `res` or return values to compute sums, heights, etc.
- For BFS: track `level` index, or track min/max, etc. at each level.

---

### T3. DFS with return value (height, balanced, etc.)

**When**: “For each node, answer depends on children.”

```python
# Check if tree is height-balanced
def is_balanced(root):
    def height(node):
        if not node:
            return 0
        left = height(node.left)
        if left == -1:
            return -1
        right = height(node.right)
        if right == -1:
            return -1
        if abs(left - right) > 1:
            return -1
        return 1 + max(left, right)
    return height(root) != -1

```

**Tweaks**

- Return a tuple (value, extra info) instead of single integer.
- Common pattern: compute something and early-return a sentinel when invalid.

---

### G1 & G2. Graph/grid BFS & DFS

**When**: Shortest paths in unweighted graphs; counting components; flood fill.

```python
from collections import deque

def bfs(start, graph):
    # graph: adjacency list dict {node: [neighbors]}
    q = deque([start])
    dist = {start: 0}
    while q:
        node = q.popleft()
        for nei in graph[node]:
            if nei not in dist:
                dist[nei] = dist[node] + 1
                q.append(nei)
    return dist

# DFS for connected components (graph or grid)
def dfs_graph(node, graph, visited):
    visited.add(node)
    for nei in graph[node]:
        if nei not in visited:
            dfs_graph(nei, graph, visited)

# Example for 2D grid (islands)
def num_islands(grid):
    if not grid:
        return 0
    m, n = len(grid), len(grid[0])
    visited = set()

    def dfs(r, c):
        if not (0 <= r < m and 0 <= c < n):
            return
        if grid[r][c] != '1' or (r, c) in visited:
            return
        visited.add((r, c))
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            dfs(r+dr, c+dc)

    count = 0
    for r in range(m):
        for c in range(n):
            if grid[r][c] == '1' and (r, c) not in visited:
                dfs(r, c)
                count += 1
    return count

```

**Tweaks**

- Change neighbor directions or movement rules.
- Track additional state (distance, layer) in BFS.

---

### B1 & B2. Backtracking for subsets / permutations / combinations

**When**: Generate all possibilities, or search with constraints.

```python
# Subsets
def subsets(nums):
    res = []
    subset = []

    def backtrack(i):
        if i == len(nums):
            res.append(subset.copy())
            return
        # choose nums[i]
        subset.append(nums[i])
        backtrack(i+1)
        # skip nums[i]
        subset.pop()
        backtrack(i+1)

    backtrack(0)
    return res

# Permutations
def permutations(nums):
    res = []
    used = [False] * len(nums)
    path = []

    def backtrack():
        if len(path) == len(nums):
            res.append(path.copy())
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack()
            path.pop()
            used[i] = False

    backtrack()
    return res

```

**Tweaks**

- Add pruning: skip when partial solution already invalid.
- For duplicates, sort and skip when `nums[i] == nums[i-1]` and `not used[i-1]`.

---

### D1. 1D DP on sequence

**When**: Simple recurrence along an array or index, often “take or skip”.

```python
# Climbing stairs: ways to reach step n taking 1 or 2 steps
def climb_stairs(n):
    if n <= 2:
        return n
    dp1, dp2 = 1, 2  # f(1), f(2)
    for _ in range(3, n+1):
        dp1, dp2 = dp2, dp1 + dp2
    return dp2

# House robber: max non-adjacent sum
def house_robber(nums):
    rob, not_rob = 0, 0
    for x in nums:
        new_rob = not_rob + x   # rob this house
        new_not = max(rob, not_rob)
        rob, not_rob = new_rob, new_not
    return max(rob, not_rob)

```

**Tweaks**

- Convert full `dp[i]` array into rolling variables to save memory.
- Many “linear DP” problems follow this pattern.

---

### D2. 2D DP on grid

**When**: Paths on a grid, obstacles, min path sum, etc.

```python
def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    return dp[m-1][n-1]

```

**Tweaks**

- Use `inf` or big positive numbers when a cell is blocked.
- You can compress to 1D DP if you only use `dp[j]` and `dp[j-1]`.

---

### D3. 2D DP on strings (edit distance / LCS)

**When**: Transform one string to another, subsequences, alignment.

```python
# Edit distance
def edit_distance(s, t):
    m, n = len(s), len(t)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # delete
                    dp[i][j-1],    # insert
                    dp[i-1][j-1]   # replace
                )
    return dp[m][n]

```

**Tweaks**

- LCS is similar but uses max and checks equality.
- Many string DP problems look very similar (just change the recurrence).

---

### H1. Greedy after sorting

**When**: You want an optimal strategy that’s “take best next” and provably works.

```python
# Minimum number of intervals to remove to avoid overlap
def erase_overlap_intervals(intervals):
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])  # sort by end
    count = 0
    _, prev_end = intervals[0]
    for start, end in intervals[1:]:
        if start < prev_end:
            count += 1    # need to remove this
        else:
            prev_end = end
    return count

```

**Tweaks**

- Change sort key or condition.
- Common with tasks: select max number of non-overlapping intervals, or min rooms needed.

---

### H2. Priority queue (heap)

**When**: Need “current min/max” repeatedly in sublinear time.

```python
import heapq

# Find k largest elements
def k_largest(nums, k):
    heap = []
    for x in nums:
        if len(heap) < k:
            heapq.heappush(heap, x)
        else:
            if x > heap[0]:
                heapq.heapreplace(heap, x)
    return heap  # contains k largest in min-heap form

# Merge k sorted lists (arrays)
def merge_k_sorted(lists):
    heap = []
    for i, arr in enumerate(lists):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))
    res = []
    while heap:
        val, i, j = heapq.heappop(heap)
        res.append(val)
        if j + 1 < len(lists[i]):
            heapq.heappush(heap, (lists[i][j+1], i, j+1))
    return res

```

**Tweaks**

- Use negative values to simulate max-heap (`heapq` is min-heap).
- Store tuples `(priority, metadata…)` to track extra info.

---

### G5. Union-Find (Disjoint Set Union)

**When**: Connectivity, components, merging sets, cycle detection in undirected graph.

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False  # already connected
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

```

**Tweaks**

- Use a map from arbitrary items to IDs if nodes aren’t 0..n-1.
- Count components by decrementing a counter whenever `union` succeeds.

---

### X3 (Good-to-know): LRU cache (linked list + hashmap)

**When**: System design-y questions; also appears as pure coding.

```python
class Node:
    def __init__(self, key=None, val=None):
        self.key = key
        self.val = val
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.map = {}
        self.head = Node()  # dummy head
        self.tail = Node()  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        p, n = node.prev, node.next
        p.next, n.prev = n, p

    def _add_to_front(self, node):
        n = self.head.next
        node.next = n
        node.prev = self.head
        self.head.next = node
        n.prev = node

    def get(self, key: int) -> int:
        if key not in self.map:
            return -1
        node = self.map[key]
        self._remove(node)
        self._add_to_front(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.map:
            node = self.map[key]
            node.val = value
            self._remove(node)
            self._add_to_front(node)
        else:
            if len(self.map) == self.cap:
                # remove LRU (node before tail)
                lru = self.tail.prev
                self._remove(lru)
                del self.map[lru.key]
            node = Node(key, value)
            self.map[key] = node
            self._add_to_front(node)

```

**Tweaks**

- LFU cache: more complex; rarely required for junior.

---

## 3. Patterns that are **unlikely** for junior roles

These appear, but relatively rarely (especially outside heavy algo companies / hard rounds):

- 🟨 **Segment tree / Fenwick (BIT)** – range update/query data structures.
- 🟨 **Bitmask DP** – advanced DP with 2^n states.
- 🟨 **Full-blown graph algorithms** beyond BFS/DFS/Dijkstra/Topo (e.g., Bellman-Ford, Floyd-Warshall, flows).
- 🟨 **Advanced geometry** – line intersection, convex hull.
- 🟨 **Very advanced greedy proofs** – e.g., obscure scheduling or matroid-like problems.

You should:

- Know what they *are* at a high level.
- Not burn tons of time deeply implementing them unless you target extremely competitive roles (top competitive programming-heavy teams).

---

## 4. How to use this to cover 90%+

If I were you and my job depended on it, I’d:

1. **Master all 🟥 patterns**
    - Be able to write each template from scratch, by memory.
    - Know 2–3 classic LeetCode problems that use each pattern.
2. **Then layer in key 🟧 patterns**
    - Binary search on answer, monotonic stack, union-find, basic string DP, topological sort, heaps.
3. **Practice mapping problems → patterns**
    - “I see sliding window here because they ask for longest/shortest substring with a constraint.”
    - “This looks like counting subarrays → prefix + hashmap.”
    - “Grid + shortest steps → BFS.”
4. **Review your own solutions**: after solving, ask:
    - Which pattern did I use?
    - Is there a more standard template that would have simplified this?

---

If you’d like, in the next step I can:

- Take a few LeetCode problems you’re worried about and classify them into these patterns,
- Or help you build a weekly training plan around this pattern list.