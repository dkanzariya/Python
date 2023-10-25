# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reorderList(self, head):
        if not head or not head.next or not head.next.next:
            return head

        # Step 1: Split the linked list into two halves
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # Step 2: Reverse the second half of the linked list
        prev, curr = None, slow
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node

        # Step 3: Merge the two halves
        p1, p2 = head, prev
        while p2.next:
            temp1, temp2 = p1.next, p2.next
            p1.next = p2
            p2.next = temp1
            p1, p2 = temp1, temp2

        return head

# Helper function to create a linked list
def create_linked_list(values):
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head

# Helper function to convert a linked list to a list
def linked_list_to_list(head):
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result

# Example
values = [1, 2, 3, 4, 5]
head = create_linked_list(values)

solution = Solution()
reordered_head = solution.reorderList(head)

print(linked_list_to_list(reordered_head))  # Output: [1, 5, 2, 4, 3]
