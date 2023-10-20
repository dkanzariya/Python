class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head):
    prev = None
    current = head

    while current is not None:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    return prev  # The new head of the reversed list

# Example usage:
# Create a sample linked list: 1 -> 2 -> 3 -> 4 -> 5
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))

# Reverse the linked list
reversed_head = reverse_linked_list(head)

# Print the reversed linked list
current = reversed_head
while current is not None:
    print(current.val, end=" -> ")
    current = current.next
