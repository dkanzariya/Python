def remove_nth_node_from_list(list1, n):
  """Removes the nth node from a list.

  Args:
    list1: A list of elements.
    n: The index of the node to be removed.

  Returns:
    A list with the nth node removed.
  """

  if not list1 or n >= len(list1):
    return list1

  if n == 0:
    list1 = list1[1:]
  else:
    previous_node = None
    current_node = list1
    for i in range(n - 1):
      previous_node = current_node
      current_node = current_node.next

    previous_node.next = current_node.next

  return list1


# Example usage:

list1 = [1, 2, 3, 4, 5]
n = 2

new_list = remove_nth_node_from_list(list1, n)

print(new_list)
