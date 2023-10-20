def max_beauty(n, k, pages, stacks):
    # Sort the pages of the books in decreasing order
    stacks.sort()
    pages.sort(reverse=True)
    # Initialize variables to keep track of the total beauty and the current book index
    total_beauty = 0
    book_index = 0
    stack_list = [[] for _ in range(len(stacks))]
    for i in range(k):
        # Initialize variables to keep track of the top and bottom books in the current stack
        if stacks[i] == 1:
            top_book = max(pages)
            bottom_book = top_book
            pages.remove(top_book)
            # pages.remove(bottom_book)
        else:
            top_book = max(pages)
            bottom_book = min(pages)
            pages.remove(top_book)
            pages.remove(bottom_book)
        total_beauty += top_book + bottom_book
        # for j in stacks:
        #     top_k_pages = pages[:j]
        #     top_book = min(top_k_pages)
        #     bottom_book = max(top_k_pages)
        #     pages = new_list = [x for x in pages if x not in top_k_pages]
        #     total_beauty += top_book + bottom_book
        # # Distribute the books into the current stack while maintaining top and bottom
        # for _ in range(stacks[i] - 2):
        #     book_index += 1
        #     top_book = max(top_book, pages[book_index])
        #     bottom_book = pages[book_index]

        # Update the total beauty with the top and bottom books of the current stack


        # Move to the next book for the next stack
        # book_index += 1

    return total_beauty


# Input
# n, k = 4, 2
# pages = [2, 5, 2, 5]
# stacks = [2, 2]

# n, k = 4, 2
# pages = [7, 1, 1, 12]
# stacks = [3, 1]

n, k = 4, 1
pages = [115, 402, 111, 720]
stacks = [4]
# Calculate and print the maximum possible beauty
result = max_beauty(n, k, pages, stacks)
print(result)
