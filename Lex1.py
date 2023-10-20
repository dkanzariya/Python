
def list_details(list):
    print("size: ", len(list))
    print(list)

list = [1, 2, 3, 6, 4]

list.append(1000)
list.insert(1, 999)
print(list)
list.pop(4)
list_details(list)