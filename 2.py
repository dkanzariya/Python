def intersect(nums1, nums2):
    dict = {}
    for i in nums1:
        if i in dict:
            dict[i] += 1
        else:
            dict[i] = 1
    dict2 = {}
    for i in nums2:
        if i in dict2:
            dict2[i] += 1
        else:
            dict2[i] = 1

    res = []
    for i in nums1:
        if i in nums2 and len(res) < len(nums2) :
            res.append(i)
    return res

list1 = [1,2,2,1]
list2 = [2]
print(intersect(list1, list2))