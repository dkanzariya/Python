nums = [0,1,0,3,12]
# nums = [0,0,1]
# nums = [0,1]

new_list = [element for element in nums if element != 0]
print(new_list)

for i in range(len(new_list)):
    nums[i] = new_list[i]
# print(nums)

for j in range(len(nums) - len(new_list)):
    nums.pop(len(nums) - j-1)
    nums.append(0)
print(nums)
count = 0
for i in nums:
    if i == 0:
        nums.pop(i)
        count += 1
for i in range(count):
    nums.append(0)
'''
for i in range(1, len(nums)):
            if nums[i-1] == 0 and :
                nums.pop(nums[0])
                i -= 1
                nums.append(0)
                i += 1
'''
print(nums)