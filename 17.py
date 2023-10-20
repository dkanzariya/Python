def hammingWeight(n):
        count = 0

        while n:
            count += n & 1
            n >>= 1
        return count
        '''
        n = bin(n)

        count = 0

        for bit in n[2:]:
            if bit == '1':
                count += 1
        return count
        '''

        '''
        n = str(n)
        count = 0
        for i in range(31):
            if n[i] == '1':
                count += 1
        return count
        '''

print(hammingWeight(13))

