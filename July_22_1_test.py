import unittest

from July_22_1_V2 import count_divisible_pairs

class TestJuly(unittest.TestCase):
    def test_july1(self):
        self.assertEquals(count_divisible_pairs(1, [1, 1, 1], 1), 1)
    def test_july2(self):
        self.assertEquals(count_divisible_pairs(4, [0, 1, 2, 3], 2), 8)
    def test_july3(self):
        self.assertEquals(count_divisible_pairs(3, [1, -1, 0], 2), 5)
if __name__ == '__main__':
    unittest.main()