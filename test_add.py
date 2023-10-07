import unittest
from Arrays import array
class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(array(1, 2), 2)
