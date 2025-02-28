'''
Created on Mar 28, 2024

@author: keith
'''
import unittest
from quant import Length


class Test(unittest.TestCase):


    def setUp(self):
        lm = Length(100)


    def tearDown(self):
        pass


    def defaultUnit(self):
        print(lm)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()