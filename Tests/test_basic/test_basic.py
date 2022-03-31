from Source.main import do_it
import numpy

def train_basic():
    args = ['-f', 'test_basic/ex1.txt', '-m', '1000', '-N', 'test_basic/ex1', '-igs']
    return do_it(args)

def test_basic():
    answer = numpy.array([[-1.8376, 0., 1.8376], [ 0., 0., 0. ], [ 0.6819, 2.7304, -3.3204]])
    result = train_basic()
    diff   = result - answer
    vector = diff.reshape((9))
    assert numpy.linalg.norm(vector,numpy.inf) < 1.0e-3
