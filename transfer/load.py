import pickle
import theano
import train
import numpy

def writeNdarrayToFile(L, filename):
    i = 0
    file = open(filename, 'w+')
    for x in numpy.nditer(L):
        file.write(str(x) + "\n")
        if x != 0:
            i += 1
    print (filename + ' ' + str (i))

def get_model_from_pickle(fn):
    f = open(fn)
    Ws, bs = pickle.load(f)

    Ws_s, bs_s = train.get_parameters(Ws=Ws, bs=bs)
    x, p = train.get_model(Ws_s, bs_s)

    predict = theano.function(
        inputs=[x],
        outputs=p)

    return predict

def run():
    func = get_model_from_pickle('model.pickle')
    X = []

    chess = [4,  2,  3,  5,  6,  3,  2,  4,
             1,  1,  1,  1,  1,  1,  1,  1,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
             8,  8,  8,  8,  0,  8,  8,  8,
             11, 9, 10, 12, 13, 10,  9, 11]

    checkers = [1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 8, 0, 8, 0, 8, 0, 8,
                8, 0, 8, 0, 8, 0, 8, 0,
                0, 8, 0, 8, 0, 8, 0, 8]

    checkers1 = [1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1,
                0, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 8, 0, 8, 0, 8, 0, 8,
                8, 0, 8, 0, 8, 0, 8, 0,
                0, 8, 0, 8, 0, 8, 0, 8]

    checkers2 = [1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1,
                0, 0, 1, 0, 1, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0,
                0, 8, 0, 8, 0, 8, 0, 8,
                8, 0, 8, 0, 8, 0, 8, 0,
                0, 8, 0, 8, 0, 8, 0, 8]

    X.append(checkers)
    X.append(checkers1)
    X.append(checkers2)

    res = func(X)

    writeNdarrayToFile(res[0], "checkers.txt")
    writeNdarrayToFile(res[1], "checkers1.txt")
    writeNdarrayToFile(res[2], "checkers2.txt")



if __name__ == '__main__':
    run()