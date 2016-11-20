import numpy
import theano
import theano.tensor as T

def get_parameters(Ws=None, bs=None):
    Ws_s = [theano.shared(W) for W in Ws]
    bs_s = [theano.shared(b) for b in bs]

    return Ws_s, bs_s

def get_model(Ws_s, bs_s):
    print 'building expression graph'
    x_s = T.matrix('x')

    # Convert input into a 12 * 64 list
    pieces = []
    for piece in [1,2,3,4,5,6,8,9,10,11,12,13]:
        # pieces.append((x_s <= piece and x_s >= piece).astype(theano.config.floatX))
        z = T.eq(x_s, piece)
        pieces.append(z)

    binary_layer = T.concatenate(pieces, axis=1)

    last_layer= binary_layer
    n = len(Ws_s)
    for l in xrange(n-1):
        # h = T.tanh(T.dot(last_layer, Ws[l]) + bs[l])
        h = T.dot(last_layer, Ws_s[l]) + bs_s[l]
        h = h * (h > 0)
        
        last_layer = h

    # p_s = T.dot(last_layer, Ws_s[-1]) + bs_s[-1]

    return x_s, last_layer
