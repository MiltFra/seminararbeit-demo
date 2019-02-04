from mxnet import nd

def indx2onehot(indx, size):
    v = nd.zeros(size)
    v[indx] = 1
    return v


def dist2prob(a, axis=0):
    s = nd.sum(a, axis=axis)
    return a / s


def sym2indx(sym):
    if sym == '\n':
        return 0
    return ord(sym) - 31


def indx2sym(indx):
    if indx == 0:
        return '\n'
    return chr(indx + 31)
def sym2vec(sym):
    indx = sym2indx(sym)
    v = nd.zeros(1)
    v[0] = indx
    return v.one_hot(depth=96)
def indx2vec(indx):
    v = nd.zeros(1)
    v[0] = indx
    return v.one_hot(depth=96)