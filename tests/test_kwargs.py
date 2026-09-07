mykwargs = {'c': 10, 'd':1.5}

def test1(a, b, c, d):
    print((a,b,c,d))

def test2(a, b, **kwargs):
    if kwargs is None:
        kwargs = {}
    torun = mykwargs.copy()
    torun.update(kwargs)
    test1(a, b, **torun)

test2(1, 2, c=20, e=13)