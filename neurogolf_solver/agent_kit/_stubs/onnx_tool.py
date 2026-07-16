def __getattr__(name):
    def f(*a, **k):
        return None
    return f
