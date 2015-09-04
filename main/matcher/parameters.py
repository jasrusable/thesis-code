

class Parameter(object):
    def __init__(self, name):
        self.name = name


class IntegerParameter(Parameter):
    def __init__(self, name, from_, to, step=1):
        Parameter.__init__(self, name)
        self.from_ = from_
        self.to = to
        self.step = step
