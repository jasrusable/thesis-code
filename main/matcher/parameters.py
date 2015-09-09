

class Parameter(object):
    def __init__(self, name):
        self.name = name


class IntegerParameter(Parameter):
    def __init__(self, name, range_):
        Parameter.__init__(self, name)
        self.range = range_
