

class Parameter(object):
    pass


class IntegerParameter(Parameter):
    def __init__(self, name, from_, to, interval=None):
        self.name = name
        self.from_ = from_
        self.to = to
        self.interval = interval
