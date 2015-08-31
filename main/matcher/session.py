

class Session(object):
    def __init__(self):
        self._test_cases = []

    def add_thing(self, test):
        self._test_cases.append(test)

    def get_things(self):
        return self._test_cases

    def get_thing(self, thing_id):
        for thing in self._test_cases:
            if thing.id == thing_id:
                return thing
