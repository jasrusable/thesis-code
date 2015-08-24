

class Session(object):
	def __init__(self):
		self._things = []

	def add_thing(self, test):
		self._things.append(test)

	def get_things(self):
		return self._things

	def get_thing(self, thing_id):
		for thing in self._things:
			if thing.id == thing_id:
				return thing


