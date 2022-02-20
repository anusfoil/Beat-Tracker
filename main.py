
class BeatTracker(object):
	"""docstring for BeatTracker"""
	def __init__(self, arg):
		super(BeatTracker, self).__init__()
		self.arg = arg
		

def beatTracker(input_file):
	bt = BeatTracker()
	beats, downbeats = bt.process(input_file)
	return beats, downbeats
