import os, sys
import copy
import numpy as np
from madmom.features.beats import *
import config
import hook
from tqdm import tqdm
from utils import * 


class BeatTracker(object):
	"""
	processor: str, "DBNBeatTrackingProcessor"
	"""
	def __init__(self, processor):
		self.processor = processor
		return

	def get_beats(self, audio_path):

		if self.processor == "SuperFluxProcessor":
			return self.get_beats_superflux(audio_path)
		else:
			return self.get_beats_rnn(audio_path)


	def get_beats_rnn(self, audio_path):
		"""
		audio_path: str, "BRD/BallroomData/Waltz/Media-105901.wav"

		Return:
			beats: (np.array) []
		"""

		"""processor to get a beat activation function"""
		act = RNNBeatProcessor()(audio_path)

		proc = eval(self.processor)(fps=100)
		return proc(act)
		# return np.array([1, 2])



class ODFBeatTracker():

	def __init__(self):

		self.odf = config.ODF_ALGOS[config.odf_idx] # "wpd"

		return 

	def get_onsets(self, audio_path):
		onsets, odfNum = onsetDetection(audio_path)
		onsets = onsets[odfNum == config.odf_idx]
		return onsets

	def get_tempo(self, onsets):
		"""
		onsets: np.array: a list of onset timestamps

		infer tempi by performing clustering of Inter-Onset-Intervals

		Returns: a list of inferred tempo (in clusters), from the most salient
		"""

		clusters = []
		for i in range(len(onsets)-2):
			for j in range(i, len(onsets)-1):
				ioi = onsets[j] - onsets[i]
				min_diff = float('inf')

				for idx, cluster in enumerate(clusters):
					diff = abs(cluster.interval - ioi)
					if (diff < config.CLUSTER_WIDTH and diff < min_diff):
						min_diff = diff
						min_cluster_idx = idx
				if min_diff != float('inf'):
					clusters[min_cluster_idx].add_ioi(ioi)
				else:
					clusters.append(Cluster(ioi))

		for cluster_i in clusters:
			for cluster_j in clusters:
				if clusters.index(cluster_j) <= clusters.index(cluster_i):
					continue
				if abs(cluster_i.interval - cluster_j.interval) < config.CLUSTER_WIDTH:
					cluster_i.merge(cluster_j)
					clusters.remove(cluster_j)

		for cluster_i in clusters:
			for cluster_j in clusters:
				n = int(cluster_i.interval / cluster_j.interval) if cluster_j.interval else 0
				if abs(cluster_i.interval - n * cluster_j.interval) < config.CLUSTER_WIDTH:
					cluster_i.score += scoring_func(n) * len(cluster_j.iois)

		clusters.sort(key=lambda x: x.score, reverse=True)
		return clusters


	def get_beats(self, audio_path):
		"""
		
		"""
		onsets = self.get_onsets(audio_path)

		# onsets = np.array([1.860,
		# 		2.627,
		# 		3.333,
		# 		4.053,
		# 		4.753,
		# 		5.480,
		# 		6.203,
		# 		6.910,
		# 		7.630,
		# 		8.340,
		# 		9.070,
		# 		9.793,
		# 		10.517,
		# 		11.230,
		# 		11.943,
		# 		12.663,
		# 		13.383,
		# 		14.110,
		# 		14.833,
		# 		15.550,
		# 		16.267,
		# 		16.980,
		# 		17.697,
		# 		18.403,
		# 		19.117,
		# 		19.837,
		# 		20.557,
		# 		21.270,
		# 		21.987,
		# 		22.710,
		# 		23.437,
		# 		24.153,
		# 		24.867,
		# 		25.590,
		# 		26.340,
		# 		27.060,
		# 		27.763,
		# 		28.447,
		# 		29.157,
		# 		29.870])

		tempo_clusters = self.get_tempo(onsets)

		tempo_clusters = tempo_clusters[:5] # take 5 highest-rank tempi

		"""initialize agents"""
		agents = []
		for cluster in tempo_clusters:
			for onset in onsets[onsets < config.STARTUP_PERIOD]:
				agents.append(Agent(cluster.interval, onset))

		for onset in (onsets):
			new_agents = []
			for agent in agents:
				if onset - agent.history[-1] > config.TIMEOUT or agent.num_missed >= config.MISSING_THR:
					agents.remove(agent)
				else:
					while agent.prediction + config.TOL_POST * agent.beat_interval < onset:
						agent.prediction += agent.beat_interval
					if agent.prediction + config.TOL_PRE * agent.beat_interval <= onset and agent.prediction + config.TOL_POST * agent.beat_interval:
						if abs(agent.prediction - onset) > config.TOL_INNER:
							new_agents.append(copy.deepcopy(agent))
						diff = onset - agent.prediction
						agent.beat_interval += diff / config.CORRECTION_FACTOR 
						agent.predction = onset + agent.beat_interval
						agent.history.append(onset)
						agent.score += (1 - diff/2)
					else:
						agent.num_missed += 1
			
			agents.extend(new_agents)

		agents.sort(key=lambda x: x.score)
		if len(agents) == 0:
			return np.array([])
		best_beats = np.array(agents[-1].history)

		return best_beats








