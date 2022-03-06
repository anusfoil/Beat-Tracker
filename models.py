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

	def __init__(self, odf_idx):

		self.odf_idx = odf_idx # "wpd"

		return 

	def get_onsets(self, audio_path):
		onsets, odfNum, peakIndex, odf, t = onsetDetection(audio_path)
		self.t = t
		self.peakTimes = onsets
		self.odfNum = odfNum
		self.peakIndex = peakIndex
		self.odf = odf
		onsets = onsets[odfNum == self.odf_idx]
		# hook()
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


	def get_beats(self, audio_path, plot=True):
		onsets = self.get_onsets(audio_path)

		tempo_clusters = self.get_tempo(onsets)

		"""initialize agents"""
		agents = []
		for cluster in tempo_clusters:
			for onset in onsets[onsets < config.STARTUP_PERIOD]:
				if cluster.interval != 0:
					agents.append(Agent(cluster.interval, onset, cluster.score))


		pop_agent = Agent(0, 0, 0)
		for idx, onset in enumerate(onsets):
			print(f"num of agents: {len(agents)}")
			for agent in agents:
				if (abs(onset - agent.history[-1]) > config.TIMEOUT 
					or (agent.beat_interval <= 0.1)):
					pop_agent = agent
					agents.remove(agent)
					continue
				else:
					while (agent.prediction + config.TOL_OUTER) < onset:
						agent.prediction += agent.beat_interval

					error = agent.prediction - onset
					"""if the error is within the inner tolerance threshold"""
					if (abs(error) < config.TOL_INNER):
						agent.forward(onset, error)

						"""if within the outter tolerance threshold"""
					elif (abs(error) < config.TOL_OUTER):
						# new_agents.append(copy.deepcopy(agent))
						agent.forward(onset, error)
						
					else:
						agent.missed_forward()
			


		agents.sort(key=lambda x: x.score)
		if len(agents) == 0:
			best_beats = np.sort(np.array(pop_agent.history))
		else:
			best_beats = np.sort(np.array(agents[-1].history))

		if plot:

			plot_idx = self.odf_idx

			plt.figure(figsize=(20, 8))
			from cycler import cycler
			c = 'bgrcmyk'
			# plt.rcParams['axes.prop_cycle'] = cycler(color=c)
			plt.plot(self.t, self.odf[:, plot_idx], color='b')
			legend = ['rms', 'hfc', 'sf ', 'cd ', 'rcd', 'pd ', 'wpd']
			for i in range(len(self.peakIndex)):
				if self.odfNum[i] == plot_idx:
					plt.plot(self.peakTimes[i], self.odf[self.peakIndex[i], plot_idx],
				 	'og', label=legend[self.odfNum[i]])
			
			for beat in best_beats:
				plt.stem(beat, 6, 'dr')

			plt.title(f'Onsets and beats found for file: {audio_path}')
			plt.xlabel('Time')
			plt.ylabel('ODF: Rectified Complex Domain')
			plt.show()
			hook()

		return best_beats








