import sys, os 
import glob
import csv
import torch
import mir_eval
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import config as config

	
AUDIO_PATH = "BRD/BallroomData"
LABEL_PATH = "BRD/BallroomAnnotations"

def generate_meta():
	"""Generate a CSV file [title, audio_path, annoation_path] for the BRD dataset"""

	with open("BRD/BallroomDataset.csv", "w") as f:
		cw = csv.writer(f)
		cw.writerow(["title", "style", "audio_path", "annotation_path"])
		for audio_path in glob.glob(f"{AUDIO_PATH}/**/*.wav"):
			title = audio_path.split("/")[-1][:-4]
			style = audio_path.split("/")[-2]
			annotation_path = f"{LABEL_PATH}/{title}.beats"
			# print(annotation_path)
			if os.path.exists(annotation_path):
				cw.writerow([title, style, audio_path, annotation_path])
	pass



class BallroomDataset(object):
	"""docstring for BallroomDataset"""
	def __init__(self):
		"""This class takes the meta of an audio segment as input, and return 
		the waveform and targets of the audio segment. This class is used by 
		DataLoader. 
		"""
		pass

	def __getitem__(self, meta):
		"""
		Meta: dict, e.g {
			"title": "Media-105901"
			"style": "Waltz"
			"audio_path": "BRD/BallroomData/Waltz/Media-105901.wav"
			"annoation_path": "BRD/BallroomAnnotations/Media-105901.beats"
		}

		Returns: dict, {
			"audio": (np.array) []
			"beats": (np.array) [0.1944671200, 0.8499773240...]
			"downbeats": (np.array)[0.8499773240, ...]
		}
		"""
		audio, _ = librosa.load(meta["audio_path"])
		with open(meta["annotation_path"]) as f:
			lines = f.readlines()
			raw_beats = [line.split() for line in lines]

		beats = np.array([float(timestamp) for timestamp, beat in raw_beats])
		downbeats = np.array([float(timestamp) for timestamp, beat in raw_beats if beat == 1])

		return {"audio": audio, "beats": beats, "downbeats": downbeats}




"""Helper class for the beatroot algorithm"""

class Cluster(object):
	def __init__(self, interval):
		self.interval = interval
		self.iois = []
		self.score = 0

	def add_ioi(self, ioi):
		"""add ioi to this cluster, update the mean interval"""
		self.iois.append(ioi)
		self.interval = sum(self.iois) / len(self.iois)

	def merge(self, cluster):
		"""merge another cluster"""
		self.iois = np.append(self.iois, cluster.iois)
		self.interval = sum(self.iois) / len(self.iois)


def scoring_func(n):
	"""n: integer ratio of cluster intervals"""
	if 1 <= n and n <= 4:
		return 6 - n
	elif 5 <= n and n <= 8:
		return 1
	else:
		return 0


class Agent(object):
	"""docstring for Agent"""
	def __init__(self, beat_interval, event, cluster_score):
		"""
		event: an onset timestamp in seconds
		"""
		self.beat_interval = round(beat_interval, 2) 
		self.prediction = event + beat_interval
		self.history = [event]
		self.score = cluster_score
		self.num_missed = 0

	def duplicate_with(self, agent_2):
		"""if tempo and phase agreed, consider being the same agent"""
		if (self.beat_interval == agent_2.beat_interval
			and self.prediction == agent_2.prediction):
			return True
		return False
				
	def forward(self, onset, error):
		"""when agent hits an onset, update beat_interval, history, score"""
		self.beat_interval += error * config.CORRECTION_FACTOR 
		self.history.append(round(onset, 2))
		self.score += 1

		self.prediction = onset + self.beat_interval
		pass

	def missed_forward(self):
		self.num_missed += 1
		self.history.append(self.prediction)
		self.prediction += self.beat_interval
		pass


"""ODF helper functions, adapted from lab 3"""

def onsetFunctions(inFile, outFile = None, start = 0, length = None,
					hopTime = 0.010, fftTime = 0.040):
	"""Calculate onset detection functions with input parameters:
		inFile (string): name of input audio file to analyse
		outFile (string): name of output file for saving results
		start: start time (in seconds) for processing audio
		duration: duration (in seconds) of audio to process
		hopTime: hop size (in seconds) (time between successive frames)
		fftTime: length of frame to process (in seconds) - the given value is
				rounded up to the next power of 2 (in samples)

		and return value (odf): (7, N)
			a matrix of onset detection function results, with one row for each"""

	snd, rate = librosa.load(os.path.join(inFile),
							offset=start, duration=length, sr=None)
	hop = round(rate * hopTime)
	# round up to next power of 2
	len1 = int(2 ** np.ceil(np.log2(rate * fftTime)))
	len2 = int(len1/2)
	# centre first frame at t=0
	snd = np.concatenate([np.zeros(len2), snd, np.zeros(len2)])
	frameCount = int(np.floor((len(snd) - len1) / hop + 1))
	prevM = np.zeros(len1)
	prevA = np.zeros(len1)
	prevprevA = np.zeros(len1)
	odf = np.zeros((frameCount, len(config.ODF_ALGOS)))
	for i in range(frameCount):
		start = i * hop
		currentFrame = np.fft.fft(np.multiply(snd[start: start+len1],
											np.hamming(len1)))
		mag = np.abs(currentFrame)
		rms = np.sqrt(np.mean(np.power(mag, 2)))
		hfc = np.mean(np.multiply(np.power(mag, 2),
								list(range(len2)) + list(range(len2,0,-1))))
		sf = np.mean(np.multiply(np.greater(mag, prevM), np.subtract(mag, prevM)))
		phase = np.angle(currentFrame)
		tPhase = np.subtract(np.multiply(prevA, 2), prevprevA)
		cdVector = np.sqrt(np.subtract(np.add(np.power(prevM,2), np.power(mag,2)),
							np.multiply(np.multiply(np.multiply(prevM, mag), 2),
							np.cos(np.subtract(phase, tPhase)))))
		rcd = np.mean(np.multiply(np.greater_equal(mag, prevM), cdVector))
		pdVector = np.abs(np.divide(np.subtract(np.mod(np.add(
				np.subtract(phase, tPhase), np.pi), 2 * np.pi), np.pi), np.pi))
		if rms != 0:
		 wpd = np.divide(np.mean(np.multiply(pdVector, mag)), rms * 2)
		else:
		 wpd = 0
		odf[i,:] = [rms, hfc, sf, np.mean(cdVector), rcd, np.mean(pdVector), wpd]
		prevprevA = prevA
		prevM = mag
		prevA = phase
	return odf



def onsetDetection(audioFile, wlen=0.040, hop=0.010, wd=9, thr=1.25, show=0):
	"""Peak-picking wrapper for onsetFunctions()"""
	odf = onsetFunctions(audioFile, hopTime=hop, fftTime=wlen)

	# normalise with median filtered sig
	odf = np.divide(odf+1, medfilt(odf+1, wd))
	# Why do we get NaN's after this np.divide?? I can't work it out.
	odf[np.isnan(odf)] = 0
	# standardise to zero mean, unit stdev
	odf = np.divide(np.subtract(odf, np.mean(odf)), np.std(odf))
	odf[np.isnan(odf)] = 0
	t = np.multiply(range(len(odf)), hop)
	d = np.diff(odf, axis=0)

	isPeak = np.multiply(np.multiply(np.greater(odf, thr),
	            np.greater(np.concatenate([np.zeros((1,len(config.ODF_ALGOS))), d]), 0)),
	            np.less(np.concatenate([d, np.zeros((1,len(config.ODF_ALGOS)))]), 0))
	peaks = np.nonzero(isPeak)
	peakIndex = peaks[0]
	odfNum = peaks[1]
	peakTimes=t[peakIndex]
	if show:
		# plt.figure(figsize=(10, 2))
		# from cycler import cycler
		# c = 'bgrcmyk'
		# plt.rcParams['axes.prop_cycle'] = cycler(color=c)
		# plt.plot(t, odf)
		# legend = ['rms', 'hfc', 'sf ', 'cd ', 'rcd', 'pd ', 'wpd']
		# for i in range(len(peaks)):
		# 	plt.plot(peakTimes[i], odf[peakIndex[i], odfNum[i]], 
		# 		'o' + c[odfNum[i]], label=legend[odfNum[i]])
		# plt.title('Onsets found for file: ' + audioFile)
		# plt.xlabel('Time')
		# plt.ylabel('ODFs')
		# plt.show()
		# hook()
  #   if plot:
		plot_idx = 0

		plt.figure(figsize=(20, 10))
		from cycler import cycler
		c = 'bgrcmyk'
		plt.rcParams['axes.prop_cycle'] = cycler(color=c)
		plt.plot(t, odf[:, plot_idx])
		legend = ['rms', 'hfc', 'sf ', 'cd ', 'rcd', 'pd ', 'wpd']
		for i in range(len(peaks[0])):
			if odfNum[i] == plot_idx:
				plt.plot(peakTimes[i], odf[peaks[0][i], odfNum[i]],
			 	'o' + c[odfNum[i]], label=legend[odfNum[i]])
		plt.title('Onsets found for file: ')
		plt.xlabel('Time')
		plt.ylabel('ODF')
		plt.show()
		hook()
	return peakTimes, odfNum, peakIndex, odf, t



def medfilt(x, k):
	"""

	Apply a length-k median filter to a 1D array x.
	Boundaries are extended by repeating endpoints.
	"""
	# assert k % 2 == 1, "Median filter length must be odd."
	if x.ndim > 1: # Input must be one-dimensional."
		y = []  #np.empty((0,100), float)
		xt = np.transpose(x)
		for i in xt:
			y.append(medfilt(i, k))
		return np.transpose(y)
	k2 = (k - 1) // 2
	y = np.zeros((len(x), k), dtype=x.dtype)
	y[:,k2] = x
	for i in range(k2):
		j = k2 - i
		y[j:,i] = x[:-j]
		y[:j,i] = x[0]
		y[:-j,-(i+1)] = x[j:]
		y[-j:,-(i+1)] = x[-1]
	return np.median(y, axis=1)



def plot_results():
	rnn_results = pd.read_csv("results/BeatTrackingProcessor.csv")
	dbn_results = pd.read_csv("results/DBNBeatTrackingProcessor.csv")

	rms_result = pd.read_csv("results/beatroot_rms.csv")
	# hfc_result = pd.read_csv("results/beatroot_hfc.csv")
	sf_result = pd.read_csv("results/beatroot_sf.csv")
	cd_result = pd.read_csv("results/beatroot_cd.csv")
	rcd_result = pd.read_csv("results/beatroot_rcd.csv")
	pd_result = pd.read_csv("results/beatroot_pd.csv")
	
	results = [
		# rnn_results, dbn_results,
				rms_result,
				# hfc_result,
				sf_result,
				cd_result,
				rcd_result,
				pd_result]

	fig, ax = plt.subplots()
	w = 0.05

	metrics = ['F-measure', 'P-score', 'Information gain']
	algos = [
		# 'rnn', 'dbn', 
		'rms', 'sf', 'cd', 'rcd', 'pd']
	for i, result in enumerate(results):
		label = algos[i]
		plt.boxplot(result[metrics].values.T.tolist(), positions=np.array(range(len(metrics)))+w*i,
			sym='', widths=w, boxprops=dict(facecolor=f"C{i}"), 
			patch_artist=True)

		ticks = metrics
		plt.plot([], color=f"C{i}", label=label)

	plt.legend()
	plt.xticks(range(len(ticks)), ticks, rotation=20)

	plt.grid()
	plt.title('results of of beatroot by different ODF')
	plt.ylabel('score')
	plt.show()
	pass

def plot_results_by_styles():
	sf_result = pd.read_csv("results/beatroot_sf.csv")

	fig, ax = plt.subplots()
	w = 0.05

	metrics = ['F-measure', 'P-score', 'Information gain']
	styles = sf_result['style'].unique()

	for i, style in enumerate(styles):
		result = sf_result[sf_result['style'] == style]
		label = style

		plt.boxplot(result[metrics].values.T.tolist(), positions=np.array(range(len(metrics)))+w*i,
			sym='', widths=w, boxprops=dict(facecolor=f"C{i}"), 
			patch_artist=True)

		ticks = metrics
		plt.plot([], color=f"C{i}", label=label)

	plt.legend()
	plt.xticks(range(len(ticks)), ticks, rotation=20)

	plt.grid()

	plt.title('results of Beatroot(SF) by genre')
	plt.ylabel('score')
	plt.show()
	pass


if __name__ == '__main__':

	pass



