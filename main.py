import sys, os 
import glob
import csv
import torch
import mir_eval
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import * 
from models import *
import config
import hook


def evaluate(processor="BeatTrackingProcessor", idx=5):
	"""
	evaluate across the entire set

	processor: str, "DBNBeatTrackingProcessor"

	Return: eval_results (pd.DataFrame)
	[title, style, F-measure', Cemgil, Cemgil Best Metric Level, Goto, P-score, 
	Correct Metric Level Continuous, Correct Metric Level Total, 
	Any Metric Level Continuous, Any Metric Level Total, Information gain]
	"""

	brd = BallroomDataset()
	# bt = BeatTracker(processor)
	bt = ODFBeatTracker(idx)

	eval_results = pd.DataFrame()

	brd_meta = pd.read_csv("BRD/BallroomDataset.csv")
	for meta_dict in tqdm(brd_meta.to_dict(orient="records")):
		# meta_dict =  {
		# 	"title": "Media-105209",
		# 	"style": "Rumba-Misc",
		# 	"audio_path": "BRD/BallroomData/Rumba-Misc/Media-105209.wav",
		# 	"annotation_path": "BRD/BallroomAnnotations/Media-105209.beats"
		# }

		data_dict = brd[meta_dict]
		est_beats = bt.get_beats(meta_dict["audio_path"])
		ref_beats = data_dict["beats"]
		
		scores = mir_eval.beat.evaluate(ref_beats, est_beats)
		scores['title'] = meta_dict['title']
		scores['style'] = meta_dict['style']

		eval_results = eval_results.append(pd.DataFrame(scores, index=[0]), ignore_index=True)

	cols = eval_results.columns.tolist()
	cols = cols[-2:] + cols[:-2]
	eval_results = eval_results[cols]
	eval_results.to_csv(f"results/beatroot_{config.ODF_ALGOS[idx]}.csv", index=False)

	return eval_results


def run_experiments():
	# for processor in config.PROCESSORS:
	# 	evaluate(processor=processor)

	for idx, odf in enumerate(config.ODF_ALGOS):
		evaluate(idx=4)

	pass

		
if __name__ == '__main__':
	# run_experiments()
	plot_results_by_styles()

	pass
