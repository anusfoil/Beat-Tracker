
bt_algo = "RNNBeatProcessor"

PROCESSORS = [
	"BeatTrackingProcessor", 
	"DBNBeatTrackingProcessor"
]
processor_idx = 0
processor = PROCESSORS[processor_idx]

ODF_ALGOS = ["rms", "hfc", "sf", "cd", "rcd", "pd"]
odf_idx = 0

CLUSTER_WIDTH = 0.025

STARTUP_PERIOD = 5
TIMEOUT = 5
TOL_OUTER = 0.4
TOL_INNER = 0.04

CORRECTION_FACTOR = 0.2
MISSING_THR = 10