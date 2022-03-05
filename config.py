
bt_algo = "RNNBeatProcessor"

PROCESSORS = [
	"BeatTrackingProcessor", 
	"DBNBeatTrackingProcessor"
]
processor_idx = 0
processor = PROCESSORS[processor_idx]

ODF_ALGOS = ["rms", "hfc", "sf", "cd", "rcd", "pd", "wpd"]
odf_idx = 4

CLUSTER_WIDTH = 0.025

STARTUP_PERIOD = 5
TIMEOUT = 7
TOL_PRE = 0.2
TOL_POST = 0.4
TOL_INNER = 0.04

CORRECTION_FACTOR = 1
MISSING_THR = 10