"""Resting and peak spike rates of the fly's three emotion populations on the cached
connectome, shared by every script that builds a ReadingSession. Measured by
scripts/calibrate_sentiment.py: re-run it and paste its output here after rebuilding the
cache or changing the input weights."""

from eternalfly.emotion_decoder import PoolCalibration

EMOTION_CALIBRATIONS = {
    "reward": PoolCalibration(resting_rate=0.0597, peak_rate=0.3276),
    "aversion": PoolCalibration(resting_rate=0.0628, peak_rate=0.2896),
    "arousal": PoolCalibration(resting_rate=0.0504, peak_rate=0.3149),
}
