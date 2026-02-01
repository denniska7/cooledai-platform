"""
CooledAI Synthetic Sensor - ML-based Missing Data Estimation

When a data source provides incomplete data (e.g., missing fan RPM),
the SyntheticSensor uses a machine learning model to estimate the
missing value based on available inputs (power, temp).

Usage:
    from synthetic import SyntheticSensor
    sensor = SyntheticSensor()
    estimated_rpm = sensor.estimate_cooling(power=150, temp=65)
"""

from core.synthetic.synthetic_sensor import SyntheticSensor

__all__ = ["SyntheticSensor"]
