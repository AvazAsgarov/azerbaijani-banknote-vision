"""Per-channel statistical moments for photometric EDA (mean, std, skewness, kurtosis)."""

import math
from dataclasses import dataclass
from typing import List


@dataclass
class ChannelStats:
    channel: str
    mean: float
    std: float
    skewness: float
    kurtosis: float
    pixel_count: int


def _moments(values: List[float]):
    n = len(values)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / n
    sigma = math.sqrt(var) if var > 0 else 1e-9
    skew = sum((v - mu) ** 3 for v in values) / (n * sigma ** 3)
    kurt = sum((v - mu) ** 4 for v in values) / (n * sigma ** 4) - 3.0
    return mu, sigma, skew, kurt


def compute_channel_stats(channel_name: str, pixel_values: List[float]) -> ChannelStats:
    mu, sigma, skew, kurt = _moments(pixel_values)
    return ChannelStats(channel=channel_name, mean=mu, std=sigma, skewness=skew, kurtosis=kurt, pixel_count=len(pixel_values))
