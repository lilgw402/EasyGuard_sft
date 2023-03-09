from .douyin_recall import DouyinSearchRecallBenchmark
from .douyin_recall_v2 import DouyinSearchRecallBenchmarkV2
from .image_recall import ImageSearchRecallBenchmark
from .image_recall_v2 import ImageSearchRecallBenchmarkV2

__all__ = ['SUPPORTED_BENCHMARKS']

SUPPORTED_BENCHMARKS = {
    'douyin_recall': DouyinSearchRecallBenchmark,
    'douyin_recall_v2': DouyinSearchRecallBenchmarkV2,
    'tusou_recall': ImageSearchRecallBenchmark,
    'tusou_recall_v2': ImageSearchRecallBenchmarkV2
}
