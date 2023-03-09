from .cmnli_linearprob import CMNLILinearProbBenchmark

__all__ = ['SUPPORTED_BENCHMARKS']

SUPPORTED_BENCHMARKS = {
    'cmnli_linearprob': CMNLILinearProbBenchmark,
}
