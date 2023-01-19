from mlperf_logging.mllog import constants
from mlperf_logging.mllog.mllog import MLLogger


def submission_info(mllogger: MLLogger, benchmark_name: str, submitter_name: str):
    """Logs required for a valid MLPerf submission."""
    mllogger.event(
        key=constants.SUBMISSION_BENCHMARK,
        value=benchmark_name,
    )
    mllogger.event(
        key=constants.SUBMISSION_ORG,
        value=submitter_name,
    )
    mllogger.event(
        key=constants.SUBMISSION_DIVISION,
        value=constants.CLOSED,
    )
    mllogger.event(
        key=constants.SUBMISSION_STATUS,
        value=constants.ONPREM,
    )
    mllogger.event(
        key=constants.SUBMISSION_PLATFORM,
        value=submitter_name,
    )
