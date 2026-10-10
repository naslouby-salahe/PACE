import logging
from collections.abc import Callable
from pathlib import Path

import structlog
from structlog.stdlib import BoundLogger, LoggerFactory, ProcessorFormatter

from pace.types import ArtifactCommitState, LogEvent, LogLevel, PaceError, TextEncoding


class _StructuredFileHandler(logging.FileHandler):
    def __init__(self, log_path: Path) -> None:
        super().__init__(log_path, encoding=TextEncoding.UTF8)
        self.log_path = log_path.resolve()


def configure_logging(log_path: Path, level: LogLevel) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    resolved_log_path = log_path.resolve()
    matching_handler = next(
        (
            handler
            for handler in root_logger.handlers
            if isinstance(handler, _StructuredFileHandler) and handler.log_path == resolved_log_path
        ),
        None,
    )
    if matching_handler is None:
        for handler in tuple(root_logger.handlers):
            if isinstance(handler, _StructuredFileHandler):
                root_logger.removeHandler(handler)
                handler.close()
        file_handler = _StructuredFileHandler(resolved_log_path)
        file_handler.setFormatter(
            ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=[
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.stdlib.add_log_level,
                ],
            )
        )
        root_logger.addHandler(file_handler)
    root_logger.setLevel(level.name)
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    structlog.get_logger().debug(
        LogEvent.LOGGING_CONFIGURED, log_path=resolved_log_path.as_posix(), level=level.name
    )


def execute_with_failure_logging[T](
    operation: Callable[[], T],
    logger: BoundLogger,
    failure_event: LogEvent,
    artifact_path: Path,
) -> T:
    try:
        return operation()
    except (PaceError, OSError, RuntimeError, ValueError) as error:
        logger.error(
            failure_event,
            failure_type=type(error).__name__,
            artifact_path=artifact_path.as_posix(),
            artifact_commit_state=ArtifactCommitState.UNKNOWN,
            safe_to_resume=True,
        )
        raise
