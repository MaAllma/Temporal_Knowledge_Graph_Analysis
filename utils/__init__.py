import logging
import sys
TRACE = 1

# adapted from https://stackoverflow.com/questions/2302315/how-can-info-and-debug-logging-message-be-sent-to-stdout-and-higher-level-messag
# and https://docs.python.org/3/library/logging.html#filter-objects
class LessThanFilter:
    def __init__(self, max_level: int):
        super(LessThanFilter, self).__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> int:
        return 1 if record.levelno < self.max_level else 0


ROOT = logging.getLogger()
ROOT.setLevel(logging.NOTSET)
for handler in ROOT.handlers:
    ROOT.removeHandler(handler)

log_formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s: %(message)s")

handler_sysout = logging.StreamHandler(sys.stdout)
handler_sysout.setLevel(logging.DEBUG)
handler_sysout.setFormatter(log_formatter)
handler_sysout.addFilter(LessThanFilter(logging.WARNING))

handler_stderr = logging.StreamHandler(sys.stderr)
handler_stderr.setLevel(logging.WARNING)
handler_stderr.setFormatter(log_formatter)


def get_logger(name: str) -> logging.Logger:
    """
    Instantiate the logger for the given name. It is expected that most modules in the solution factory, will have
    exactly one logger (instantiated at the top of the file) where the name is equal to the module name, as per the
    convention established in the`official logging tutorial <https://docs.python.org/3/howto/logging.html>`. Only very
    implementation heavy classes should consider creating additional loggers using the full-qualified names (accessed
    via [<clazz-object>].__qualname__). The logger will be configured to print log-levels below warning to stdout
    and log-levels of warning and above to stderr.  All loggers will be using the same standardized format similar to
    the one recommended by the :ref:`python logging tutorial <https://docs.python.org/3/howto/logging.html#formatters>`:

    ``[YYYY-MM-DD HH-MM-SS,ms] <name> - <log-level>: <logged message>``

    .. note::
        As of now logs are not written to a log-file, as this requires decisions on where these are to be stored
        alongside utils for handling custom storage locations defined in the meta_inf.json. If you want your logs to be
        captured otherwise (for example in a file), as of now you have to replace sys.stdout and
        sys.stderr with a file-like object **before this module is imported for the first time** (e.g. in your _pipeline
        definition  **before importing any solution factory module**).
    .. note::
        For further information about how to best use the logging facilities integrated in python please refer to the
        :ref:`official logging tutorial <https://docs.python.org/3/howto/logging.html>` and the documentation of the
        :ref:`logging module <https://docs.python.org/3/library/logging.html>`
    :param name: The name of the logger to return. This should almost always be the name of the module that you're
                 calling this function from.
    :returns: A configured python logger that can be used log information using the standardized solution factory
              format.
    """
    logger = logging.getLogger(name)
    logger.addHandler(handler_sysout)
    logger.addHandler(handler_stderr)
    return logger