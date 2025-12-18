import logging


def setup_logging(level: int = logging.INFO):
    """Configure logging for CLI usage."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )