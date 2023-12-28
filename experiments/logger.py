# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import logging


FORMATTER = logging.Formatter("[%(asctime)-19.19s %(levelname)-1.1s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

MEMORY_HANDLER = logging.handlers.MemoryHandler(capacity=100) #, flushLevel=logging.INFO)
MEMORY_HANDLER.setFormatter(FORMATTER)

LOGGER = logging.getLogger("lorentz-gatr")
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(MEMORY_HANDLER)
LOGGING_INITIALIZED = False
