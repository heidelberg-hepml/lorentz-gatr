# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import logging

MEMORY_HANDLER = logging.handlers.MemoryHandler(capacity=100) #, flushLevel=logging.INFO)

LOGGER = logging.getLogger("gatr")
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(MEMORY_HANDLER)
logging_initialized = False
