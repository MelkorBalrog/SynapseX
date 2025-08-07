"""Shared constants for the SynapseX project."""

# Base address of the image buffer in bytes.
# Historically this was specified in words (0x5000). Store it once here in
# bytes to make conversions explicit.
IMAGE_BUFFER_BASE_ADDR_BYTES = 0x5000 * 4

__all__ = ["IMAGE_BUFFER_BASE_ADDR_BYTES"]
