# quantization.py

import numpy as np
import struct


def quantize_parameters(parameters):

    quantized_parameters = []

    original_size = 0
    quantized_size = 0

    for param in parameters:

        param = np.asarray(param, dtype=np.float32)

        original_size += param.nbytes

        max_abs = np.max(np.abs(param))

        if max_abs == 0:
            scale = np.float32(1.0)
        else:
            scale = np.float32(max_abs / 127.0)

        q = np.round(param / scale)
        q = np.clip(q, -127, 127).astype(np.int8)

        # ------------------------------------------------
        # Store:
        # scale + original dimensions + INT8 data
        # ------------------------------------------------

        shape = param.shape

        metadata = np.array(
            [
                scale,
                *shape
            ],
            dtype=np.float32
        )

        # Keep metadata and INT8 data as separate arrays
        # BUT mark them explicitly.
        quantized_parameters.append(
            metadata
        )

        quantized_parameters.append(
            q
        )

        quantized_size += metadata.nbytes
        quantized_size += q.nbytes

    quantized_mb = quantized_size / (1024 * 1024)

    compression_ratio = (
        original_size / quantized_size
    )

    reduction_percent = (
        1 - quantized_size / original_size
    ) * 100

    return (
        quantized_parameters,
        quantized_size,
        quantized_mb,
        compression_ratio,
        reduction_percent
    )


def dequantize_parameters(
    quantized_parameters
):

    parameters = []

    # Each original tensor = metadata + q
    for i in range(
        0,
        len(quantized_parameters),
        2
    ):

        metadata = np.asarray(
            quantized_parameters[i],
            dtype=np.float32
        )

        q = np.asarray(
            quantized_parameters[i + 1],
            dtype=np.int8
        )

        scale = float(
            metadata[0]
        )

        shape = tuple(
            int(x)
            for x in metadata[1:]
        )

        parameter = (
            q.astype(np.float32)
            * scale
        )

        parameter = parameter.reshape(
            shape
        )

        parameters.append(
            parameter
        )

    return parameters