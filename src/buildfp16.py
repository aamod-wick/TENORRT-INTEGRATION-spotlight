import tensorrt as trt
import torch
import tensorflow as tf
import os
import sys
class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose= True, workspace=8):
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        )

        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the TensorRT network definition.
        Adds dynamic shape support for batch size [1..32].
        """
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                print(f"ERROR: Failed to load ONNX file {onnx_path}")
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                raise RuntimeError("ONNX parsing failed")
        # Define optimization profile for dynamic batch
        profile = self.builder.create_optimization_profile()
        for i in range(self.network.num_inputs):
            input_tensor = self.network.get_input(i)
            print(f"Input: {input_tensor.name}, shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

            # Replace -1 with dynamic range
            min_shape = (1, 256, 256, 1)
            opt_shape = (8, 256, 256, 1)
            max_shape = (32, 256, 256, 1)
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)

        self.config.add_optimization_profile(profile)

    def create_engine_fp16(self, engine_path):
        if not self.builder.platform_has_fast_fp16:
            print("WARNING: FP16 not natively supported on this platform.")
        self.config.set_flag(trt.BuilderFlag.FP16)

        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            raise RuntimeError("Failed to create engine.")

        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        print(f"FP16 Engine saved at: {engine_path}")


# ====unit testing hardcoded ====
onnx_file = "/content/model_a.onnx"
engine_file = "model-a-fp16.engine"

builder = EngineBuilder(verbose=True)
builder.create_network(onnx_file)  # Now supports dynamic batch [1..32]
builder.create_engine_fp16(engine_file)
