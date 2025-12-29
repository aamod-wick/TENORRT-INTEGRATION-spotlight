


import os
import sys
import logging
import argparse

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import common

from image_batcher import ImageBatcher

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(
            np.dtype(self.image_batcher.dtype).itemsize
            * np.prod(self.image_batcher.shape)
        )
        self.batch_allocation = common.cuda_call(cudart.cudaMalloc(size))
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _ = next(self.batch_generator)
            log.info(
                "Calibrating image {} / {}".format(
                    self.image_batcher.image_index, self.image_batcher.num_images
                )
            )
            common.memcpy_host_to_device(
                self.batch_allocation, np.ascontiguousarray(batch)
            )
            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        with open(self.cache_file, "wb") as f:
            log.info("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 8 * (2**30)
        )  # 8 GB

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """

        self.network = self.builder.create_network(0)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info(
                "Input '{}' with shape {} and dtype {}".format(
                    input.name, input.shape, input.dtype
                )
            )
        for output in outputs:
            log.info(
                "Output '{}' with shape {} and dtype {}".format(
                    output.name, output.shape, output.dtype
                )
            )
        assert self.batch_size > 0

    def create_engine(
        self,
        engine_path,
        precision,
        calib_input=None,
        calib_cache=None,
        calib_num_images=25000,
        calib_batch_size=8,
        calib_preprocessor=None,
    ):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        :param calib_preprocessor: The ImageBatcher preprocessor algorithm to use.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        log.info("Reading timing cache from file: {:}".format(args.timing_cache))
        common.setup_timing_cache(self.config, args.timing_cache)

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                log.warning("INT8 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)
                self.config.int8_calibrator = EngineCalibrator(calib_cache)
                if not os.path.exists(calib_cache):
                    calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
                    calib_dtype = trt.nptype(inputs[0].dtype)
                    self.config.int8_calibrator.set_image_batcher(
                        ImageBatcher(
                            calib_input,
                            calib_shape,
                            calib_dtype,
                            max_num_images=calib_num_images,
                            exact_batches=True,
                            preprocessor=calib_preprocessor,
                        )
                    )

        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            log.error("Failed to create engine")
            sys.exit(1)

        log.info("Serializing timing cache to file: {:}".format(args.timing_cache))
        common.save_timing_cache(self.config, args.timing_cache)

        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)


def main(args):
    builder = EngineBuilder(args.verbose)
    builder.create_network(args.onnx)
    builder.create_engine(
        args.engine,
        args.precision,
        args.calib_input,
        args.calib_cache,
        args.calib_num_images,
        args.calib_batch_size,
        args.calib_preprocessor,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument(
        "-p",
        "--precision",
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable more verbose log output"
    )
    parser.add_argument(
        "--calib_input", help="The directory holding images to use for calibration"
    )
    parser.add_argument(
        "--calib_cache",
        default="./calibration.cache",
        help="The file path for INT8 calibration cache to use, default: ./calibration.cache",
    )
    parser.add_argument(
        "--calib_num_images",
        default=25000,
        type=int,
        help="The maximum number of images to use for calibration, default: 25000",
    )
    parser.add_argument(
        "--calib_batch_size",
        default=8,
        type=int,
        help="The batch size for the calibration process, default: 1",
    )
    parser.add_argument(
        "--calib_preprocessor",
        default="V2",
        choices=["V1", "V1MS", "V2"],
        help="Set the calibration image preprocessor to use, either 'V2', 'V1' or 'V1MS', default: V2",
    )
    parser.add_argument(
        "--timing_cache",
        default="./timing.cache",
        help="The file path for timing cache, default: ./timing.cache",
    )
    args = parser.parse_args()
    if not all([args.onnx, args.engine]):
        parser.print_help()
        log.error("These arguments are required: --onnx and --engine")
        sys.exit(1)
    if args.precision == "int8" and not any([args.calib_input, args.calib_cache]):
        parser.print_help()
        log.error(
            "When building in int8 precision, either --calib_input or --calib_cache are required"
        )
        sys.exit(1)
    main(args)


--------------------------------------------------------------------------------------------------------------------------------------------------------------
import os
import sys
import time
import argparse
import numpy as np
import tensorrt as trt
import ctypes
from cuda import cudart

# Define the common module with CUDA utilities
class Common:
    @staticmethod
    def cuda_call(call):
        """
        Call CUDA function and check for errors.
        """
        result = call
        if result[0].value:
            raise RuntimeError(f"CUDA error: {result[0].value}")
        return result[1] if len(result) > 1 else None

    @staticmethod
    def memcpy_host_to_device(device_ptr, host_arr):
        """
        Copy data from host to device.
        """
        cudart.cudaMemcpy(
            device_ptr,  # device destination
            host_arr.ctypes.data,  # host source
            host_arr.nbytes,  # size
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,  # kind
        )

    @staticmethod
    def memcpy_device_to_host(host_arr, device_ptr):
        """
        Copy data from device to host.
        """
        cudart.cudaMemcpy(
            host_arr.ctypes.data,  # host destination
            device_ptr,  # device source
            host_arr.nbytes,  # size
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,  # kind
        )

# Create common instance 
common = Common()

class TensorRTInfer:
    """
    Implements inference for a two-input TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []

        # Store inputs by name for easy access
        self.inputs_by_name = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self.context.get_tensor_shape(name)

            # Handle dynamic shapes for inputs
            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_tensor_profile_shape(name, 0)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_input_shape(name, profile_shape[2])
                shape = self.context.get_tensor_shape(name)

            # Calculate memory size and allocate
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))

            # For inputs: no host allocation (provided during inference)
            # For outputs: pre-allocate host buffer
            host_allocation = None if is_input else np.zeros(shape, dtype)

            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }

            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
                self.inputs_by_name[name] = binding  # Store by name
            else:
                self.outputs.append(binding)

            print(
                "{} '{}' with shape {} and dtype {}".format(
                    "Input" if is_input else "Output",
                    binding["name"],
                    binding["shape"],
                    binding["dtype"],
                )
            )

        # For multi-input, batch size might be determined by first input
        if len(self.inputs) > 0:
            self.batch_size = self.inputs[0]["shape"][0]
        else:
            self.batch_size = 0

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for all input tensors.
        :return: Dictionary with input names as keys and (shape, dtype) as values.
        """
        specs = {}
        for inp in self.inputs:
            specs[inp["name"]] = (inp["shape"], inp["dtype"])
        return specs

    def output_spec(self):
        """
        Get the specs for the output tensors.
        :return: Dictionary with output names as keys and (shape, dtype) as values.
        """
        specs = {}
        for o in self.outputs:
            specs[o["name"]] = (o["shape"], o["dtype"])
        return specs

    def infer(self, ft_batch, dt_batch):
        """
        Execute inference on FT and DT batches.
        :param ft_batch: A numpy array holding the FT image batch.
        :param dt_batch: A numpy array holding the DT image batch.
        :return: Dictionary with output names as keys and numpy arrays as values.
        """
        # Copy inputs to device - using tensor names
        ft_input = self.inputs_by_name.get("ft_batch")
        dt_input = self.inputs_by_name.get("dt_batch")

        if ft_input is None or dt_input is None:
            # Fallback to order-based if names not found
            if len(self.inputs) >= 2:
                ft_input, dt_input = self.inputs[0], self.inputs[1]
            else:
                raise ValueError("Engine doesn't have two inputs as expected")

        # Validate input shapes and types
        assert ft_batch.shape == tuple(ft_input["shape"]), \
            f"FT batch shape {ft_batch.shape} doesn't match expected {ft_input['shape']}"
        assert dt_batch.shape == tuple(dt_input["shape"]), \
            f"DT batch shape {dt_batch.shape} doesn't match expected {dt_input['shape']}"
        assert ft_batch.dtype == ft_input["dtype"], \
            f"FT batch dtype {ft_batch.dtype} doesn't match expected {ft_input['dtype']}"
        assert dt_batch.dtype == dt_input["dtype"], \
            f"DT batch dtype {dt_batch.dtype} doesn't match expected {dt_input['dtype']}"

        # Copy input data to device
        common.memcpy_host_to_device(ft_input["allocation"], ft_batch)
        common.memcpy_host_to_device(dt_input["allocation"], dt_batch)

        # Execute inference
        self.context.execute_v2(self.allocations)

        # Copy outputs from device to host
        for output in self.outputs:
            common.memcpy_device_to_host(
                output["host_allocation"], output["allocation"]
            )

        # Return outputs as dictionary
        return {output["name"]: output["host_allocation"] for output in self.outputs}

    def infer_batch(self, batch_dict):
        """
        Alternative inference method that takes a dictionary of inputs.
        :param batch_dict: Dictionary with input names as keys and numpy arrays as values.
        :return: Dictionary with output names as keys and numpy arrays as values.
        """
        # Copy all inputs to device
        for input_name, input_data in batch_dict.items():
            input_binding = self.inputs_by_name.get(input_name)
            if input_binding is None:
                raise ValueError(f"Input '{input_name}' not found in engine")

            # Validate shape and dtype
            assert input_data.shape == tuple(input_binding["shape"]), \
                f"Input {input_name} shape mismatch: got {input_data.shape}, expected {input_binding['shape']}"
            assert input_data.dtype == input_binding["dtype"], \
                f"Input {input_name} dtype mismatch: got {input_data.dtype}, expected {input_binding['dtype']}"

            common.memcpy_host_to_device(input_binding["allocation"], input_data)

        # Execute inference
        self.context.execute_v2(self.allocations)

        # Copy outputs from device to host
        for output in self.outputs:
            common.memcpy_device_to_host(
                output["host_allocation"], output["allocation"]
            )

        return {output["name"]: output["host_allocation"] for output in self.outputs}
##-------------unit tests for now-------------------------------------##
engine_path = "model-a-fp16.engine"
probablity = 0.5
data_dir = "/"
inferer = TensorRTInfer(engine_path)
import os
import sys
import time
import argparse
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


# Check input specifications
input_specs = inferer.input_spec()
print("Input specs:", input_specs)
# Output: {'ft_batch': ([8, 256, 256, 1], dtype.float32), 'dt_batch': ([8, 256, 256, 1], dtype.float32)}

# Perform inference
for i in range(1):
  ft_batch = np.random.randn(32, 256, 256, 1).astype(np.float32)
  dt_batch = np.random.randn(32, 256, 256, 1).astype(np.float32)

  outputs = inferer.infer(ft_batch, dt_batch)
  # or
  '''outputs = inferer.infer_batch({
    "ft_batch": ft_batch,
    "dt_batch": dt_batch
  })'''

  print("Outputs:", outputs.keys())
    
