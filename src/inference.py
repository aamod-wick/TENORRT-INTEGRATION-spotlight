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
    
