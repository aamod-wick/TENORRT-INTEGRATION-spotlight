class TensorRTInfer:
    """
    Implements inference for a two-input TensorRT engine with dynamic batching.
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
        self.inputs_by_name = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))

            # For dynamic shapes, we don't set a fixed shape during initialization
            # We'll set the shape during inference based on actual input
            shape = self.context.get_tensor_shape(name)

            # Store the tensor info without fixed allocation
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "is_input": is_input,
            }

            if is_input:
                self.inputs.append(binding)
                self.inputs_by_name[name] = binding
            else:
                self.outputs.append(binding)

            print(
                "{} '{}' with dynamic shape and dtype {}".format(
                    "Input" if is_input else "Output",
                    name,
                    dtype,
                )
            )

    def set_input_shapes(self, input_shapes):
        """
        Set the input shapes for dynamic inference.
        :param input_shapes: Dictionary with input names as keys and shapes as values
        """
        for name, shape in input_shapes.items():
            if not self.context.set_input_shape(name, shape):
                raise ValueError(f"Failed to set shape {shape} for input {name}")

        # Update output shapes and allocate memory
        self._allocate_io_buffers()

    def _allocate_io_buffers(self):
        """Allocate I/O buffers based on current context shapes"""
        # Clear previous allocations
        for allocation in self.allocations:
            common.cuda_call(cudart.cudaFree(allocation))
        self.allocations.clear()

        # Allocate for all tensors
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self.context.get_tensor_shape(name)

            # Calculate memory size and allocate
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))

            # Update binding information
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_input:
                binding = self.inputs_by_name[name]
            else:
                # Find output binding
                binding = next((out for out in self.outputs if out["name"] == name), None)

            if binding:
                binding["shape"] = list(shape)
                binding["allocation"] = allocation
                if not is_input:
                    binding["host_allocation"] = np.zeros(shape, dtype)

            self.allocations.append(allocation)

    def infer(self, ft_batch, dt_batch):
        """
        Execute inference on FT and DT batches with dynamic shapes.
        """
        # Set input shapes based on actual data
        input_shapes = {
            "data_freq_time": ft_batch.shape,
            "data_dm_time": dt_batch.shape
        }
        self.set_input_shapes(input_shapes)

        # Copy inputs to device
        ft_input = self.inputs_by_name.get("data_freq_time")
        dt_input = self.inputs_by_name.get("data_dm_time")

        if ft_input is None or dt_input is None:
            # Try alternative names
            ft_input = self.inputs_by_name.get("ft_batch")
            dt_input = self.inputs_by_name.get("dt_batch")

        if ft_input is None or dt_input is None:
            raise ValueError("Could not find input bindings for FT and DT data")

        common.memcpy_host_to_device(ft_input["allocation"], ft_batch.ravel())
        common.memcpy_host_to_device(dt_input["allocation"], dt_batch.ravel())

        # Execute inference
        self.context.execute_v2(self.allocations)

        # Copy outputs from device to host
        outputs = {}
        for output in self.outputs:
            host_array = np.zeros(output["shape"], dtype=output["dtype"])
            common.memcpy_device_to_host(host_array, output["allocation"])
            outputs[output["name"]] = host_array

        return outputs

    def input_spec(self):
        """Get current input specifications"""
        specs = {}
        for inp in self.inputs:
            shape = self.context.get_tensor_shape(inp["name"])
            specs[inp["name"]] = (list(shape), inp["dtype"])
        return specs
