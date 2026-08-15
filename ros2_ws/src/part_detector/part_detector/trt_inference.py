"""
Thin wrapper around a serialized TensorRT engine for YOLO-style detectors.

Handles engine deserialization, CUDA buffer allocation, and a synchronous
infer() call that takes a preprocessed CHW float32 image and returns the
raw model output (still needs NMS/decoding — see utils.postprocess).
"""
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTEngine:
    def __init__(self, engine_path: str):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_binding = None
        self.output_bindings = []
        self.host_buffers = {}
        self.device_buffers = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape)) if -1 not in shape else int(np.prod([d for d in shape if d != -1]))

            host_mem = cuda.pagelocked_empty(max(size, 1), dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.host_buffers[name] = host_mem
            self.device_buffers[name] = device_mem
            self.context.set_tensor_address(name, int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_binding = name
                self.input_shape = shape
            else:
                self.output_bindings.append(name)

        if self.input_binding is None:
            raise RuntimeError("Engine has no input binding — check ONNX export.")

    def infer(self, chw_image: np.ndarray) -> dict:
        """
        chw_image: float32 array, shape (3, H, W), already resized/normalized
                   to match the shape the engine was built with.
        Returns: dict {output_name: np.ndarray} of raw model outputs.
        """
        batched = np.expand_dims(chw_image, axis=0).astype(np.float32, copy=False)
        flat = np.ascontiguousarray(batched).ravel()

        host_in = self.host_buffers[self.input_binding]
        np.copyto(host_in[: flat.size], flat)
        cuda.memcpy_htod_async(self.device_buffers[self.input_binding], host_in, self.stream)

        self.context.execute_async_v3(self.stream.handle)

        outputs = {}
        for name in self.output_bindings:
            cuda.memcpy_dtoh_async(self.host_buffers[name], self.device_buffers[name], self.stream)
        self.stream.synchronize()

        for name in self.output_bindings:
            out_shape = self.context.get_tensor_shape(name)
            outputs[name] = np.array(self.host_buffers[name]).reshape(out_shape)

        return outputs

    def close(self):
        # CUDA context/memory is cleaned up on process exit via pycuda.autoinit;
        # explicit del here helps if you're instantiating multiple engines in one process.
        del self.context
        del self.engine
