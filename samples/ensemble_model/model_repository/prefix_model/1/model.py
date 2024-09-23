import triton_python_backend_utils as pb_utils

import numpy as np

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
            input_string = input_tensor.as_numpy()[0][0].decode('utf-8')
            output_string = f"gpt2 results: {input_string}"
            output_np_array = np.array([output_string.encode('utf-8')])
            output_tensor = pb_utils.Tensor("text_output", 
                                             output_np_array)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor])
            responses.append(inference_response)
        return responses