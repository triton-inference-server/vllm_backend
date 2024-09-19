
import json
import numpy as np


# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args["model_config"])
        self.model_name = "gpt2" # Choose 

    # Converts all input to uppercase
    def preprocess(self, request):
        text_input = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()
        # Pre-Processing Step (Making input all Uppercase)
        processed_text = text_input[0].upper()

        preprocessed_tensor = pb_utils.Tensor(
                                              "text_input", # Name of input tensor for gpt2 
                                              np.array([processed_text])
                                            )        

        return preprocessed_tensor

    def execute(self, requests):
        response = []

        for request in requests:

            preprocessed_tensor = self.preprocess(request)                

            infer_request = pb_utils.InferenceRequest(
                model_name=self.model_name, 
                requested_output_names=["text_output"],
                inputs=[preprocessed_tensor]
            )


            infer_responses = infer_request.exec(decoupled=True)

            
            response_text = ""

            for infer_response in infer_responses:
                # If inference response has an error, raise an exception
                if infer_response.has_error():
                    raise pb_utils.TritonModelException(infer_response.error().message())

                # Check for the last empty response.
                if len(infer_response.output_tensors()) > 0:
                    response_text += pb_utils.get_output_tensor_by_name(
                        infer_response, "text_output"
                    ).as_numpy()[0].decode("utf-8") # Convert bytes to string


            # Post-Processing Step (Adding Model Name as Prefix)
            output_tensor = self.postprocess(text_output=response_text)
            response += [
                pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
            ]

        # Since the model is using the default mode in this example, we
        # will be returning a single response.
        return response

    def postprocess(self, text_output):
        # Post-Processing Step: Adding Model Name as Prefix
        output_text = f"{self.model_name}: {text_output}" 
        return pb_utils.Tensor("text_output", np.array([output_text.encode()])) # Convert string to bytes
    

    def finalize(self):
        print("Cleaning up...")