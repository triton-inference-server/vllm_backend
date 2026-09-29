import json
import queue
import unittest
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class EmbedRequestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = grpcclient.InferenceServerClient(url="localhost:8001")
        cls.client.load_model("vllm_embed")

    @classmethod
    def tearDownClass(cls):
        cls.client.unload_model("vllm_embed")
        cls.client.close()

    def setUp(self):
        self.completed_requests = queue.Queue()
        self.client.start_stream(
            callback=partial(_on_response, self.completed_requests)
        )

    def tearDown(self):
        self.client.stop_stream()

    def test_string_prompt(self):
        self._check_embedding(input_value="hello world")

    def test_token_prompt(self):
        self._check_embedding(input_value=[101, 7592, 102])

    def _check_embedding(self, input_value):
        text_input = grpcclient.InferInput("text_input", [1], "BYTES")
        text_input.set_data_from_numpy(np.array([b""], dtype=np.object_))
        embedding_request = grpcclient.InferInput("embedding_request", [1], "BYTES")
        embedding_request.set_data_from_numpy(
            np.array(
                [
                    json.dumps({"input": input_value, "pooling_params": {}}).encode(
                        "utf-8"
                    )
                ],
                dtype=np.object_,
            )
        )

        self.client.async_stream_infer(
            model_name="vllm_embed",
            inputs=[text_input, embedding_request],
            outputs=[grpcclient.InferRequestedOutput("text_output")],
            request_id=self.id(),
            parameters={},
        )
        try:
            result = self.completed_requests.get(timeout=120)
        except queue.Empty:
            self.fail("Embedding request did not complete")

        self.assertNotIsInstance(result, InferenceServerException, str(result))
        embedding = json.loads(result.as_numpy("text_output")[0].decode("utf-8"))
        self.assertGreater(len(embedding), 0)
        self.assertTrue(all(isinstance(value, float) for value in embedding))


def _on_response(completed_requests, result, error):
    completed_requests.put(error if error is not None else result)


if __name__ == "__main__":
    unittest.main()
