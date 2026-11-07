import json
import unittest
from unittest.mock import patch

import llm_bench_lite


class ChunkedResponse:
    def __init__(self, body):
        self.body = body
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        chunk = self.body[self.offset:self.offset + min(size, 7)]
        self.offset += len(chunk)
        return chunk


class StreamingUtf8Tests(unittest.TestCase):
    def test_streamed_unicode_survives_chunk_boundaries(self):
        content = "café 🧢"
        event = {"choices": [{"delta": {"content": content}}]}
        body = f"data: {json.dumps(event, ensure_ascii=False)}\n\ndata: [DONE]\n".encode()
        response = ChunkedResponse(body)

        with patch.object(llm_bench_lite.urllib.request, "urlopen", return_value=response):
            result = llm_bench_lite.chat_completion(
                "http://localhost/v1", "", "test-model", "prompt", 10, 0
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["response_preview"], content)


if __name__ == "__main__":
    unittest.main()
