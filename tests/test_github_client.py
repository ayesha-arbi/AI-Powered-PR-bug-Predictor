import unittest
from unittest.mock import patch

from src.github_client import GitHubAPIError, get_paginated, request_json


class TestPagination(unittest.TestCase):
    @patch("src.github_client.request_json")
    def test_stops_on_short_page(self, mock_json):
        mock_json.side_effect = [
            [{"id": 1}, {"id": 2}],
            [{"id": 3}],
        ]
        items = get_paginated(
            "https://example.test/items",
            operation="list",
            per_page=2,
            max_pages=10,
        )
        self.assertEqual([i["id"] for i in items.items], [1, 2, 3])
        self.assertFalse(items.truncated)
        self.assertEqual(mock_json.call_count, 2)

    @patch("src.github_client.request_json")
    def test_item_limit(self, mock_json):
        mock_json.return_value = [{"id": i} for i in range(5)]
        items = get_paginated(
            "https://example.test/items",
            operation="list",
            per_page=100,
            item_limit=3,
        )
        self.assertEqual(len(items.items), 3)
        self.assertTrue(items.truncated)

    @patch("src.github_client.request_json")
    def test_max_pages_marks_truncated(self, mock_json):
        mock_json.return_value = [{"id": 1}, {"id": 2}]
        page = get_paginated(
            "https://example.test/items",
            operation="list",
            per_page=2,
            max_pages=2,
        )
        self.assertTrue(page.truncated)
        self.assertEqual(len(page.items), 4)

    @patch("src.github_client.time.sleep")
    @patch("src.github_client.requests.get")
    def test_rate_limit_over_cap_raises(self, mock_get, mock_sleep):
        class Resp:
            status_code = 429
            reason = "Too Many Requests"
            headers = {"Retry-After": "999"}
            text = "rate limit"

        mock_get.return_value = Resp()
        with self.assertRaises(GitHubAPIError) as ctx:
            request_json("https://example.test/x", operation="get")
        self.assertIn("rate-limited", str(ctx.exception).lower())
        mock_sleep.assert_not_called()

    @patch("src.github_client.time.sleep")
    @patch("src.github_client.requests.get")
    def test_http_error_not_swallowed(self, mock_get, _sleep):
        class Resp:
            status_code = 404
            reason = "Not Found"
            headers = {}
            text = "missing"

        mock_get.return_value = Resp()
        with self.assertRaises(GitHubAPIError):
            request_json("https://example.test/x", operation="get")


if __name__ == "__main__":
    unittest.main()
