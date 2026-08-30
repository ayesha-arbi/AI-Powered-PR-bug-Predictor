import unittest

from src.labeling import (
    files_for_labeling,
    is_bugfix_message,
    is_doc_path,
    is_test_path,
)


class TestBugfixMessage(unittest.TestCase):
    def test_positive_subjects(self):
        self.assertTrue(is_bugfix_message("Fix crash on startup"))
        self.assertTrue(is_bugfix_message("fixed memory leak"))
        self.assertTrue(is_bugfix_message("hotfix: login"))
        self.assertTrue(is_bugfix_message("Patch the parser"))
        self.assertTrue(is_bugfix_message("Revert 'add cache'"))
        self.assertTrue(is_bugfix_message("bug in serializer"))

    def test_false_friends(self):
        self.assertFalse(is_bugfix_message("add prefix helper"))
        self.assertFalse(is_bugfix_message("add fixture for db"))
        self.assertFalse(is_bugfix_message("dispatch event"))
        self.assertFalse(is_bugfix_message("debug logging"))
        self.assertFalse(is_bugfix_message("update packaging"))

    def test_subject_only_not_body(self):
        msg = "Improve docs\n\nThis also mentions a bug in passing."
        self.assertFalse(is_bugfix_message(msg))

    def test_empty(self):
        self.assertFalse(is_bugfix_message(""))
        self.assertFalse(is_bugfix_message(None))


class TestPaths(unittest.TestCase):
    def test_doc_and_test(self):
        self.assertTrue(is_doc_path("README.md"))
        self.assertTrue(is_doc_path("docs/guide.rst"))
        self.assertTrue(is_test_path("tests/test_app.py"))
        self.assertTrue(is_test_path("src/test_foo.py"))
        self.assertFalse(is_test_path("src/latest.py"))

    def test_label_files_prefer_non_docs(self):
        files = ["README.md", "src/app.py", "docs/x.md"]
        self.assertEqual(files_for_labeling(files, 10), ["src/app.py"])


if __name__ == "__main__":
    unittest.main()
