from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ocr_pipeline.results import Result, ResultWriter


class ResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def read_rows(self, path: Path) -> list[list[str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.reader(handle))

    def test_header_and_result_row(self) -> None:
        path = self.root / "out.csv"
        with ResultWriter(path) as writer:
            writer.write(Result("img0001.png", "hello", 87.421))

        rows = self.read_rows(path)
        self.assertEqual(rows[0], ["id", "filename", "text", "processing_ms"])
        self.assertEqual(rows[1], ["1", "img0001.png", "hello", "87.42"])

    def test_ids_are_sequential(self) -> None:
        path = self.root / "out.csv"
        with ResultWriter(path) as writer:
            for index in range(1, 4):
                self.assertEqual(
                    writer.write(Result(f"img{index}.png", "text", 1.0)), index
                )

        rows = self.read_rows(path)
        self.assertEqual([row[0] for row in rows[1:]], ["1", "2", "3"])

    def test_text_with_commas_and_quotes_round_trips(self) -> None:
        path = self.root / "out.csv"
        with ResultWriter(path) as writer:
            writer.write(Result("img0001.png", 'hello, "world"', 5.0))

        rows = self.read_rows(path)
        self.assertEqual(rows[1][2], 'hello, "world"')

    def test_failed_result_writes_empty_text(self) -> None:
        path = self.root / "out.csv"
        with ResultWriter(path) as writer:
            writer.write(Result("bad.png", "", 3.5, error="UnidentifiedImageError"))

        rows = self.read_rows(path)
        self.assertEqual(rows[1], ["1", "bad.png", "", "3.50"])

    def test_parent_directory_is_created(self) -> None:
        path = self.root / "nested" / "folder" / "out.csv"
        with ResultWriter(path) as writer:
            writer.write(Result("img0001.png", "text", 1.0))

        self.assertTrue(path.is_file())

    def test_file_is_closed_after_context_manager(self) -> None:
        path = self.root / "out.csv"
        with ResultWriter(path) as writer:
            writer.write(Result("img0001.png", "text", 1.0))
            handle = writer._handle

        self.assertTrue(handle.closed)

    def test_error_defaults_to_none(self) -> None:
        result = Result("img0001.png", "text", 1.0)
        self.assertIsNone(result.error)


if __name__ == "__main__":
    unittest.main()
