from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from main import build_parser, main


class MainTests(unittest.TestCase):
    def run_cli(self, argv: list[str]) -> int:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main(argv)

    def test_parser_defaults(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.config, "config.txt")
        self.assertIsNone(args.workers)
        self.assertIsNone(args.output)
        self.assertIsNone(args.limit)
        self.assertFalse(args.quiet)

    def test_rejects_zero_workers(self) -> None:
        self.assertEqual(self.run_cli(["--workers", "0"]), 2)

    def test_missing_config_returns_error(self) -> None:
        self.assertEqual(self.run_cli(["--config", "absent.txt"]), 2)

    def test_empty_image_directory_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "images"
            images.mkdir()
            config = root / "config.txt"
            config.write_text(
                f"image_dir={images}\nn_workers=2\n", encoding="utf-8"
            )

            self.assertEqual(self.run_cli(["--config", str(config)]), 2)


if __name__ == "__main__":
    unittest.main()
