# Parallel OCR Pipeline

This is a Python OCR pipeline that reads images from the provided dataset, preprocesses each image, extracts the text with Tesseract, and writes a CSV file.

## What it does

1. Reads `config.txt`.
2. Finds all images in the image folder.
3. Preprocesses each image.
4. Extracts the text with Tesseract.
5. Postprocesses the result.
6. Writes the final results to `outputs/ocr_results.csv`.

## Requirements

- Python 3.10 or newer
- Tesseract OCR
- Pillow
- pytesseract

## Install Tesseract

Install Tesseract for Windows:

```powershell
winget install UB-Mannheim.TesseractOCR
```

The program finds Tesseract automatically, checking in order:

1. the `TESSERACT_CMD` environment variable (full path to `tesseract.exe`)
2. your `PATH`
3. `C:\Program Files\Tesseract-OCR\tesseract.exe`

If you installed it somewhere else, set `TESSERACT_CMD`:

```powershell
$env:TESSERACT_CMD = "D:\Tools\Tesseract-OCR\tesseract.exe"
```

## Install Python packages

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Prepare the images

```powershell
python scripts/prepare_dataset.py
```

This extracts the images and the ground truth (`labels.txt`) into `data/images`.

## Configuration

`config.txt` must contain exactly two lines:

```ini
image_dir=data/images
n_workers=2
```

- `image_dir` is the image folder.
- `n_workers` is the number of worker processes.

## Run

```powershell
python main.py
```

Example output:

```text
=== Parallel OCR Pipeline ===
config:    config.txt
images:    ...\data\images
workers:   2
output:    ...\outputs\ocr_results.csv

pipeline started
[10/100] images written
...
[100/100] images written
pipeline finished

images:      100
processed:   100
failed:      0
total time:  7.134 s
throughput:  14.02 images/s
avg worker:  137.3 ms/image
output:      ...\outputs\ocr_results.csv
```

## Accuracy

If `labels.txt` is in the image folder, the run ends with two scores (case-insensitive):

```text
exact match: 85/100 (85.0%)
char accuracy: 97.5% (21 edits / 838 chars)
```

- `exact match` — images whose text equals the label.
- `char accuracy` — `1 - edit distance / label characters`. The project requires at least 50%.

Without `labels.txt` the accuracy step is skipped.

## Output

`outputs/ocr_results.csv` has four columns:

- `id` — sequential number
- `filename` — image name
- `text` — recognized text
- `processing_ms` — time for that image, in milliseconds

## How it works

```text
images → loader process → work queue → N worker processes
       → result queue → parent process → CSV
```

- One loader process loads the images and puts them into a shared queue.
- N worker processes take one image at a time.
- Each worker cleans the image and runs Tesseract.
- The parent process writes the CSV and assigns sequential IDs.
- Workers stop when they receive a `None` sentinel.

## Tests

```powershell
python -m unittest discover tests
```

## Benchmark

```powershell
python scripts/benchmark.py
```

Results are written to `benchmarks/`.

## Model

The bundled English model is:

```text
models/tessdata/eng.traineddata
```

If it exists, it is used automatically.
If it is missing, Tesseract uses its installed language data.

## Notes

- The program uses processes, not threads.
- `processing_ms` is per image, not the total run time.
