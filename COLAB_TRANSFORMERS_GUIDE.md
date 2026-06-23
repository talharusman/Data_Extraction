# Colab Transformers Run Guide

This guide shows how to run the project in Google Colab with a Transformers-based Qwen3-14B model.

## What changed

- The old GGUF / `llama-cpp-python` flow is replaced by Hugging Face Transformers.
- The model is loaded with `AutoModelForCausalLM` and `AutoTokenizer`.
- The default setup uses 4-bit loading so Qwen3-14B can fit on Colab GPUs.
- The pipeline still exports the same Excel outputs.

## 1. Create a Colab notebook

1. Open Google Colab.
2. Set runtime to GPU: `Runtime` > `Change runtime type` > `Hardware accelerator` > `GPU`.
3. Clone or upload this repository into the notebook environment.

### Option A: Upload to Google Drive first

If you want to keep the project in Drive, do this first on your laptop:

1. Put the project folder in a single parent folder if it is not already grouped together.
2. Zip the folder if you want one upload file, or upload the folder contents directly.
3. Open Google Drive in your browser.
4. Drag the project folder or zip file into Drive, or use `New` > `Folder upload` / `File upload`.
5. Wait for the upload to finish completely before opening Colab.

### Option B: Upload directly into Colab

1. In Colab, click the folder icon on the left side.
2. Click the upload icon.
3. Upload the project files or zip.
4. If you uploaded a zip file, unzip it in a code cell before running the project.

If you want to clone from GitHub, use a cell like this:

```python
!git clone https://github.com/talharusman/Data_Extraction.git
%cd Data_Extraction
```

If you already uploaded the repo folder to Drive, `cd` into that folder instead.

## 2. Install dependencies

Run this in one Colab cell:

```python
!pip install -r requirements.txt
```

If you see a warning about `bitsandbytes`, restart the runtime and run the install cell again.

## 3. Set the model and runtime variables

Colab can use environment variables directly in a cell:

```python
import os

os.environ["LLM_MODEL_NAME_OR_PATH"] = "Qwen/Qwen3-14B-Instruct"
os.environ["LLM_USE_4BIT"] = "true"
os.environ["LLM_TEMPERATURE"] = "0.0"
os.environ["LLM_TOP_P"] = "0.9"
os.environ["LLM_REPETITION_PENALTY"] = "1.05"
os.environ["LLM_MAX_NEW_TOKENS"] = "1024"
```

Optional Qdrant settings:

```python
os.environ["QDRANT_URL"] = "http://localhost:6333"
```

The project already falls back to in-memory vector storage if Qdrant is unavailable, so you can usually leave this alone.

## 4. Provide the documents

Put the `Documents/` folder in the Colab workspace, or mount Google Drive and point to the folder there.

Example using Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then set the source folder path, for example:

```python
SOURCE_FOLDER = "/content/drive/MyDrive/Data_Extraction/Documents"
```

To open the uploaded project folder in Colab after mounting Drive:

1. Click the folder icon on the left sidebar in Colab.
2. Expand `drive`.
3. Open `MyDrive`.
4. Find your project folder, for example `Data_Extraction`.
5. Click the folder name to browse files inside it.
6. In a code cell, use `!cd` only for checking the path; for actual execution, set the notebook working directory with `%cd`.

## 5. Run the pipeline

You can run the existing CLI entry point from Colab:

```python
!python main.py "$SOURCE_FOLDER"
```

Or process a single file:

```python
!python main.py --file "/content/drive/MyDrive/Data_Extraction/Documents/Sample.pdf"
```

Important: do not type plain `python main.py` in a Colab code cell by itself. In a notebook cell, use `!python main.py ...` so Colab runs it as a shell command. If you want to execute Python code directly inside the notebook kernel, use a normal Python cell instead.

The main outputs are:

- `output/nbo_master_dataset.xlsx`
- `output/nbo_review_required.xlsx`
- `output/extraction_trace.jsonl`

## 6. If the model is gated on Hugging Face

Some Qwen model variants require Hugging Face access approval.

If loading fails with an authentication or permission error, run:

```python
from huggingface_hub import login
login()
```

Then paste your Hugging Face token in the prompt.

## 7. What to expect

- The first model load is slow because Colab downloads the model weights.
- 4-bit loading reduces GPU memory use.
- The extraction and validation stages still produce JSON-first outputs before Excel export.

## 8. Common problems

- If you get an out-of-memory error, restart the runtime and try again with a larger GPU or keep `LLM_USE_4BIT=true`.
- If you get tokenizer or model loading errors, confirm that `LLM_MODEL_NAME_OR_PATH` is exactly `Qwen/Qwen3-14B-Instruct` or a valid local model directory.
- If output files are not appearing, check that the notebook is running from the repo root before calling `main.py`.

## 9. Recommended notebook order

1. Clone the repo or open the uploaded folder.
2. Install requirements.
3. Set environment variables.
4. Mount Drive if needed.
5. Run the pipeline.
6. Download the Excel files from the `output/` folder.

If you are using Drive, the shortest path is usually:

1. Upload the project folder to `MyDrive`.
2. Mount Drive in Colab.
3. `%cd` into the project folder.
4. Install requirements.
5. Run `!python main.py` with the correct folder path.