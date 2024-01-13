# Youtube AI Summarization

Repo showcasing Youtube AI summarization tool.

YOUTUBE VIDEO SUMMARIZATION


## Summary
This repo showcases a simple, yet effective tool for youtube summarization. It can work with any youtube video and in any language supported by underlying LLM (Mistral by default).

## Demo
[streamlit-app2-2024-01-13-17-01-14.webm](https://github.com/sufyn/youtube_summarizer/assets/97327266/2496f792-b9e8-49a0-9256-976651d3b285)


## Setup

### Installing Dependencies

Install following dependencies (on macOS):

- Python 3 installation (e.g. [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Homebrew package](https://formulae.brew.sh/formula/python@3.10))
- Python packages: run `pip3 install -r requirements.txt`
- [OPTIONAL] Download `mistral-7b-openorca.Q5_K_M.gguf` model from Hugging Face [TheBloke/Mistral-7B-OpenOrca-GGUF](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/tree/main) repo into local `models` directory.

Note you can experiment with anternatives models, just update the `MODEL_FILE` variables in `web-ui.py` and/or `Notebook.ipynb`.

## Running

### StramLit Web UI

In order to run Stram Lit Web UI just run `stramlit run app.py` in the repo folder. This should open Web UI interface in the browser.

### Collab Notebook

The tool can be used as Google Collab/Notebook as well, you open the  `Notebook.ipynb` in [Collab](https://colab.research.google.com/drive/1guksSyCsGZZ_arUPJyTiAsdIhCRAliMr?usp=sharing).

## Details

### Workflow

Depending on the video size, this tool works in following modes:
1. In the simple case, if the whole video is small then summarizartion is based on adding relevant summarization prompt.
2. In case of large videos, its processed using "map-reduce" pattern:
  1. The document is first split into smaller chunks using `RecursiveCharacterTextSplitter`` which tries to respect paragraph and sentence boundarions.
  2. Each chunk is summarized separately (map step).
  3. Chunk summarization are summarized again to give final summary (reduce step).

### Local processing
processing can be done locally on the user's machine.
- Quantified Mistral model (`mistral-7b-openorca.Q5_K_M.gguf`) has around 5,1 GB.

### Performance

Relatively small to medium videos (couple of minutes) should result in processing time of around 1 min on cpu.

## Troubleshooting

None know issue.
