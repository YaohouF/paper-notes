# Multimodal Paper Notes Site

This is a MkDocs Material site for structured paper-reading notes.

## Install

```bash
cd /Users/yoho/Desktop/my_paper/paper_notes_site
python3 -m pip install -r requirements.txt
```

## Preview locally

```bash
mkdocs serve
```

Then open:

```text
http://127.0.0.1:8000
```

## Build static site

```bash
mkdocs build
```

The generated HTML will be in:

```text
paper_notes_site/site/
```

## Publish with GitHub Pages

This repository is configured to deploy automatically with GitHub Actions.

1. Create a public GitHub repository named `paper-notes` under `YaohouF`.
2. Push this project to the repository's `main` branch.
3. In GitHub, open `Settings -> Pages`.
4. Set `Build and deployment -> Source` to `GitHub Actions`.

The published site will be:

```text
https://YaohouF.github.io/paper-notes/
```

## Add a new paper

Create:

```text
docs/papers/<paper-name>/
  index.md
  00_overview.md
  01_model_architecture.md
  02_data_construction.md
  03_training_recipe.md
  04_paper_code_crosscheck.md
  05_reproducibility_gaps.md
```

Then update `mkdocs.yml` under `nav`.
