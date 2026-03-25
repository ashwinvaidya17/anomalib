---
name: anomalib-write-docs
description: Use when writing, updating, or reviewing anomalib documentation. Covers Sphinx + MyST Markdown conventions, the Diátaxis framework (tutorials, how-to guides, explanations, references), sphinx-design components (grids, cards, tabs, dropdowns), API reference generation via autodoc, code inclusion patterns, cross-referencing, and the full docs site hierarchy. Triggers on phrases like "write docs", "add documentation", "update docs", "document this", "write a tutorial", "write a guide", "API reference", "docstring".
---

# Writing Anomalib Documentation

This skill guides writing documentation for anomalib. The docs use **Sphinx** with **MyST Markdown** as the primary format, the **sphinx-book-theme**, and **sphinx-design** for rich UI components. The content follows the **Diátaxis framework**.

**Before starting, read these context files for code-level understanding:**

- `docs/agent-context/index.md` — Architecture overview
- `docs/agent-context/models.md` — Model architecture (for documenting models)
- `docs/agent-context/engine.md` — Engine (for documenting training/inference)
- `docs/agent-context/data.md` — Data pipeline (for documenting datasets)

---

## 1. Documentation Stack

### Core Tools

| Tool                     | Purpose                       | Config Location                              |
| ------------------------ | ----------------------------- | -------------------------------------------- |
| Sphinx                   | Doc generator                 | `docs/source/conf.py`                        |
| MyST Parser              | Markdown → Sphinx             | `myst_parser` extension in conf.py           |
| sphinx-book-theme        | HTML theme                    | `html_theme` in conf.py                      |
| sphinx-design            | Cards, grids, tabs, dropdowns | `sphinx_design` extension                    |
| nbsphinx                 | Jupyter notebook rendering    | `nbsphinx` extension                         |
| autodoc + napoleon       | API docs from docstrings      | `sphinx.ext.autodoc`, `sphinx.ext.napoleon`  |
| sphinx_autodoc_typehints | Type hints in API docs        | Extension in conf.py                         |
| intersphinx              | Cross-project references      | Links to `python`, `torch`, `lightning` docs |

### Key Sphinx Config Settings (`docs/source/conf.py`)

````python
# MyST extensions — all available in docs
myst_enable_extensions = [
    "colon_fence",       # ::: directive syntax
    "linkify",           # Auto-link URLs
    "substitution",      # Variable substitution
    "tasklist",          # - [ ] checkboxes
    "deflist",           # Definition lists
    "fieldlist",         # Field lists
    "amsmath",           # Math blocks
    "dollarmath",        # Inline $math$
]
myst_enable_eval_rst = True  # CRITICAL: allows ```{eval-rst} blocks in Markdown
````

### Build Command

```bash
# From project root
sphinx-build -b html docs/source docs/build

# Or using the docs README guidance
cd docs && make html
```

---

## 2. Site Hierarchy — Where Pages Live

```text
docs/source/
├── conf.py                          # Sphinx configuration
├── index.md                         # Landing page with top-level toctrees
├── _static/                         # Static assets (logos, favicons)
│   └── images/
│       └── logos/                    # Brand logos
├── _templates/                      # Custom Sphinx templates
├── markdown/                        # ALL narrative documentation
│   ├── get_started/                 # Tutorials (Diátaxis: Tutorial)
│   │   ├── anomalib.md              # "Anomalib in 15 Minutes" quickstart
│   │   └── migration.md             # Migration guide
│   ├── guides/
│   │   ├── how_to/                  # How-To Guides (Diátaxis: How-To)
│   │   │   ├── index.md             # Hub with grid cards
│   │   │   ├── data/                # Data-related how-tos
│   │   │   ├── models/              # Model-related how-tos
│   │   │   ├── evaluation/          # Evaluation/metrics how-tos
│   │   │   ├── pipelines/           # Pipeline how-tos
│   │   │   ├── visualization/       # Visualization how-tos
│   │   │   └── training_on_intel_gpus/  # Hardware-specific
│   │   ├── reference/               # API Reference (Diátaxis: Reference)
│   │   │   ├── index.md             # Hub with grid cards
│   │   │   ├── data/                # Data module API
│   │   │   ├── models/              # Models API
│   │   │   ├── engine/              # Engine API
│   │   │   ├── metrics/             # Metrics API
│   │   │   ├── callbacks/           # Callbacks API
│   │   │   ├── pre_processing/      # PreProcessor API
│   │   │   ├── post_processing/     # PostProcessor API
│   │   │   ├── visualization/       # Visualization API
│   │   │   ├── cli/                 # CLI reference
│   │   │   ├── deploy/              # Deployment API
│   │   │   ├── loggers/             # Logger API
│   │   │   ├── pipelines/           # Pipelines API
│   │   │   └── utils/               # Utilities API
│   │   └── developer/               # Explanation (Diátaxis: Explanation)
│   │       ├── index.md             # Developer hub
│   │       ├── sdd.md               # Software Design Document
│   │       ├── contributing.md      # Contribution guide
│   │       ├── code_review_checklist.md
│   │       └── release_guidelines.md
│   └── announcements/               # News & community
│       ├── recognition.md
│       └── engagement.md
├── snippets/                        # Reusable code snippets (literalinclude targets)
│   ├── install/                     # Installation commands
│   ├── train/                       # Training code
│   ├── pipelines/                   # Pipeline examples
│   └── logging/                     # Logger config
└── examples/                        # Runnable example scripts
    ├── api/                         # Python API examples
    │   └── 01_getting_started/
    │       └── basic_training.py
    ├── cli/                         # CLI command examples
    └── notebooks/                   # Jupyter notebooks
```

### Where New Pages Go

| Page Type                | Location                              | Example                      |
| ------------------------ | ------------------------------------- | ---------------------------- |
| Getting started tutorial | `markdown/get_started/`               | `anomalib.md`                |
| How-to guide (data)      | `markdown/guides/how_to/data/`        | `datamodules.md`             |
| How-to guide (models)    | `markdown/guides/how_to/models/`      | `custom_model.md`            |
| How-to guide (new topic) | `markdown/guides/how_to/{topic}/`     | Create `index.md` + subpages |
| API reference page       | `markdown/guides/reference/{module}/` | `index.md`                   |
| Developer/explanation    | `markdown/guides/developer/`          | `sdd.md`                     |
| Announcement             | `markdown/announcements/`             | `recognition.md`             |

**After creating a new page, ALWAYS:**

1. Add it to the parent `index.md` toctree
2. Add a grid-item-card on the parent hub page (if it's a new section)

---

## 3. Diátaxis Framework — The Four Page Types

Anomalib follows the [Diátaxis framework](https://diataxis.fr/). Every doc page is one of four types. **Never mix them.**

### 3.1 Tutorial (Learning-Oriented)

**Location:** `markdown/get_started/`
**Purpose:** Walk a newcomer through a complete experience. They LEARN by DOING.
**Tone:** Friendly, encouraging, step-by-step. "Let's..." / "You'll..."

**Rules:**

- Start with a clear goal and time estimate ("In 15 minutes, you'll train your first model")
- Every step must produce a visible result
- Don't explain WHY — just show WHAT to do
- Don't offer choices — pick one path and follow it
- Include complete, runnable code (via `literalinclude` from `examples/`)
- End with "What's Next" pointing to how-to guides

**Template:**

````markdown
# Tutorial Title

Brief description of what the reader will accomplish and approximately how long it will take.

## Prerequisites

- List what they need before starting
- Link to installation instructions if needed

## Step 1: Set Up

Narrative explaining what we're about to do (1-2 sentences).

::::{tab-set}
:::{tab-item} API
:sync: api

```{literalinclude} ../../../../examples/api/path/to/example.py
:language: python
:lines: 1-15
```

:::
:::{tab-item} CLI
:sync: cli

```bash
anomalib train --model Patchcore --data anomalib.data.MVTecAD
```

:::
::::

## Step 2: Train the Model

Continue step-by-step...

## Step 3: View Results

Show the output they should expect.

## What's Next

- {doc}`../guides/how_to/models/index` — Learn to customize model parameters
- {doc}`../guides/how_to/data/index` — Use your own dataset
````

### 3.2 How-To Guide (Task-Oriented)

**Location:** `markdown/guides/how_to/{topic}/`
**Purpose:** Show how to accomplish a specific task. Assumes basic competence.
**Tone:** Direct, imperative. "To do X, configure Y."

**Rules:**

- Title starts with "How to..." (or an action verb)
- Address ONE specific task per page
- Assume the reader knows the basics
- Show the most common approach first, then alternatives
- Include both API and CLI approaches using tab-sets
- Link to reference docs for parameter details

**Template:**

````markdown
# How to [Accomplish Specific Task]

Brief context (1-2 sentences) — what problem this solves.

## Prerequisites

- What they need to have done first (with links)

## Steps

### 1. Configure [Component]

Explanation of what to configure and why.

::::{tab-set}
:::{tab-item} API
:sync: api

```python
from anomalib.data import Folder

datamodule = Folder(
    root="path/to/dataset",
    normal_dir="good",
    abnormal_dir="defective",
)
```

:::
:::{tab-item} CLI
:sync: cli

```bash
anomalib train --data anomalib.data.Folder \
    --data.root path/to/dataset \
    --data.normal_dir good \
    --data.abnormal_dir defective
```

:::
::::

### 2. Run [Action]

Next step...

## Common Variations

::::{dropdown} Using a different [option]
Alternative approach details.
::::

## See Also

- {doc}`../reference/data/index` — Full API reference for data modules
````

### 3.3 Reference (Information-Oriented)

**Location:** `markdown/guides/reference/{module}/`
**Purpose:** Describe the API precisely. The reader LOOKS UP specific details.
**Tone:** Dry, accurate, complete. No opinions, no tutorials.

**Rules:**

- One page per module/class (or group of related classes)
- Use `eval-rst` blocks with `autoclass`/`automodule` for API surface
- Include ALL parameters, return types, and exceptions
- Show type signatures
- Order: classes, then functions, then constants
- Cross-reference related modules via intersphinx or `{doc}` links

**Template:**

````markdown
# Module Name

Brief one-sentence description of the module's purpose.

```{eval-rst}
.. currentmodule:: anomalib.module.path

.. autoclass:: ClassName
   :members:
   :show-inheritance:
```

## Related

- {doc}`../data/index` — Data modules used with this component
- {doc}`../engine/index` — Engine that orchestrates this component
````

### 3.4 Explanation (Understanding-Oriented)

**Location:** `markdown/guides/developer/`
**Purpose:** Explain WHY things work the way they do. Provide context and background.
**Tone:** Discursive, analytical. "The reason for this design is..."

**Rules:**

- Don't include step-by-step instructions (that's a how-to)
- Don't include complete runnable examples (that's a tutorial)
- DO explain design decisions, architecture choices, tradeoffs
- DO reference related tutorials and how-tos
- May include diagrams, architecture sketches
- May include code snippets to illustrate concepts (not to be copied)

**Template:**

```markdown
# [Topic] — Design & Architecture

## Overview

High-level explanation of the concept and why it exists.

## How It Works

Technical explanation of the mechanism, with diagrams if helpful.

## Design Decisions

### Why [Choice A] Over [Choice B]

Rationale...

### Trade-offs

What we gained and what we gave up.

## See Also

- {doc}`../../get_started/anomalib` — Tutorial using this concept
- {doc}`../how_to/topic/index` — Practical guides
```

---

## 4. MyST Markdown Syntax

### 4.1 Format Rules

- **ALL docs are MyST Markdown** (`.md`), NOT reStructuredText
- RST is ONLY used inside `{eval-rst}` blocks for autodoc directives
- Use MyST colon-fence syntax (`:::`) for directives, NOT backtick-fence for non-code directives
- Headings: `#` for title (one per page), `##` for sections, `###` for subsections
- One blank line before and after every directive block

### 4.2 Admonitions

```markdown
:::{note}
Helpful information that clarifies something.
:::

:::{warning}
Something that could cause problems if ignored.
:::

:::{tip}
A useful suggestion or shortcut.
:::

:::{important}
Critical information the reader must not miss.
:::

:::{seealso}
Related resources or documentation pages.
:::
```

### 4.3 Inline Markup

```markdown
`code_element` — Code, functions, classes, parameters
**bold** — Emphasis on key terms (sparingly)
_italic_ — First use of a technical term
{doc}`path/to/page` — Cross-reference another doc page
{class}`anomalib.models.Patchcore` — Cross-reference a Python class
{func}`anomalib.engine.Engine.fit` — Cross-reference a function
{ref}`anchor-name` — Cross-reference a named anchor
```

### 4.4 Cross-Referencing

**Within the docs:**

```markdown
See {doc}`../guides/how_to/data/index` for data setup instructions.
```

**To a named anchor:**

```markdown
(my-anchor)=

## Section Title

...elsewhere...
See {ref}`my-anchor` for details.
```

**To Python objects (via intersphinx or autodoc):**

```markdown
See {class}`anomalib.engine.Engine` for the full API.
See {class}`torch.nn.Module` for the base class. <!-- intersphinx: torch -->
See {class}`lightning.pytorch.LightningModule` for Lightning integration. <!-- intersphinx: lightning -->
```

---

## 5. sphinx-design Components

These are the UI building blocks used throughout anomalib docs. Use them consistently.

### 5.1 Grid + Cards (Navigation Hubs)

Used on index pages to create visual navigation to sub-sections.

```markdown
::::{grid} 2 2 2 3
:gutter: 2
:padding: 1

:::{grid-item-card} {octicon}`database` Data
:link: ./data/index
:link-type: doc

Learn how to configure and use anomalib data modules.
:::

:::{grid-item-card} {octicon}`cpu` Models
:link: ./models/index
:link-type: doc

Explore available anomaly detection models.
:::

:::{grid-item-card} {octicon}`gear` Engine
:link: ./engine/index
:link-type: doc

Training, inference, and export orchestration.
:::

::::
```

**Grid spec:** `::::{grid} <xs> <sm> <md> <lg>` — columns at each breakpoint.

**Card link types:**

- `:link-type: doc` — Link to another document page (relative path, no `.md` extension)
- `:link-type: ref` — Link to a named anchor (defined with `(anchor-name)=`)
- `:link-type: url` — Link to an external URL

**Common octicon icons used in anomalib docs:**

- `{octicon}\`database\`` — Data
- `{octicon}\`cpu\`` — Models
- `{octicon}\`gear\`` — Engine
- `{octicon}\`meter\`` — Metrics
- `{octicon}\`eye\`` — Visualization
- `{octicon}\`package\`` — Installation / Packages
- `{octicon}\`terminal\`` — CLI
- `{octicon}\`rocket\`` — Deploy
- `{octicon}\`tools\`` — Utils / Developer
- `{octicon}\`graph\`` — Pipelines
- `{octicon}\`log\`` — Loggers

### 5.2 Tab Sets (API / CLI Alternatives)

Used to present parallel approaches (Python API vs CLI) on the same page.

````markdown
::::{tab-set}
:::{tab-item} API
:sync: api

```python
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

engine = Engine()
engine.fit(model=Patchcore(), datamodule=MVTecAD())
```

:::
:::{tab-item} CLI
:sync: cli

```bash
anomalib train --model Patchcore --data anomalib.data.MVTecAD
```

:::
::::
````

**Rules for tab-sets:**

- ALWAYS use `:sync:` keys so tabs stay synced across the page
- Use `api` and `cli` as sync keys (established convention)
- Both tabs must accomplish the same task
- If only one approach exists, don't use tabs — just show the code

### 5.3 Dropdowns (Collapsible Content)

Used for advanced options, long output, or optional details that would break reading flow.

````markdown
::::{dropdown} Advanced: Configure Hardware-Specific Options
:icon: gear

Content that is hidden by default. Use for:

- Advanced configuration that most readers don't need
- Long code examples or output
- Alternative approaches

```python
# Advanced configuration example
engine = Engine(accelerator="gpu", devices=2)
```

::::
````

**When to use dropdowns:**

- Installation extras / hardware variants
- Advanced configuration options
- Verbose output or logs
- Implementation details in tutorials
- **Never** for critical information the reader needs

### 5.4 Cards (Standalone)

For highlighting important links or resources outside of grids.

```markdown
:::{card} Getting Started Guide
:link: ../get_started/anomalib
:link-type: doc

Train your first anomaly detection model in 15 minutes.
:::
```

---

## 6. Code Inclusion Patterns

### 6.1 Literalinclude from Examples

The preferred pattern for showing code — pull from real, tested example files.

````markdown
```{literalinclude} ../../../../examples/api/01_getting_started/basic_training.py
:language: python
:lines: 10-30
```
````

**Path conventions:**

- From `docs/source/markdown/get_started/`: use `../../../../examples/...`
- From `docs/source/markdown/guides/how_to/data/`: use `../../../../../../examples/...`
- The path is relative to the current `.md` file

**Options:**

- `:language: python` — Syntax highlighting (always specify)
- `:lines: 10-30` — Include only specific lines (prevents massive blocks)
- `:linenos:` — Show line numbers
- `:emphasize-lines: 3,5` — Highlight specific lines
- `:caption: Training example` — Add a caption above the block

### 6.2 Literalinclude from Snippets

Reusable code fragments stored in `docs/source/snippets/`.

````markdown
```{literalinclude} /snippets/install/pypi.txt
:language: bash
```
````

**Snippet path convention:** Use absolute path from `docs/source/` root, starting with `/snippets/`.

**When to create a snippet:**

- Command or code block used in 2+ pages
- Installation instructions (they change with releases)
- Configuration examples that must stay in sync

### 6.3 Inline Code Blocks

For small, illustrative code that doesn't need to be runnable.

````markdown
```python
from anomalib.models import Patchcore

model = Patchcore(backbone="resnet18")
```
````

**Rules:**

- ALWAYS specify the language for syntax highlighting
- Prefer `literalinclude` over inline blocks for anything > 10 lines
- Inline blocks for conceptual snippets; literalinclude for runnable examples

---

## 7. API Reference Generation

### 7.1 Autodoc via eval-rst Blocks

API reference pages use `{eval-rst}` blocks to invoke Sphinx autodoc directives inside MyST Markdown.

````markdown
# Engine

The Engine class orchestrates training, validation, testing, prediction, and export.

```{eval-rst}
.. currentmodule:: anomalib.engine.engine

.. autoclass:: Engine
   :members:
   :show-inheritance:
```
````

**Common autodoc directives:**

| Directive                     | Use For                                 |
| ----------------------------- | --------------------------------------- |
| `.. autoclass:: ClassName`    | Document a class with its methods       |
| `.. automodule:: module.path` | Document all public members of a module |
| `.. autofunction:: func_name` | Document a single function              |

**Common autodoc options:**

| Option                               | Effect                                |
| ------------------------------------ | ------------------------------------- |
| `:members:`                          | Include all public methods/attributes |
| `:show-inheritance:`                 | Show base classes                     |
| `:undoc-members:`                    | Include members without docstrings    |
| `:inherited-members:`                | Include methods from parent classes   |
| `:exclude-members: method1, method2` | Skip specific members                 |
| `:no-index:`                         | Don't add to the general index        |

### 7.2 Docstring Format (Google Style)

Anomalib uses **Google-style docstrings** parsed by Napoleon. Follow this format exactly.

```python
def fit(
    self,
    model: AnomalibModule,
    datamodule: AnomalyDataModule | None = None,
    ckpt_path: str | None = None,
) -> None:
    """Train the model on the given dataset.

    This method wraps Lightning's Trainer.fit() with anomalib-specific
    setup for pre-processing, post-processing, and evaluation.

    Args:
        model: The anomaly detection model to train.
        datamodule: Data module providing train/val dataloaders.
            If ``None``, uses the model's default dataset.
        ckpt_path: Path to a checkpoint to resume training from.

    Raises:
        ValueError: If neither ``datamodule`` nor model default is available.

    Example:
        >>> from anomalib.models import Patchcore
        >>> from anomalib.data import MVTecAD
        >>> engine = Engine()
        >>> engine.fit(model=Patchcore(), datamodule=MVTecAD())
    """
```

**Docstring rules:**

- First line: Imperative mood, one sentence, ends with period
- Blank line after first line
- `Args:` section with type + description for each parameter
- Use double backticks for inline code in docstrings: ` `None` `
- `Returns:` section with type + description
- `Raises:` section listing exception types
- `Example:` section with doctest-compatible code (>>> prefix)
- Type hints go in the signature, NOT duplicated in the docstring Args section

---

## 8. Toctree Patterns

### 8.1 Hub Page with Hidden Toctree

Every index page has a grid of cards for visual navigation PLUS a hidden toctree for Sphinx's document tree.

````markdown
# How-To Guides

Practical guides for common anomalib tasks.

::::{grid} 2 2 2 3
:gutter: 2

:::{grid-item-card} {octicon}`database` Data
:link: ./data/index
:link-type: doc

Configure datasets and data modules.
:::

:::{grid-item-card} {octicon}`cpu` Models
:link: ./models/index
:link-type: doc

Customize and configure models.
:::

::::

```{toctree}
:caption: Guides
:hidden:

./data/index
./models/index
```
````

**Rules:**

- The toctree MUST list every subpage (Sphinx needs this for navigation)
- Use `:hidden:` so the toctree list doesn't render visually (the grid replaces it)
- The toctree goes at the BOTTOM of the page
- `:caption:` is optional but recommended for sidebar navigation labels
- Paths are relative, no `.md` extension

### 8.2 Adding to the Top-Level Toctree

If creating a new top-level section, add it to `docs/source/index.md`:

````markdown
```{toctree}
:caption: Section Name
:hidden:

markdown/path/to/new_section/index
```
````

---

## 9. Images and Assets

### 9.1 Model Architecture Diagrams

Store in `docs/source/images/{model_name}/` or alongside the model README.

```markdown
![PatchCore Architecture](../../images/patchcore/architecture.png)
```

Or using MyST figure directive for captions:

```markdown
:::{figure} ../../images/patchcore/architecture.png
:alt: PatchCore Architecture
:width: 600px

PatchCore extracts patch-level features from a pretrained backbone and stores them in a memory bank.
:::
```

### 9.2 Static Assets

Brand logos, favicons, and CSS go in `docs/source/_static/`.

```markdown
![Anomalib Logo](/_static/images/logos/anomalib-wide-blue.png)
```

---

## 10. Writing Style Guide

### Tone and Voice

- **Direct and instructional** — Don't waffle. Get to the point.
- **Second person** — "You can configure..." not "One can configure..."
- **Active voice** — "The engine trains the model" not "The model is trained by the engine"
- **Present tense** — "This function returns..." not "This function will return..."
- **Imperative for instructions** — "Run the following command" not "You should run..."

### Formatting Conventions

- **Code elements** always in backticks: `Engine`, `fit()`, `--model`, `config.yaml`
- **File paths** in backticks: `src/anomalib/models/`
- **Parameter names** in backticks: `backbone`, `learning_rate`
- **First mention** of a technical term in _italics_
- **Key terms** in headings or at the start of definitions in **bold**
- **One sentence per line** in source (makes git diffs cleaner) — OPTIONAL but preferred

### Page Structure Rules

- **One H1 (`#`) per page** — This is the page title
- **H2 (`##`) for major sections** — These appear in the sidebar
- **H3 (`###`) for subsections** — These appear nested in the sidebar
- **Don't skip heading levels** — Never go from `##` to `####`
- **Keep pages focused** — If a page exceeds ~500 lines, split it

### Common Mistakes to Avoid

| Mistake                                            | Correct Approach                                                       |
| -------------------------------------------------- | ---------------------------------------------------------------------- |
| Mixing tutorial and reference content              | Keep them in separate pages                                            |
| Writing long paragraphs of explanation in a how-to | Move explanation to a developer/explanation page                       |
| Using RST syntax outside `eval-rst` blocks         | Use MyST Markdown syntax                                               |
| Hardcoding code examples inline                    | Use `literalinclude` from `examples/` or `snippets/`                   |
| Forgetting to add page to toctree                  | ALWAYS update the parent `index.md` toctree                            |
| Using raw HTML                                     | Use sphinx-design components instead                                   |
| Skipping the `:sync:` key on tab-items             | Always sync tabs so they stay coordinated                              |
| Writing "click here" links                         | Use descriptive link text: "See the {doc}`data guide <../data/index>`" |

---

## 11. Checklist — Before Submitting Documentation

### New Page Checklist

- [ ] Page follows ONE Diátaxis type (tutorial, how-to, reference, or explanation)
- [ ] Placed in the correct directory per the hierarchy
- [ ] Added to parent `index.md` toctree
- [ ] Added grid-item-card on hub page (if new section)
- [ ] Has exactly one H1 heading (page title)
- [ ] Heading levels don't skip (H1 → H2 → H3)
- [ ] Code blocks specify language (`:language: python`)
- [ ] Runnable code uses `literalinclude` from `examples/` or `snippets/`
- [ ] API/CLI alternatives use `tab-set` with `:sync:` keys
- [ ] Cross-references use `{doc}`, `{ref}`, or `{class}` — not raw URLs for internal links
- [ ] No raw HTML — use sphinx-design components
- [ ] Images have alt text

### API Reference Page Checklist

- [ ] Uses `{eval-rst}` block with `autoclass` or `automodule`
- [ ] `.. currentmodule::` set correctly
- [ ] `:members:` and `:show-inheritance:` included
- [ ] Brief intro paragraph above the autodoc block
- [ ] Related modules linked with `{doc}` or `{class}`

### Docstring Checklist

- [ ] Google style (Args, Returns, Raises, Example sections)
- [ ] First line: imperative, one sentence, period
- [ ] All parameters documented in Args
- [ ] Type hints in signature (not duplicated in docstring)
- [ ] Double backticks for inline code (` `None` `, ` `True` `)
- [ ] Example with `>>>` prefix for doctest compatibility

### Pre-Submission

- [ ] Build passes: `sphinx-build -b html docs/source docs/build` with no errors
- [ ] No broken cross-references (check build warnings)
- [ ] Pages render correctly in local preview
- [ ] Snippet files exist if referenced via `literalinclude`

---

## 12. Common Recipes

### Recipe: Add a New How-To Guide

1. Create the file: `docs/source/markdown/guides/how_to/{topic}/{page}.md`
2. Write content following the How-To template (Section 3.2)
3. Add to the topic's `index.md` toctree:

   ````markdown
   ```{toctree}
   :hidden:
   ./{page}
   ```
   ````

4. Add a grid-item-card on the topic's `index.md` (if new topic area)
5. Build and verify: `sphinx-build -b html docs/source docs/build`

### Recipe: Add API Reference for a New Module

1. Create: `docs/source/markdown/guides/reference/{module}/index.md`
2. Add the eval-rst autodoc block:

   ````markdown
   # Module Name

   ```{eval-rst}
   .. currentmodule:: anomalib.{module}

   .. autoclass:: ClassName
      :members:
      :show-inheritance:
   ```
   ````

3. Add to `docs/source/markdown/guides/reference/index.md`:
   - Grid-item-card in the grid
   - Entry in the toctree
4. Build and verify

### Recipe: Add a Reusable Snippet

1. Create: `docs/source/snippets/{category}/{name}.txt`
2. Include in docs:

   ````markdown
   ```{literalinclude} /snippets/{category}/{name}.txt
   :language: python
   ```
   ````

3. Reuse the same path in any page that needs it

### Recipe: Document a New Model

1. Write the model README: `src/anomalib/models/image/{model}/README.md`
   - Include algorithm overview, architecture diagram, benchmark table
   - Follow existing model READMEs (e.g., `patchcore/README.md`)
2. Add API reference page: `docs/source/markdown/guides/reference/models/{model}.md`
3. Add to models reference index toctree
4. Optionally add a how-to guide if the model has unique configuration

---

## 13. Quick Reference — Directive Syntax

````markdown
# Admonitions

:::{note} :::
:::{warning} :::
:::{tip} :::
:::{important} :::
:::{seealso} :::

# sphinx-design

::::{grid} 2 2 2 3 ::::
:::{grid-item-card} :::
::::{tab-set} ::::
:::{tab-item} Name :::
::::{dropdown} Title ::::
:::{card} Title :::

# Code inclusion

```{literalinclude} path
:language: python
:lines: 1-20
```

# API docs

```{eval-rst}
.. autoclass:: anomalib.module.Class
   :members:
```

# Cross-references

{doc}`relative/path` — link to doc page
{ref}`anchor-name` — link to anchor
{class}`anomalib.module.Class` — link to class
{func}`anomalib.module.func` — link to function

# Anchors

(anchor-name)=

## Heading

# Figures

:::{figure} path/to/image.png
:alt: Description
:width: 600px
Caption text.
:::

# Toctree

```{toctree}
:caption: Section
:hidden:
./subpage
```
````
