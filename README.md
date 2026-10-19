# LLM randomness experiments

Notebooks, recorded replies, and collection code for [LLMs Failure Modes](https://taruntrivedi3.github.io/writing/llms-failure-modes/): direct normal sampling, sums of binary replies using the central limit theorem, and normal sampling from the ordering of ten integer replies.

The target is **mean 10 and standard deviation 2**. These are measured results, not a claim that fresh model calls are independent or normally distributed.

| Experiment | Notebook | Samples | Mean | Sample SD |
| --- | --- | ---: | ---: | ---: |
| Ask agents directly for normal samples | [Direct sampling](notebooks/agent_normal_sampling_2000.ipynb) | 2,000 | See notebook | See notebook |
| CLT, 40 flips, 1,000 calibration replies | [Original CLT](notebooks/llm_clt_normal_2000.ipynb) | 2,000 | 11.031815 | 2.139442 |
| CLT, 40 flips, 160,000 pooled calibration replies | [Expanded calibration](notebooks/llm_clt_40flips_160000cal_2000.ipynb) | 2,000 | 10.088840 | 2.093129 |
| Ordering ten integer replies, then inverse normal CDF | [Permutation method](notebooks/llm_permutation_normal_2000.ipynb) | 2,000 | 10.022888 | 1.957421 |

The [NumPy baseline](notebooks/numpy_baseline.ipynb) contains just the histogram example extracted from the original neural-network tutorial. It generates new NumPy values when run. The four LLM notebooks analyze saved replies and make **no API calls**.

## Reproduce the recorded analysis

Use Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py
python -m ipykernel install --user --name llm-randomness --display-name "LLM randomness"
jupyter lab notebooks
```

Select the `LLM randomness` kernel and run a notebook from top to bottom. `prepare_data.py` verifies the compressed inputs, decompresses them, and checks the resulting hashes. Allow about 1 GB of free disk space for inputs and notebook exports. No hosted dataset service is required.

Each notebook contains the intermediate inputs, grouping, arithmetic, and plots. The CLT notebooks show individual flips and frozen calibration. The permutation notebook shows the ten replies, their ranks, the permutation index, the uniform grid value, the final normal value, and rejected groups.

You can also execute and validate every notebook, then independently reconcile the outputs:

```bash
python execute_notebooks.py
python notebooks/llm_randomness/audit_results.py
python notebooks/llm_randomness/audit_pooled_clt.py
```

## What the saved experiments contain

The CLT and permutation calls used `gpt-4.1-mini`, temperature 1, top_p 1, max_tokens 32, one fresh user message per request, no seed, and up to 128 concurrent requests. The direct-agent experiment used the inherited agent-session model/settings; it is a different collection setup, and the exact underlying model was not exposed in its records.

The binary prompt was:

> Pick one at random: B or A. Respond with only the letter.

Recorded values retain the original **A = -1, B = +1** encoding. If you prefer Bernoulli outcomes, use `X = (R + 1) / 2` to get A = 0 and B = 1. Standardizing either encoding gives the same final samples.

The larger calibration pooled 91,042 historical replies and 68,958 new replies, then froze p = 0.55051875 before collecting 80,000 fresh production replies. The historical pool includes 10,042 replies from a stopped experiment. Its archived manifest may retain a running/partial status; it is provenance, not a job to restart. None of these calibration inputs are reused as that run's evaluation samples.

The integer prompt was:

> Choose an integer at random from 0 through 999999999. Respond with digits only, no explanation.

The permutation experiment accepted 2,000 out of 3,304 groups, discarding 1,304 groups with ties. It used 33,040 production replies plus a separate 1,000-call pilot. Rank order is based on the numerical answers in predetermined request positions, never completion times.

Under iid draws, conditioning on distinct replies makes rank permutations uniform even if integer preferences are uneven. Independence alone is not enough if the distributions differ across positions. Forty binary flips give only 41 possible CLT outputs; the ordering method gives a finite 10!-point grid. Neither is exactly a continuous normal distribution. The observed excess CLT variance is not explained by the finite-binomial shape approximation alone.

## Public export and provenance

The original notebooks and collection scripts have been adapted for portability. This repository preserves numerical replies, request IDs, provider response IDs, ordering, source-run membership, model/fingerprint metadata, timestamps, retry counts, and exclusions. Private provider URLs, machine-specific absolute paths, and transport-error text are omitted or replaced. Notebook outputs were cleared and regenerated using the public data.

[PUBLIC_EXPORT.json](PUBLIC_EXPORT.json) records the original and public file hashes. The notebook manifests use hashes of the public copies; they do not pretend to be byte-for-byte copies of the original private journals. [data_manifest.json](data_manifest.json) verifies the compressed and decompressed inputs. Original local datasets were not modified.

The raw response journals are large, so Git stores compressed `.jsonl.gz` files. Offline notebook execution writes uncompressed data and derived CSV/PNG exports locally; those regenerable files are ignored by Git. The calibration pool includes the entire stopped run needed to trace its source replies.

## Collect new replies

The code in [notebooks/llm_randomness](notebooks/llm_randomness) is the original asynchronous collection/resume implementation, with portable configuration paths and separate default directories for new runs. It uses an OpenAI-compatible HTTP interface, including compatible LiteLLM deployments.

Copy `provider.example.toml` to `provider.toml`, set the provider URL, and set `LLM_API_KEY` in your environment. Do not commit credentials. Choose a model name supported by your provider. This command makes paid API requests:

```bash
python notebooks/llm_randomness/run_experiment.py \
  --config provider.toml --model gpt-4.1-mini \
  --concurrency 32 --calibration-calls 1000 --flips-per-sample 40 \
  --samples 2000 --phase all
```

The output goes to `notebooks/llm_randomness/new_run/`. Provider availability, tokenization, and decoding behavior can change, so a new run need not reproduce these results. The direct-agent notebook records a separate agent-orchestration experiment; the HTTP collector is not an exact recreation of that agent setup.

`run_pooled_clt.py` supports pooling your own compatible recorded run. It validates that the pool and provider configuration match. The bundled public archive has its private endpoint removed, so it is intended for offline analysis and cannot be resumed as a live collection against a different provider. Do not alter endpoint metadata merely to bypass that check.

## Offline collection tests

```bash
python notebooks/llm_randomness/test_runner.py
python notebooks/llm_randomness/test_pooled_clt.py
```

These use mocked HTTP responses and make no model calls. They cover grouping, parsing, retries, recovery, duplicate protection, pool import, and resume behavior. The permutation notebook also verifies the ranking formula against 5,913 independently enumerated permutations.
