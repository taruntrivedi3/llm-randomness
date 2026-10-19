"""Build the offline, auditable integer-ordering LLM randomness notebook."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "llm_permutation_normal_2000.ipynb"
cells = []


def markdown(source):
    cells.append(nbf.v4.new_markdown_cell(dedent(source).strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(dedent(source).strip()))


markdown(r"""
# Normal samples from the ordering of ten LLM calls

**Target:** 2,000 samples with mean 10 and **standard deviation 2**. This notebook
reads recorded LiteLLM responses; it never calls a model or uses a random-number
generator. Missing or unfinished collection remains visible below.

Ten fresh calls receive the same integer prompt. Calls are assigned a group ID
and position **before dispatch**. For a complete group with ten distinct valid
integers, replace each answer by its rank (0 through 9), preserving request
position. Rank that permutation lexicographically to obtain

\[
K\in\{0,\ldots,10!-1\},\qquad U=\frac{K+1/2}{10!},\qquad
X=10+2\Phi^{-1}(U).
\]

Under independent, identically distributed calls, the rank permutation is uniform
conditional on all answers being distinct, even when integer preferences are
uneven. Independence alone is insufficient if call distributions differ.
Tied or malformed groups are discarded in their entirety, without tie breaking.
The first 2,000 accepted **production** groups in group-ID order are used.
The pilot is separate. These are finite-grid approximations to uniform and normal
distributions, not continuous or guaranteed random samples.
""")

code(r"""
from pathlib import Path
from itertools import permutations
import hashlib
import json
import math

import numpy as np
import pandas as pd
from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from scipy import stats

plt.style.use('ggplot')
pd.set_option('display.max_columns', 30)

# Works from the notebook directory, repository root, or experiment directory.
candidates = [Path.cwd() / 'llm_randomness',
              Path.cwd() / 'notebooks/llm_randomness', Path.cwd()]
ROOT = next((p for p in candidates if (p / 'PLAN.md').is_file()), None)
if ROOT is None:
    raise FileNotFoundError('Run from the repository, notebook, or llm_randomness directory.')
DATA = ROOT / 'data'
EXPORTS = ROOT / 'derived' / 'permutation'
EXPORTS.mkdir(parents=True, exist_ok=True)
manifest_path = DATA / 'manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
display(Markdown('**Manifest present:** ' + str(bool(manifest))))
display({k: manifest.get(k) for k in ['status', 'settings', 'endpoint', 'target_mean',
    'target_standard_deviation', 'integer_prompt', 'terminal_calls']})
display(pd.DataFrame([dict(phase=name, **{k:v for k,v in phase.items()
    if k != 'selected_group_ids'}) for name,phase in manifest.get('phases', {}).items()]))

def read_jsonl(path):
    if not path.exists():
        return [], 0
    lines = path.read_bytes().splitlines(keepends=True)
    records, trailing_partial = [], 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not line.endswith(b'\n'):
                trailing_partial += 1  # A writer may still be appending this record.
            else:
                raise
    return records, trailing_partial

responses, partial_responses = read_jsonl(DATA / 'responses.jsonl')
attempts, partial_attempts = read_jsonl(DATA / 'attempts.jsonl')
print(f'Terminal responses: {len(responses):,}; HTTP attempts: {len(attempts):,}')
print(f'Ignored unfinished trailing records: responses={partial_responses}, attempts={partial_attempts}')
print('Snapshot file SHA256:')
for filename in ['manifest.json', 'responses.jsonl', 'attempts.jsonl']:
    path = DATA / filename
    if path.exists():
        print(filename, hashlib.sha256(path.read_bytes()).hexdigest())

# The collector publishes immutable-completion hashes with this manifest key.
expected_hashes = manifest.get('file_sha256', {})
for filename, expected in expected_hashes.items():
    path = DATA / filename
    if isinstance(expected, dict):
        expected = expected.get('sha256')
    if expected:
        assert path.is_file(), f'Missing hashed file: {filename}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, filename
print('Completion hashes verified.' if expected_hashes else
      'No completion hashes supplied; this may be a partial snapshot.')

request_ids = [r['request_id'] for r in responses]
assert len(request_ids) == len(set(request_ids)), 'Duplicate terminal request IDs'
records = [r for r in responses if r.get('phase') in ['integer_pilot', 'permutation']]
for record in records:
    if record.get('valid') is True:
        token = record['raw_response'].strip()
        assert token and all('0' <= ch <= '9' for ch in token), record['request_id']
        assert int(token) == record['value'], record['request_id']
    if 'planned_index' in record:
        assert record['planned_index'] == 10 * record['group_id'] + record['position']
raw = pd.DataFrame(records)
print(f'Integer experiment terminal responses: {len(records):,}')
if records:
    display(raw.groupby(['phase', 'status'], dropna=False).size().rename('calls').to_frame())
    for key in ['model', 'system_fingerprint', 'prompt_id']:
        if key in raw:
            display(raw.groupby(['phase', key], dropna=False).size().rename('calls').to_frame())
    routing = raw.copy()
    routing['fingerprint_label'] = routing.get('system_fingerprint', pd.Series(index=routing.index, dtype=object)).fillna('<missing>')
    fingerprint_positions = routing.groupby(['phase', 'prompt_id', 'position', 'fingerprint_label'],
                                             dropna=False).size().rename('calls').reset_index()
    fingerprint_positions['position_total'] = fingerprint_positions.groupby(
        ['phase', 'prompt_id', 'position'])['calls'].transform('sum')
    fingerprint_positions['proportion_within_position'] = (
        fingerprint_positions.calls / fingerprint_positions.position_total)
    display(Markdown('**Returned fingerprint proportions by planned position.** '
                     'Each denominator is all recorded terminal calls at that position within '
                     'the phase and prompt, including invalid calls. Missing fingerprints remain visible.'))
    display(fingerprint_positions.pivot(index=['phase', 'prompt_id', 'position'],
                columns='fingerprint_label', values='proportion_within_position').fillna(0))
    fingerprint_positions.to_csv(EXPORTS / 'fingerprint_by_position.csv', index=False)
    if 'provider_response_id' in raw:
        provider_ids = raw['provider_response_id'].dropna()
        print('Repeated non-null provider response IDs:', int(provider_ids.duplicated().sum()))
    display(raw[[c for c in ['request_id', 'phase', 'group_id', 'position', 'prompt',
        'raw_response', 'value', 'valid', 'status', 'elapsed_seconds'] if c in raw]].head(30))
else:
    display(Markdown('**No integer responses have been collected in this snapshot.**'))
""")

markdown(r"""
## Independently checking the permutation arithmetic

The Lehmer digit at position \(i\) counts later ranks smaller than the current
rank. Its place value is \((9-i)!\). The checks below compare this implementation
against independently enumerated lexicographic permutations of sizes 1 through 7,
check ascending/descending endpoints, and verify that complementing every rank
reverses the permutation index. No simulated normal data is used.
""")

code(r"""
def lehmer_rank(perm):
    perm = tuple(int(x) for x in perm)
    m = len(perm)
    assert sorted(perm) == list(range(m))
    digits = [sum(other < value for other in perm[i+1:])
              for i, value in enumerate(perm)]
    k = sum(digit * math.factorial(m - 1 - i) for i, digit in enumerate(digits))
    return digits, k

checked = 0
for m in range(1, 8):
    for expected, perm in enumerate(permutations(range(m))):
        digits, actual = lehmer_rank(perm)
        assert actual == expected
        assert lehmer_rank([m - 1 - value for value in perm])[1] == math.factorial(m) - 1 - actual
        checked += 1
assert lehmer_rank(range(10))[1] == 0
assert lehmer_rank(range(9, -1, -1))[1] == math.factorial(10) - 1
assert lehmer_rank([1, 0, 2])[1] == 2  # Explicit small worked case: 102.
M = math.factorial(10)
assert M == 3628800
assert np.isclose(stats.norm.ppf(0.5, loc=10, scale=2), 10)
assert np.isclose(stats.norm.ppf(0.8, loc=10, scale=2), 11.683242467145828)
print(f'Passed {checked:,} independent enumeration cases, reverse identities, endpoints, and quantile checks.')
print(f'Uniform grid has {M:,} points; attainable normal endpoints: '
      f'{stats.norm.ppf(0.5/M, loc=10, scale=2):.6f}, '
      f'{stats.norm.ppf(1-0.5/M, loc=10, scale=2):.6f}')
""")

markdown(r"""
## Raw calls → group decisions → normal samples

Each row below contains ten integer columns in immutable request-position order,
all request IDs, ranks, Lehmer digits, \(K\), \(U\), and \(X\).
Missing positions are **incomplete**, not successful or tied calls. Any invalid
answer excludes the entire complete group. Ties are reported separately.
Raw responses and every excluded group are exported, including pilot groups and
any accepted production overshoot. HTTP attempts are preserved separately by the
collector; retries do not create extra positions.
""")

code(r"""
from collections import defaultdict

buckets = defaultdict(list)
unassigned = []
for record in records:
    if record.get('group_id') is None:
        unassigned.append(record)
    else:
        buckets[(record['phase'], int(record['group_id']))].append(record)

rows = []
for (phase, group_id), members in sorted(buckets.items()):
    positions = [int(r['position']) for r in members]
    assert len(positions) == len(set(positions)), f'Duplicate positions: {phase}/{group_id}'
    assert all(0 <= p < 10 for p in positions), f'Unexpected positions: {positions}'
    by_position = {int(r['position']): r for r in members}
    prompt_ids = sorted(set(r.get('prompt_id', '') for r in members))
    prompts = set(r.get('prompt', '') for r in members)
    assert len(prompt_ids) == len(prompts) == 1, f'Mixed prompts within {phase}/{group_id}'
    row = dict(phase=phase, prompt_id=prompt_ids[0], group_id=group_id, calls_present=len(members),
               request_ids=json.dumps([by_position[p]['request_id'] if p in by_position else None
                                       for p in range(10)]))
    fingerprints = [r.get('system_fingerprint') for r in members]
    known_fingerprints = set(f for f in fingerprints if f is not None)
    missing_fingerprints = sum(f is None for f in fingerprints)
    row['fingerprints_in_position_order'] = json.dumps([
        by_position[p].get('system_fingerprint') if p in by_position else None for p in range(10)])
    row['distinct_known_fingerprints'] = len(known_fingerprints)
    row['missing_fingerprint_calls'] = missing_fingerprints
    row['fingerprint_pattern'] = ('unknown' if missing_fingerprints else
                                  'homogeneous' if len(known_fingerprints) == 1 else 'mixed')
    values = [by_position[p].get('value') if p in by_position else None for p in range(10)]
    row.update({f'integer_{p}': value for p, value in enumerate(values)})
    complete = set(positions) == set(range(10))
    valid = complete and all(r.get('valid') is True and
                             isinstance(r.get('value'), int) and not isinstance(r.get('value'), bool)
                             for r in members)
    row['invalid_calls'] = sum(r.get('valid') is not True for r in members)
    row['distinct_integers'] = len(set(v for v in values if v is not None))
    row['decision'] = ('incomplete' if not complete else 'invalid' if not valid else
                       'tied' if len(set(values)) < 10 else 'accepted')
    if row['decision'] == 'accepted':
        rank_by_value = {value: rank for rank, value in enumerate(sorted(values))}
        perm = [rank_by_value[value] for value in values]
        digits, k = lehmer_rank(perm)
        u = (k + 0.5) / M
        row.update(permutation=json.dumps(perm), lehmer_digits=json.dumps(digits), K=k,
                   U=u, X=float(stats.norm.ppf(u, loc=10, scale=2)))
    rows.append(row)

groups = pd.DataFrame(rows)
samples = pd.DataFrame()
if not groups.empty:
    production = groups[groups.phase.eq('permutation')].sort_values('group_id')
    # Do not allow a still-incomplete earlier group to change the final first-2000 set.
    unfinished = production.loc[production.decision.eq('incomplete'), 'group_id']
    stable_cutoff = unfinished.min() if len(unfinished) else float('inf')
    present_ids = set(production.group_id)
    missing_ids = sorted(set(range(int(production.group_id.max()) + 1)) - present_ids) if len(production) else []
    if missing_ids:
        stable_cutoff = min(stable_cutoff, missing_ids[0])
    print('Entirely missing production groups before the largest recorded group ID:', missing_ids[:30])
    samples = production[production.decision.eq('accepted') &
                         production.group_id.lt(stable_cutoff)].head(2000).copy()
    samples.insert(0, 'sample_index', range(len(samples)))
    if manifest.get('phases', {}).get('permutation', {}).get('status') == 'complete':
        assert len(samples) == 2000
        assert samples.group_id.tolist() == manifest['phases']['permutation']['selected_group_ids']
    chosen_ids = set(samples.group_id)
    groups['used_for_normal_sample'] = groups.phase.eq('permutation') & groups.group_id.isin(chosen_ids)
    display(groups.groupby(['phase', 'prompt_id', 'decision']).size().rename('groups').to_frame())
    complete_routing = groups[~groups.decision.eq('incomplete')].copy()
    complete_routing['accepted'] = complete_routing.decision.eq('accepted')
    fingerprint_acceptance = complete_routing.groupby(
        ['phase', 'prompt_id', 'fingerprint_pattern']).agg(
        complete_groups=('group_id', 'size'), accepted_groups=('accepted', 'sum'),
        acceptance_rate=('accepted', 'mean'))
    display(Markdown('**Acceptance by fingerprint composition.** Denominators are complete '
                     'groups, including malformed and tied groups. A group with any missing '
                     'fingerprint is labeled unknown; incomplete groups are excluded from this table.'))
    display(fingerprint_acceptance)
    fingerprint_acceptance.to_csv(EXPORTS / 'acceptance_by_fingerprint.csv')
    display(groups.head(20))
    excluded = groups[~groups.decision.eq('accepted')]
    display(excluded.head(30))
    print(f'Excluded/incomplete groups: {len(excluded):,}; complete list exported below.')
    print(f'Accepted production samples used: {len(samples):,} / 2,000')
    for phase in ['integer_pilot', 'permutation']:
        phase_groups = groups[groups.phase.eq(phase)]
        complete_groups = phase_groups[~phase_groups.decision.eq('incomplete')]
        if len(complete_groups):
            print(phase, 'acceptance among complete groups:',
                  f'{complete_groups.decision.eq("accepted").mean():.2%}',
                  f'({complete_groups.decision.eq("accepted").sum()}/{len(complete_groups)})')
    if len(samples):
        display(samples.head(20))
else:
    print('No groups available yet.')
print('Calls without a group ID:', len(unassigned))
if unassigned:
    display(pd.DataFrame(unassigned))

raw.to_csv(EXPORTS / 'integer_calls.csv', index=False)
groups.to_csv(EXPORTS / 'all_group_intermediates.csv', index=False)
samples.to_csv(EXPORTS / 'normal_samples.csv', index=False)
pd.DataFrame(unassigned).to_csv(EXPORTS / 'unassigned_calls.csv', index=False)
if not groups.empty:
    groups[~groups.decision.eq('accepted')].to_csv(EXPORTS / 'excluded_groups.csv', index=False)
print('Full intermediate CSV exports:', EXPORTS)
if len(samples) < 2000:
    display(Markdown('**PARTIAL RESULT:** fewer than 2,000 stable accepted production groups '
                     'are present. Plots and summaries use only recorded eligible samples.'))
""")

markdown(r"""
## Uniformity, normal shape, and stability diagnostics

The comparison distribution is fixed at \(\mathcal N(10,2^2)\); we do not fit,
recenter, or rescale production samples. Continuous-uniform and continuous-normal
KS p-values below are approximate references because the generated distribution
has a finite grid. They also assume independent observations. Good histograms or
large p-values cannot establish independence, and small p-values do not identify
the cause of a discrepancy. Model/fingerprint changes, caching, dependencies, and
selection effects remain possible limitations.
""")

code(r"""
if len(samples):
    x = samples.X.to_numpy(dtype=float)
    u = samples.U.to_numpy(dtype=float)
    count = len(x)
    summary = pd.DataFrame({
        'statistic': ['count', 'mean', 'standard deviation (ddof=1)', 'minimum', 'maximum'],
        'observed': [count, x.mean(), x.std(ddof=1) if count > 1 else np.nan, x.min(), x.max()],
        'continuous target': [np.nan, 10, 2, -np.inf, np.inf],
    })
    display(summary)
    probs = np.array([0.001, 0.01, 0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 0.99, 0.999])
    quantiles = pd.DataFrame({'probability': probs, 'observed': np.quantile(x, probs),
                              'continuous target': stats.norm.ppf(probs, loc=10, scale=2)})
    display(quantiles)
    uniform_ks = stats.kstest(u, 'uniform')
    normal_ks = stats.kstest(x, 'norm', args=(10, 2))
    print('Uniform KS (continuous iid reference):', uniform_ks)
    print('Normal KS (continuous iid reference):', normal_ks)
    if count > 2:
        print('Lag-one correlation of U in group-ID order:', np.corrcoef(u[:-1], u[1:])[0, 1])
    block = pd.DataFrame({'block': np.arange(count) // 200, 'U': u, 'X': x})
    display(block.groupby('block').agg(count=('X', 'size'), mean_U=('U', 'mean'),
                                      mean_X=('X', 'mean'), std_X=('X', 'std')))

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes[0, 0].hist(u, bins=np.linspace(0, 1, 31), density=True, color='#0891b2', edgecolor='white')
    axes[0, 0].axhline(1, color='black', label='Uniform target')
    axes[0, 0].set(title='Extracted uniform values', xlabel='U', ylabel='Density')
    axes[0, 0].legend()
    empirical = np.arange(1, count + 1) / count
    axes[0, 1].step(np.sort(u), empirical, where='post', label='Observed')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Uniform target')
    axes[0, 1].set(title='Uniform ECDF', xlabel='U', ylabel='Cumulative probability')
    axes[0, 1].legend()
    axes[0, 2].plot(samples.group_id, u, '.', alpha=0.35, markersize=3)
    axes[0, 2].set(title='Immutable group ordering', xlabel='Production group ID', ylabel='U')
    grid = np.linspace(min(2, x.min()), max(18, x.max()), 600)
    axes[1, 0].hist(x, bins=40, density=True, color='#7c3aed', edgecolor='white', alpha=0.7)
    axes[1, 0].plot(grid, stats.norm.pdf(grid, loc=10, scale=2), 'k-', label='Normal(10, SD=2)')
    axes[1, 0].set(title=f'{count:,} recorded normal samples', xlabel='X', ylabel='Density')
    axes[1, 0].legend()
    theoretical = stats.norm.ppf((np.arange(count) + 0.5) / count, loc=10, scale=2)
    axes[1, 1].plot(theoretical, np.sort(x), '.', alpha=0.5, markersize=3)
    endpoints = [min(theoretical.min(), x.min()), max(theoretical.max(), x.max())]
    axes[1, 1].plot(endpoints, endpoints, 'k--')
    axes[1, 1].set(title='Q–Q against fixed target', xlabel='Target quantile', ylabel='Observed quantile')
    axes[1, 2].step(np.sort(x), empirical, where='post', label='Observed')
    axes[1, 2].plot(grid, stats.norm.cdf(grid, loc=10, scale=2), 'k--', label='Normal target')
    axes[1, 2].set(title='Normal ECDF', xlabel='X', ylabel='Cumulative probability')
    axes[1, 2].legend()
    fig.tight_layout()
    fig.savefig(EXPORTS / 'permutation_normal_diagnostics.png', dpi=160, bbox_inches='tight')
    fig.savefig(EXPORTS / 'permutation_normal_diagnostics.svg', bbox_inches='tight')
    plt.show()
    summary.to_csv(EXPORTS / 'summary.csv', index=False)
    quantiles.to_csv(EXPORTS / 'quantiles.csv', index=False)
else:
    print('No eligible production samples: no distribution plots or numerical claims generated.')
""")

markdown(r"""
## Interpretation and audit limits

One accepted group costs ten terminal calls, plus any HTTP retries. Discarded
groups and the pilot increase total cost. No value from the pilot is reused for
production, no tied group is repaired, and each selected group yields exactly
one normal sample. Invalid-response rejection preserves the ordering argument
only when validity is applied identically to identically distributed calls.

The ten-position permutation retains more information than the rank of just one
answer: it has \(10!\), rather than ten, possible values. Nevertheless its quality
depends on the upstream model calls. The notebook records returned model and
fingerprint summaries but cannot prove provider-side independence or exclude
hidden caching. Repeated outputs alone neither establish nor disprove caching.
Multiple returned fingerprints also do not, by themselves, establish a failure of
exchangeability: the same routing mixture could apply independently at every
position. The position-specific proportions and homogeneous/mixed-group acceptance
tables inspect routing provenance; they neither prove identical output
distributions nor repair a difference. Temporal routing changes can matter even
when aggregate proportions look similar. Missing fingerprints are not treated as
evidence that requests used the same backend.

The CSV files in `llm_randomness/derived/permutation/` contain all raw integer calls,
all group decisions and intermediate arithmetic, and selected normal samples.
The original `responses.jsonl` and `attempts.jsonl` remain the authoritative call
and retry audit. A final complete report requires 2,000 eligible production
samples and the collector's completed, hash-verified snapshot.
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.update(
    kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
    language_info={"name": "python", "version": "3.11"},
)
nbf.validate(notebook)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
