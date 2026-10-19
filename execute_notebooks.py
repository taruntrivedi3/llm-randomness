"""Execute the offline notebooks and retain portable, rendered outputs."""
import argparse
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from prepare_data import main as prepare

ROOT = Path(__file__).resolve().parent
NAMES = [
    'numpy_baseline.ipynb',
    'agent_normal_sampling_2000.ipynb',
    'llm_clt_normal_2000.ipynb',
    'llm_clt_40flips_160000cal_2000.ipynb',
    'llm_permutation_normal_2000.ipynb',
]


def portable(value):
    if isinstance(value, str):
        return value.replace(str(ROOT), '<repository>')
    if isinstance(value, list):
        return [portable(x) for x in value]
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', default='llm-randomness')
    parser.add_argument('--connection-dir', type=Path, help='Optional private local directory for kernel connection files')
    args = parser.parse_args()
    prepare()
    results = []
    for name in NAMES:
        path = ROOT / 'notebooks' / name
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=600, kernel_name=args.kernel,
            resources={'metadata': {'path': str(path.parent)}})
        if args.connection_dir is not None:
            args.connection_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            client.create_kernel_manager()
            client.km.connection_file = str(args.connection_dir / (path.stem + '.json'))
        client.execute()
        code = [c for c in notebook.cells if c.cell_type == 'code']
        outputs = [o for c in code for o in c.outputs]
        assert all(c.execution_count is not None for c in code)
        assert not any(o.output_type == 'error' for o in outputs)
        images = sum('image/png' in o.get('data', {}) for o in outputs)
        assert images >= 1, name
        for cell in code:
            cell.outputs = [nbformat.from_dict(portable(o)) for o in cell.outputs]
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        result = {'notebook': name, 'executed_code_cells': len(code), 'embedded_plots': images, 'errors': 0}
        results.append(result)
        print(json.dumps(result), flush=True)
    (ROOT / 'notebook_validation.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
