# Examples

Local development may use a symlink under `maps/` to an existing EMDB map
(e.g. `emd_32336.map.gz`). Map files are git-ignored.

Minimal run after installing weights and a FASTA database:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
python run.py \
  --map /path/to/emd_32336.map.gz \
  --fasta-db /path/to/uniprot_trembl.fasta \
  --output-dir outputs/emd_32336 \
  --job-name 7w72 \
  --device cuda
```
