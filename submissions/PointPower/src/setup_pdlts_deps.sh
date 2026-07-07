#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SRC_DIR}/common.sh"
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate superpc
fi
pip install -q ninja scikit-learn fvcore iopath
pip install -q torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install -q --extra-index-url https://miropsota.github.io/torch_packages_builder \
  "pytorch3d==0.7.8+pt2.8.0cu128"
cd "${PDLTS_ROOT}/metric/PyTorchCD/chamfer3D"
python setup.py install -q
for f in "${PDLTS_ROOT}/models/layers/base/pila.cu" "${PDLTS_ROOT}/modules/base/pila.cu"; do
  if [[ -f "$f" ]] && grep -q 'x\.type()' "$f" 2>/dev/null; then
    sed -i 's/x\.type()/x.scalar_type()/g' "$f"
  fi
done
echo "[setup_pdlts_deps] OK"
