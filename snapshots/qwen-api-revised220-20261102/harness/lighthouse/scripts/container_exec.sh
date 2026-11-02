#!/usr/bin/env bash
set -euo pipefail
ROBODOJO_WORK=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ROBODOJO_STATE_DIR=${ROBODOJO_STATE_DIR:-$ROBODOJO_WORK}
module load singularity
args=(exec --nv --cleanenv --pwd /isaac-sim
  --home "$ROBODOJO_STATE_DIR/runtime-home:/home/ROBODATA_USER"
  --bind "$ROBODOJO_WORK:/work"
  --bind "$ROBODOJO_STATE_DIR/cache/kit:/isaac-sim/kit/cache"
  --bind "$ROBODOJO_STATE_DIR/cache/main:/isaac-sim/.cache"
  --bind "$ROBODOJO_STATE_DIR/cache/computecache:/isaac-sim/.nv/ComputeCache"
  --bind "$ROBODOJO_STATE_DIR/logs:/isaac-sim/.nvidia-omniverse/logs"
  --bind "$ROBODOJO_STATE_DIR/config:/isaac-sim/.nvidia-omniverse/config"
  --bind "$ROBODOJO_STATE_DIR/data:/isaac-sim/.local/share/ov/data"
  --bind "$ROBODOJO_STATE_DIR/data/documents:/isaac-sim/kit/data/documents"
 --bind "$ROBODOJO_STATE_DIR/data/kit-user:/isaac-sim/kit/data/Kit"
  --bind "$ROBODOJO_STATE_DIR/pkg:/isaac-sim/.local/share/ov/pkg")
for driver_lib in /lib64/libnvidia-allocator.so* /lib64/libnvidia-sandboxutils.so* /lib64/libnvidia-tileiras.so* /lib64/libnvidia-ngx.so*; do
  if [[ -f "$driver_lib" ]]; then args+=(--bind "$driver_lib:/opt/robodojo-driver-libs/$(basename "$driver_lib"):ro"); fi
done
if [[ -f /usr/share/nvidia/nvoptix.bin ]]; then args+=(--bind /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro); fi
args+=(--bind "/srv/ROBODATA_RUNTIME/RoboDojo/.cache/robodojo_assets_repo/Assets:/assets:ro")
args+=(--bind /sw/pkgs/arc/git/2.39.1:/sw/pkgs/arc/git/2.39.1:ro)
container_env=(/usr/bin/env VK_DRIVER_FILES=/work/config/nvidia_egl_icd.json VK_ICD_FILENAMES=/work/config/nvidia_egl_icd.json)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then container_env+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"); fi
if [[ "${ACCEPT_EULA:-}" == Y ]]; then container_env+=(ACCEPT_EULA=Y); fi
if [[ -d /usr/share/vulkan/icd.d ]]; then args+=(--bind /usr/share/vulkan/icd.d:/etc/vulkan/icd.d:ro); fi
exec singularity "${args[@]}" "$ROBODOJO_WORK/containers/isaac-sim-6.0.0.sif" /bin/bash -c 'export PATH=/sw/pkgs/arc/git/2.39.1/bin:$PATH; export LD_LIBRARY_PATH=/opt/robodojo-driver-libs:"${LD_LIBRARY_PATH:-}"; exec "$@"' bash "${container_env[@]}" "$@"
