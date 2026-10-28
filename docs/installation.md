# Install a cluster runtime

This is a reproducible installation recipe, not a claim that a new cluster has
passed GPU acceptance. The public migration has CPU regression coverage. Its
installer and newly pinned container must still be exercised on the destination
cluster. The active Lighthouse campaign is a separate frozen deployment and is
not modified by any command below.

## Hardware and software prerequisites

The reference worker uses four GPUs in **one Slurm allocation on one node**:
GPU slot 0 renders Isaac Sim, slots 1–2 serve Qwen with TP2, and slot 3 runs SAM3
and Contact-GraspNet. Use an RTX-capable simulator GPU with sufficient memory;
four L40S cards are the reference topology. A100/H100/H200 are rejected in the
renderer slot. A Hopper-only cluster cannot run this renderer by adding more
inference GPUs. Splitting services across Hopper and RTX nodes needs a separate
network deployment; this version deliberately uses node-local services.

The site must provide Slurm, NVIDIA drivers compatible with both the Isaac Sim 6
image and the CUDA 13 model wheels, Apptainer or Singularity with NVIDIA passthrough,
Git, `uv`, and preferably `skopeo` for OCI digest resolution. A driver that merely
recognizes the GPUs is not sufficient evidence of rendering compatibility. Use
shared storage accessible at the same absolute paths on every worker; allocate
space for the model, assets, container, independent environments, videos and logs.
`SLURM_TMPDIR` should provide private node-local scratch for relays and caches.

No Docker daemon is needed. Building an Apptainer image may need the site's
approved image builder or fakeroot configuration. If local image construction is
disallowed, ask the cluster operator to build the same immutable OCI source and
retain its SIF checksum; do not substitute a similarly named image silently.

## Configure the site and credentials

From the repository root:

```bash
export ROBODATA_DATA=/shared/your-project/robodata-runtime
cp configs/clusters/rtx-slurm.example.json configs/clusters/local.json
bash deploy/bootstrap.sh
export ROBODATA_PYTHON="$ROBODATA_DATA/envs/tools/bin/python"
uv pip install --python "$ROBODATA_PYTHON" -e third_party/aspire -e .
export PATH="$ROBODATA_DATA/envs/tools/bin:$PATH"
```

Edit `configs/clusters/local.json`: select the site's account, RTX partition, GPU
resource name, CPU/memory request and array concurrency. Change absolute storage
paths if needed. Keep the four device roles distinct. The reviewed agent profile
is `configs/agents/qwen38_27b.json`; runtime rejects changes to its frozen model,
sampling, MTP, context, history or budget settings. The data directory must be
outside this Git checkout. Machine-specific `configs/**/local*.json` files are
ignored and excluded from source snapshots.

Read and accept NVIDIA's [Isaac Sim additional license](https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/)
and the external model/provider licenses yourself. Only then set
`container.accept_eula` to `true`. The installer does not accept terms on your
behalf. For SAM3, obtain access to the pinned `facebook/sam3` repository using your
own Hugging Face account. Save its token in a user-owned regular file with mode
600, outside the repository. Pass the **path**, never the token itself, to the
commands below. The installer reads it without printing it, adding it to manifests
or removing your credential file. Evaluation needs no OpenAI API key.

## Fetch pinned sources and artifacts

Downloads are CPU/network preparation and may run on an approved login or transfer
node. They do not execute a model or simulator.

```bash
robodata fetch --config configs/clusters/local.json \
  --components sources assets model providers \
  --hf-token-file /private/path/hf.token
robodata build-container --config configs/clusters/local.json
```

`deploy/sources.lock.json` pins RoboDojo, its native XPolicyLab/IsaacLab/CuRobo forks,
SAM3 and CGN commits, assets revision and provider weights. The installer applies
checksummed compatibility patches and records the resulting tracked source bytes.
These pinned leaf projects have no required nested Git submodules. Existing
checkouts with a different revision are refused instead of being reset. Qwen's
exact revision is in the agent profile; all downloaded model files receive an
inventory. SAM3 and CGN weights are checksum-verified. Assets keep their pinned
Hugging Face dataset identity and official state-content checks.

The Isaac OCI tag is resolved to an immutable digest using `skopeo`. Alternatively
configure an explicit `docker://nvcr.io/nvidia/isaac-sim@sha256:...` source. The
built SIF gets an adjacent `.manifest.json` containing that source and the image's
SHA256 and size. Workers verify it before starting services. Replacing an existing
image or source checkout requires an explicit new installation directory.

## Install isolated environments inside Slurm

Request the same GPU class that will run the evaluation. Adapt this allocation to
your site's syntax and resource policy:

```bash
salloc --account=YOUR_ACCOUNT --partition=YOUR_RTX_PARTITION \
  --nodes=1 --ntasks=1 --gres=gpu:l40s:4 --cpus-per-task=32 \
  --mem=240G --time=02:00:00
srun robodata setup --config configs/clusters/local.json
srun robodata doctor --config configs/clusters/local.json --gpu
```

Setup refuses to run outside a Slurm allocation. It creates separate Qwen, SAM3,
CGN and simulator environments. Model runtime-critical packages are pinned in
`deploy/locks/qwen.txt`; the historical full freeze is retained separately as
reference evidence. The provider locks and simulator overlay were derived from
the working environment. Full provider freezes are reconstructed with `--no-deps`:
every recorded distribution is explicit, including historical combinations whose
current dependency metadata may disagree. This preserves the tested runtime
versions; it does not claim a resolver-clean environment. The model critical pins
use ordinary dependency resolution. Installed package versions are recorded.
This is not a wheel-hashed lock of every transitive dependency: a clean install
still needs acceptance testing, and a resolver or driver failure must be reported
rather than hidden by substituting package versions.

The simulator overlay is installed **inside** the container at the same paths
used by workers: `/work/RoboDojo`, `/opt/robodata`, and `/opt/robodata-env/sim`.
This keeps editable `.pth` files valid. The benchmark `Assets` link points to the
container mount `/assets`. Native CuRobo uses its pinned default `cuda.core`
runtime compiler; the legacy optional pybind/nvcc build is not enabled. Its CUDA
header paths are taken from the pinned Isaac image. `CUDA_HOME`, `CUDA_PATH` and
`TRITON_PTXAS_PATH` from unrelated shell environments are cleared for model
installation and workers.

Kernels compile for the actual GPU architecture on first use. Set
`setup.cuda_arch_list` for the destination GPU (L40S: `8.9`); do not transfer an L40S
compiled cache and assume it is valid for a different GPU. Keep scratch caches
node-local and allow startup time for vLLM, CUDA graphs, vision and native motion
planning. `doctor --gpu` checks visibility and rejects known unsupported renderer
families; **it does not render an RGB-D image or certify a successful grasp**.

## Accept the destination deployment before broad submission

Use a new campaign, freeze it, and run one whole planned group first, following
[cluster execution](cluster.md). This is an actual scored group, not hidden
training or disposable feedback for prompt tuning. Startup verifies the model
engine and multimodal/code interface, provider load/repair evidence, source and
weight inventories, exact native context, and GPU roles. Inspect the first native
episodes, three-camera videos, budgets, public observations, provider receipts and
scoring evidence. Run the campaign audit. A successful import or model response
alone does not establish simulator correctness.

If the group passes infrastructure checks, resume the same frozen plan across the
array: already committed episodes are preserved. If the deployment needs a harness
or protocol correction, retain those results and create a new explicitly labeled
campaign. Never rerun scored failures to obtain a better score. Do not launch the
full array while the initial group is still active.

Retain the configuration, dependency installation records, source/weight/container
inventories, scheduler logs, model smoke evidence, provider health and final audit
with the result archive. See [third-party notices](third-party.md) for the separate
licenses; weights, containers and credentials are never published by this project.
