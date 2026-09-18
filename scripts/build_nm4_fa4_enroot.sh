#!/bin/bash
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --job-name=nm4-fa4-container
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/rohitkumarj/code/RL/nemo-rl-sft_v2_up_n4/container-build-nm4-fa4-%j.log

set -euo pipefail

workspace=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/rohitkumarj/code/RL/nemo-rl-sft_v2_up_n4
output_dir=/home/rohitkumarj/data/enroot-containers
output_image="${OUTPUT_IMAGE:-${output_dir}/nemo-rl-nm4-fa4-20260918.sqsh}"
partial_image="${output_image}.partial"
build_root="${SLURM_TMPDIR:-/tmp}/nm4-fa4-container-${SLURM_JOB_ID}"
buildkit_version=v0.33.0
rootlesskit_version=v3.1.0
buildkit_dir="${build_root}/buildkit"
buildkit_socket="${build_root}/buildkitd.sock"
oci_archive="${build_root}/nemo-rl-nm4-fa4.oci.tar"
podman_root="${build_root}/podman-root"
podman_runroot="${build_root}/podman-runroot"
wrapper_dir="${build_root}/bin"

if [[ -e "${output_image}" || -e "${partial_image}" ]]; then
    echo "Refusing to overwrite an existing output: ${output_image}[.partial]" >&2
    exit 1
fi

mkdir -p \
    "${buildkit_dir}" \
    "${output_dir}" \
    "${podman_root}" \
    "${podman_runroot}" \
    "${wrapper_dir}" \
    "${build_root}/xdg-runtime" \
    "${build_root}/enroot-cache" \
    "${build_root}/enroot-data" \
    "${build_root}/enroot-runtime" \
    "${build_root}/enroot-tmp"
chmod 700 "${build_root}/xdg-runtime" "${build_root}/enroot-runtime"

echo "Downloading BuildKit ${buildkit_version} for $(uname -m)"
case "$(uname -m)" in
    aarch64) buildkit_arch=arm64 ;;
    x86_64) buildkit_arch=amd64 ;;
    *) echo "Unsupported build architecture: $(uname -m)" >&2; exit 1 ;;
esac
curl -fL --retry 3 \
    -o "${build_root}/buildkit.tar.gz" \
    "https://github.com/moby/buildkit/releases/download/${buildkit_version}/buildkit-${buildkit_version}.linux-${buildkit_arch}.tar.gz"
tar -xzf "${build_root}/buildkit.tar.gz" -C "${buildkit_dir}"
curl -fL --retry 3 \
    -o "${build_root}/rootlesskit.tar.gz" \
    "https://github.com/rootless-containers/rootlesskit/releases/download/${rootlesskit_version}/rootlesskit-$(uname -m).tar.gz"
tar -xzf "${build_root}/rootlesskit.tar.gz" -C "${buildkit_dir}/bin"
export PATH="${buildkit_dir}/bin:${PATH}"
buildctl --version
buildkitd --version
rootlesskit --version

export XDG_RUNTIME_DIR="${build_root}/xdg-runtime"
rootlesskit --net=host \
    buildkitd \
        --root "${build_root}/buildkit-state" \
        --addr "unix://${buildkit_socket}" \
        --oci-worker=true \
        --containerd-worker=false \
        --oci-worker-no-process-sandbox \
        > "${workspace}/buildkitd-nm4-fa4-${SLURM_JOB_ID}.log" 2>&1 &
buildkitd_pid=$!
cleanup() {
    kill "${buildkitd_pid}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
    [[ -S "${buildkit_socket}" ]] && break
    sleep 1
done
if [[ ! -S "${buildkit_socket}" ]]; then
    echo "BuildKit did not create ${buildkit_socket}" >&2
    exit 1
fi

echo "Building the NeMo-RL release image"
cd "${workspace}"
buildctl --addr "unix://${buildkit_socket}" build \
    --progress=plain \
    --frontend=dockerfile.v0 \
    --local context=. \
    --local dockerfile=. \
    --local nemo-rl=. \
    --opt filename=docker/Dockerfile \
    --opt context:nemo-rl=local:nemo-rl \
    --opt target=release \
    --opt platform=linux/arm64 \
    --opt build-arg:UV_VERSION=0.12.5 \
    --opt build-arg:MAX_JOBS=64 \
    --opt build-arg:NVTE_BUILD_MAX_JOBS=16 \
    --opt build-arg:NVTE_BUILD_THREADS_PER_JOB=2 \
    --opt 'build-arg:NVTE_CUDA_ARCHS=100a;103a' \
    --opt 'build-arg:TORCH_CUDA_ARCH_LIST=10.0 10.3 10.3a' \
    --opt build-arg:SKIP_RUST_BUILD=1 \
    --opt build-arg:SKIP_VLLM_BUILD=1 \
    --opt build-arg:SKIP_SGLANG_BUILD=1 \
    --opt build-arg:SKIP_TRTLLM_BUILD=1 \
    --opt build-arg:CUSTOM_SETUP_FNAME= \
    --opt build-arg:NEMO_RL_COMMIT=nm4-fa4-local \
    --output "type=oci,name=localhost/nemo-rl:nm4-fa4,dest=${oci_archive}"

echo "Loading the OCI archive into temporary Podman storage"
podman \
    --root "${podman_root}" \
    --runroot "${podman_runroot}" \
    load --input "${oci_archive}"

install -m 0755 "${workspace}/scripts/podman-enroot-wrapper.sh" "${wrapper_dir}/podman"
export NM4_PODMAN_ROOT="${podman_root}"
export NM4_PODMAN_RUNROOT="${podman_runroot}"
export ENROOT_CACHE_PATH="${build_root}/enroot-cache"
export ENROOT_DATA_PATH="${build_root}/enroot-data"
export ENROOT_RUNTIME_PATH="${build_root}/enroot-runtime"
export ENROOT_TEMP_PATH="${build_root}/enroot-tmp"
export PATH="${wrapper_dir}:${PATH}"

echo "Importing the OCI image as Enroot squashfs"
enroot import --output "${partial_image}" podman://localhost/nemo-rl:nm4-fa4
unsquashfs -s "${partial_image}"

echo "The Docker build imported CUTLASS, FA4, Transformer Engine, and DeepEP in every MCore worker venv"
unsquashfs -l "${partial_image}" | grep -q \
    '/nvidia_cutlass_dsl/dsl_packages/cutlass/cute/__init__.py$'

mv "${partial_image}" "${output_image}"
echo "Completed: ${output_image}"
