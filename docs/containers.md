# Container workflow

Phase 3 provides two independently tagged images:

- `pet-trainer`: Linux CUDA 13.0 / cu130 PyTorch runtime for NVIDIA GPU training.
- `pet-inference`: CPU-only runtime for local checks and inexpensive CPU hosting.

Both images pin Python, PyTorch, and direct/transitive Python runtime package versions, run as an
unprivileged user, and default to a real ResNet-34 forward-pass smoke test. The trainer deliberately
fails instead of falling back to CPU when GPU access is missing. Its default is an overrideable
Docker command rather than a fixed entrypoint, allowing Runpod to replace it with a batch training
command at Pod creation time.

## Local smoke workflow

Build and run the portable CPU image:

```bash
docker compose build inference-smoke
docker compose run --rm inference-smoke
```

On a Linux NVIDIA host with the NVIDIA Container Toolkit configured:

```bash
docker compose build trainer-smoke
docker compose run --rm trainer-smoke
```

Successful runs print one JSON object containing package/runtime versions, component versions,
the selected device, and output tensor shapes. Compose uses a read-only root filesystem and only
provides a temporary `/tmp` mount. Later training/evaluation phases can add explicit read-only data
mounts and writable artifact mounts without baking either into an image.

## GitHub Container Registry and image tags

GitHub Container Registry is GitHub's service for storing and distributing container images. It
uses the hostname `ghcr.io`. In this project, it provides one central place from which a RunPod
machine or another deployment environment can download an exact, versioned trainer or inference
image. The images are associated with a GitHub owner or organization, but they are versioned
separately from the source repository.

Use the same version suffix for both image roles. Publish one human-readable release tag and one
immutable Git revision tag, all lowercase:

```text
ghcr.io/<owner>/pet-trainer:0.1.0
ghcr.io/<owner>/pet-trainer:sha-<12-character-git-revision>
ghcr.io/<owner>/pet-inference:0.1.0
ghcr.io/<owner>/pet-inference:sha-<12-character-git-revision>
```

Do not use `latest` for a reproducible run record. Store the pulled image digest alongside the Git
revision, model/data versions, configuration, metrics, predictions, plots, latency, and GPU memory.
The publishing script also embeds the source revision as `PET_GIT_REVISION` in each image, so
provenance collection does not depend on installing Git or copying the `.git` directory into the
container.

## Runpod training command

The trainer image includes `/app/scripts/runpod_training.sh`. It prepares the dataset, finalizes or
resumes classification when `latest.pt` already exists, then finalizes or resumes segmentation.
All output is mirrored to `/workspace/logs/training.log`. A successful run creates
`/workspace/TRAINING_COMPLETE`.

Set task-specific run names when creating the Pod and use the script as its Docker command:

```text
PET_CLASSIFICATION_RUN_NAME=classification-<revision>
PET_SEGMENTATION_RUN_NAME=segmentation-<revision>
/app/scripts/runpod_training.sh
```

For recovery of a run trained by an older image, set `PET_CLASSIFICATION_GIT_REVISION` to that
run's source revision. Provenance then distinguishes the original training revision from the
revision embedded in the image used to finalize it.

If a command fails, the script writes `/workspace/TRAINING_FAILED`, prints the exit status, and
keeps the container alive for inspection. This prevents Runpod from repeatedly restarting the
same fixed run directory. Delete the Pod promptly after inspecting a failure; the hard Pod
termination deadline remains the final cost guard.

## GitHub Container Registry script commands

The helper script performs the login, build, tagging, publishing, download, and digest-inspection
steps. The token needs GitHub's `write:packages` permission to publish images and `read:packages`
permission to download private images.

Display the built-in command summary at any time:

```bash
./scripts/github_container_registry.sh --help
```

Log in once. The script prompts for the GitHub username and token, and it does not display the
token or store it in shell history:

```bash
./scripts/github_container_registry.sh login
```

For automation, `GITHUB_USER` and `GITHUB_CONTAINER_REGISTRY_TOKEN` can instead be supplied as
environment variables. These are environment-variable names, not values that should be committed
to the repository. Docker saves a successful login through its configured credential store.

Build, tag, and publish both images with one command from anywhere inside the repository. Replace
`YOUR_GITHUB_OWNER` with the lowercase GitHub username or organization that owns the images:

```bash
./scripts/github_container_registry.sh publish YOUR_GITHUB_OWNER 0.1.0
```

The script builds both images for 64-bit Linux, creates the version and Git-revision tags, and
publishes all four tags. It never creates a `latest` tag.

Download an image by its immutable Git-revision tag and print its digest. Replace `GIT_REVISION`
with the 12-character revision printed by the publish command:

```bash
./scripts/github_container_registry.sh pull \
  YOUR_GITHUB_OWNER trainer GIT_REVISION

./scripts/github_container_registry.sh pull \
  YOUR_GITHUB_OWNER inference GIT_REVISION
```

Never commit, print, or request registry tokens.
