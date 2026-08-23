# Container workflow

Phase 3 provides two independently tagged images:

- `pet-mlops-trainer`: Linux CUDA 13.0 / cu130 PyTorch runtime for NVIDIA GPU training.
- `pet-mlops-inference`: CPU-only runtime for local checks and inexpensive CPU hosting.

Both images pin Python, PyTorch, and direct/transitive Python runtime package versions, run as an
unprivileged user, and default to a real ResNet-34 forward-pass smoke test. The trainer deliberately
fails instead of falling back to CPU when GPU access is missing.

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

## Image tags

Use the same version suffix for both image roles. Publish one human-readable release tag and one
immutable Git revision tag, all lowercase:

```text
ghcr.io/<owner>/pet-mlops-trainer:0.1.0
ghcr.io/<owner>/pet-mlops-trainer:sha-<12-character-git-revision>
ghcr.io/<owner>/pet-mlops-inference:0.1.0
ghcr.io/<owner>/pet-mlops-inference:sha-<12-character-git-revision>
```

Do not use `latest` for a reproducible run record. Store the pulled image digest alongside the Git
revision, model/data versions, configuration, metrics, predictions, plots, latency, and GPU memory.

## Simplest GHCR publish and pull flow

Authenticate without putting a token in a command argument or file. The token supplied on standard
input needs `write:packages` to publish and `read:packages` to pull private packages:

```bash
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io --username "$GITHUB_USER" --password-stdin
```

Build, tag, and push from the repository root:

```bash
OWNER=<lowercase-github-owner>
VERSION=0.1.0
REVISION=$(git rev-parse --short=12 HEAD)

docker build --platform linux/amd64 -f docker/Dockerfile.trainer \
  -t "ghcr.io/$OWNER/pet-mlops-trainer:$VERSION" .
docker tag "ghcr.io/$OWNER/pet-mlops-trainer:$VERSION" \
  "ghcr.io/$OWNER/pet-mlops-trainer:sha-$REVISION"
docker push "ghcr.io/$OWNER/pet-mlops-trainer:$VERSION"
docker push "ghcr.io/$OWNER/pet-mlops-trainer:sha-$REVISION"

docker build -f docker/Dockerfile.inference -t "ghcr.io/$OWNER/pet-mlops-inference:$VERSION" .
docker tag "ghcr.io/$OWNER/pet-mlops-inference:$VERSION" \
  "ghcr.io/$OWNER/pet-mlops-inference:sha-$REVISION"
docker push "ghcr.io/$OWNER/pet-mlops-inference:$VERSION"
docker push "ghcr.io/$OWNER/pet-mlops-inference:sha-$REVISION"
```

Pull by immutable Git tag (or, preferably, a recorded digest):

```bash
docker pull "ghcr.io/$OWNER/pet-mlops-trainer:sha-$REVISION"
docker image inspect "ghcr.io/$OWNER/pet-mlops-trainer:sha-$REVISION" \
  --format '{{index .RepoDigests 0}}'
```

Never commit, print, or request registry tokens. Unset `GHCR_TOKEN` after login; Docker stores the
credential through its configured credential store.
