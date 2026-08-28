#!/usr/bin/env bash

set -Eeuo pipefail

readonly REGISTRY_HOST="ghcr.io"
readonly PLATFORM="linux/amd64"

usage() {
  cat <<'EOF'
Manage this project's images in GitHub Container Registry.

Usage:
  scripts/github_container_registry.sh login
  scripts/github_container_registry.sh publish <github-owner> <version>
  scripts/github_container_registry.sh pull <github-owner> <trainer|inference> <git-revision>

Examples:
  scripts/github_container_registry.sh login
  scripts/github_container_registry.sh publish my-github-name 0.1.0
  scripts/github_container_registry.sh pull my-github-name trainer a1b2c3d4e5f6

The login action securely prompts for values that are not already available in
GITHUB_USER and GITHUB_CONTAINER_REGISTRY_TOKEN. The publish and pull actions
reuse Docker's saved login. No image is ever tagged as "latest".
EOF
}

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command '$1' is not installed."
}

validate_owner() {
  local owner="$1"
  [[ "$owner" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]] || fail \
    "GitHub owner must be lowercase and contain only letters, numbers, or interior hyphens."
}

validate_version() {
  local version="$1"
  [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][a-zA-Z0-9._-]+)?$ ]] || fail \
    "Version must look like 0.1.0 or 0.1.0-rc1."
}

repository_root() {
  git rev-parse --show-toplevel 2>/dev/null || fail "Run this script from inside the project."
}

login() {
  local github_user="${GITHUB_USER:-}"
  local registry_token="${GITHUB_CONTAINER_REGISTRY_TOKEN:-}"
  if [[ -z "$github_user" ]]; then
    read -r -p "GitHub username: " github_user
  fi
  if [[ -z "$registry_token" ]]; then
    read -r -s -p "GitHub Container Registry token: " registry_token
    printf '\n'
  fi
  [[ -n "$github_user" ]] || fail "GitHub username cannot be empty."
  [[ -n "$registry_token" ]] || fail "GitHub Container Registry token cannot be empty."
  printf '%s' "$registry_token" | \
    docker login "$REGISTRY_HOST" --username "$github_user" --password-stdin
}

build_tag_and_push() {
  local role="$1"
  local owner="$2"
  local version="$3"
  local revision="$4"
  local version_image="$REGISTRY_HOST/$owner/pet-$role:$version"
  local revision_image="$REGISTRY_HOST/$owner/pet-$role:sha-$revision"

  printf '\nBuilding %s image for %s...\n' "$role" "$PLATFORM"
  docker build \
    --platform "$PLATFORM" \
    --file "docker/Dockerfile.$role" \
    --build-arg "GIT_REVISION=$revision" \
    --tag "$version_image" \
    .

  docker tag "$version_image" "$revision_image"
  docker push "$version_image"
  docker push "$revision_image"

  printf 'Published:\n  %s\n  %s\n' "$version_image" "$revision_image"
}

publish() {
  [[ "$#" -eq 2 ]] || fail "publish requires <github-owner> and <version>."
  local owner="$1"
  local version="$2"
  validate_owner "$owner"
  validate_version "$version"

  local root
  local revision
  root="$(repository_root)"
  cd "$root"
  revision="$(git rev-parse --short=12 HEAD)"

  build_tag_and_push trainer "$owner" "$version" "$revision"
  build_tag_and_push inference "$owner" "$version" "$revision"

  printf '\nPublished both images for Git revision %s.\n' "$revision"
  printf 'Record the immutable sha-%s tags in the corresponding run metadata.\n' "$revision"
}

pull_image() {
  [[ "$#" -eq 3 ]] || fail \
    "pull requires <github-owner>, <trainer|inference>, and <git-revision>."
  local owner="$1"
  local role="$2"
  local revision="$3"
  validate_owner "$owner"
  [[ "$role" == "trainer" || "$role" == "inference" ]] || fail \
    "Image role must be 'trainer' or 'inference'."
  revision="${revision#sha-}"
  [[ "$revision" =~ ^[0-9a-f]{7,40}$ ]] || fail \
    "Git revision must contain 7 to 40 lowercase hexadecimal characters."

  local image="$REGISTRY_HOST/$owner/pet-$role:sha-$revision"
  docker pull --platform "$PLATFORM" "$image"
  printf 'Downloaded image digest:\n'
  docker image inspect "$image" --format '{{index .RepoDigests 0}}'
}

main() {
  require_command docker
  require_command git

  local action="${1:-}"
  case "$action" in
    login)
      [[ "$#" -eq 1 ]] || fail "login does not accept arguments."
      login
      ;;
    publish)
      shift
      publish "$@"
      ;;
    pull)
      shift
      pull_image "$@"
      ;;
    help|--help|-h)
      usage
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
