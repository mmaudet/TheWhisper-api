# CI/CD Documentation

This document explains the automated build and deployment pipeline for TheWhisper-api Docker images.

## Overview

TheWhisper-api uses **GitHub Actions** to automatically build and publish Docker images to **GitHub Container Registry (GHCR)** whenever code is pushed to the main branch or a version tag is created.

## Automated Builds

### What Gets Built

Two Docker images are built automatically:

1. **faster-whisper image** (default)
   - Dockerfile: `Dockerfile`
   - Uses OpenAI Whisper models via faster-whisper
   - Open source, no licensing concerns
   - Image tags: `latest`, `cuda-latest`, version tags

2. **TheStageAI image** (advanced)
   - Dockerfile: `Dockerfile.thestage`
   - Uses TheStageAI optimized models
   - Includes TheStageAI SDK
   - Image tags: `thestage-latest`, version tags with `-thestage` suffix

### Trigger Events

Builds are triggered on:

- **Push to `main` branch**: Creates `latest` and `cuda-latest` / `thestage-latest` tags
- **Version tags** (e.g., `v1.0.0`): Creates versioned tags (e.g., `1.0.0`, `1.0`, `1`)
- **Pull requests**: Builds images for testing (not pushed)
- **Manual trigger**: Via GitHub Actions "Run workflow" button

### Image Tags

The CI/CD creates multiple tags for flexibility:

#### faster-whisper Image

| Tag | Description | Example |
|-----|-------------|---------|
| `latest` | Latest stable build from main | `ghcr.io/mmaudet/thewhisper-api:latest` |
| `cuda-latest` | Explicit CUDA version tag | `ghcr.io/mmaudet/thewhisper-api:cuda-latest` |
| `{version}` | Semantic version | `ghcr.io/mmaudet/thewhisper-api:1.0.0` |
| `{major}.{minor}` | Major.minor version | `ghcr.io/mmaudet/thewhisper-api:1.0` |
| `{major}` | Major version | `ghcr.io/mmaudet/thewhisper-api:1` |
| `main-{sha}` | Commit-specific | `ghcr.io/mmaudet/thewhisper-api:main-abc1234` |

#### TheStageAI Image

| Tag | Description | Example |
|-----|-------------|---------|
| `thestage-latest` | Latest TheStageAI build | `ghcr.io/mmaudet/thewhisper-api:thestage-latest` |
| `{version}-thestage` | Versioned TheStageAI | `ghcr.io/mmaudet/thewhisper-api:1.0.0-thestage` |
| `{major}.{minor}-thestage` | Major.minor | `ghcr.io/mmaudet/thewhisper-api:1.0-thestage` |
| `main-{sha}-thestage` | Commit-specific | `ghcr.io/mmaudet/thewhisper-api:main-abc1234-thestage` |

## Using Pre-Built Images

### Option 1: Production Docker Compose (Recommended)

Use the pre-built images with `docker-compose.prod.yml`:

```bash
# Download configuration
cp .env.cuda .env

# Pull and run with pre-built images
docker compose -f docker-compose.prod.yml up -d

# For TheStageAI version
docker compose -f docker-compose.prod.yml --profile thestage up -d
```

**Benefits:**
- ✅ No build time - instant deployment
- ✅ Images are tested via CI/CD
- ✅ Smaller download than building locally
- ✅ Automatic updates when pulling `latest`

### Option 2: Direct Docker Run

```bash
# Pull the image
docker pull ghcr.io/mmaudet/thewhisper-api:cuda-latest

# Run the container
docker run -d \
  --name whisper-api \
  --gpus all \
  -p 8000:8000 \
  -v whisper-models:/models \
  -e MODEL_NAME=large-v3-turbo \
  -e DEVICE=cuda \
  -e COMPUTE_TYPE=float16 \
  --restart unless-stopped \
  ghcr.io/mmaudet/thewhisper-api:cuda-latest
```

### Option 3: Local Build (Development)

For development or customization, use the standard docker-compose:

```bash
# Build locally
docker compose up -d
```

## Updating to Latest Version

### Automatic Updates

```bash
# Pull latest images
docker compose -f docker-compose.prod.yml pull

# Restart with new images
docker compose -f docker-compose.prod.yml up -d
```

### Pinned Versions

For production stability, pin to specific versions:

```yaml
# docker-compose.prod.yml
services:
  whisper-api:
    image: ghcr.io/mmaudet/thewhisper-api:1.0.0  # Pinned version
```

## CI/CD Workflow Details

### Build Process

1. **Checkout code** from repository
2. **Set up Docker Buildx** for advanced features
3. **Log in to GHCR** using GitHub token
4. **Extract metadata** for tags and labels
5. **Build and push** Docker image with caching
6. **Update cache** for faster subsequent builds

### Caching Strategy

The workflow uses GitHub Actions cache to speed up builds:

- **Cache layers**: Docker layers are cached between builds
- **Scope**: Separate caches for faster-whisper and TheStageAI
- **Mode**: Maximum caching (`mode=max`) for optimal performance

Typical build times:
- **First build**: ~10-15 minutes
- **Cached build**: ~2-5 minutes (only changed layers)

### Permissions

The workflow requires:
- `contents: read` - Read repository code
- `packages: write` - Push to GitHub Container Registry

These are automatically granted via `GITHUB_TOKEN`.

## Viewing Build Status

### GitHub Actions Tab

1. Go to repository: https://github.com/mmaudet/TheWhisper-api
2. Click "Actions" tab
3. View workflow runs and logs

### Build Badge (Optional)

Add to README.md:

```markdown
![Docker Build](https://github.com/mmaudet/TheWhisper-api/actions/workflows/docker-build.yml/badge.svg)
```

## Troubleshooting

### Build Failures

Check GitHub Actions logs:
1. Navigate to Actions tab
2. Click on failed workflow run
3. Expand failed step to see error

Common issues:
- **Dependency errors**: Update requirements files
- **Dockerfile syntax**: Validate Dockerfile locally
- **CUDA compatibility**: Ensure base image version matches

### Image Pull Failures

If you can't pull images:

```bash
# Public images don't require authentication, but if needed:
docker login ghcr.io -u YOUR_GITHUB_USERNAME
# Token: Personal access token with read:packages scope
```

### Cache Issues

If builds are slow or failing due to cache:

```bash
# In GitHub Actions, manually clear cache via:
# Settings > Actions > Caches > Delete specific cache
```

## Manual Triggering

### Via GitHub UI

1. Go to Actions tab
2. Select "Build and Push Docker Images" workflow
3. Click "Run workflow"
4. Select branch (usually `main`)
5. Click "Run workflow"

### Via GitHub CLI

```bash
# Install GitHub CLI: https://cli.github.com/

# Trigger workflow
gh workflow run docker-build.yml
```

## Versioning Strategy

### Semantic Versioning

We follow [Semantic Versioning](https://semver.org/):

- **Major** (1.0.0): Breaking changes
- **Minor** (1.1.0): New features, backward compatible
- **Patch** (1.1.1): Bug fixes

### Creating a Release

```bash
# Tag the release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# GitHub Actions will automatically:
# 1. Build images
# 2. Tag as: 1.0.0, 1.0, 1, latest
# 3. Push to GHCR
```

## Registry Management

### Viewing Published Images

Visit: https://github.com/mmaudet?tab=packages

Or using Docker CLI:

```bash
# List tags
docker search ghcr.io/mmaudet/thewhisper-api

# Pull specific tag
docker pull ghcr.io/mmaudet/thewhisper-api:1.0.0
```

### Image Cleanup

GitHub retains all tagged images indefinitely. To manage storage:

1. Go to package settings
2. Delete old or unused tags
3. Keep: `latest`, recent versions, production versions

## Security

### Image Scanning

Consider adding security scanning:

```yaml
# Add to .github/workflows/docker-build.yml
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: ghcr.io/mmaudet/thewhisper-api:latest
    format: 'sarif'
    output: 'trivy-results.sarif'
```

### Dependabot

GitHub Dependabot can update Docker base images:

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
```

## Best Practices

### For Development

- Use local builds: `docker compose up -d`
- Test changes before pushing to main
- Use feature branches for new features

### For Production

- Use pre-built images: `docker compose -f docker-compose.prod.yml up -d`
- Pin specific versions in production
- Update during maintenance windows
- Monitor for security updates

### For CI/CD

- Tag releases for important versions
- Let CI/CD build and test automatically
- Review build logs for warnings
- Keep cache strategy optimized

## Future Enhancements

Potential improvements:

- [ ] Multi-architecture builds (ARM64 for future NVIDIA Jetson support)
- [ ] Automated testing in CI/CD
- [ ] Security scanning integration
- [ ] Docker Hub mirroring
- [ ] Automated changelog generation
- [ ] Performance benchmarking in CI

## Support

For CI/CD issues:
- Check GitHub Actions logs
- Review this documentation
- Open an issue: https://github.com/mmaudet/TheWhisper-api/issues
