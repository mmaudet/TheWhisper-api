# Docker Hub Setup for CI/CD

This guide explains how to configure Docker Hub credentials in GitHub to enable automated image publishing.

## Prerequisites

- Docker Hub account: https://hub.docker.com/u/mmaudet
- GitHub repository: https://github.com/mmaudet/TheWhisper-api
- Repository admin access

## Step 1: Create Docker Hub Access Token

### 1.1 Go to Docker Hub Security Settings

1. Visit: https://hub.docker.com/settings/security
2. Or: Docker Hub → Account Settings → Security

### 1.2 Generate New Access Token

1. Click **"New Access Token"**
2. **Access Token Description**: `GitHub Actions - TheWhisper-api`
3. **Access permissions**: **Read & Write** (or Read, Write, Delete)
4. Click **"Generate"**
5. **⚠️ IMPORTANT**: Copy the token immediately - it won't be shown again!

Example token format:
```
dckr_pat_AbCdEf1234567890XyZ...
```

## Step 2: Add Secret to GitHub Repository

### 2.1 Go to Repository Settings

1. Visit: https://github.com/mmaudet/TheWhisper-api/settings/secrets/actions
2. Or: Repository → Settings → Secrets and variables → Actions

### 2.2 Create New Repository Secret

1. Click **"New repository secret"**
2. **Name**: `DOCKERHUB_TOKEN` (must match exactly)
3. **Secret**: Paste your Docker Hub access token
4. Click **"Add secret"**

### Verification

The secret should now appear in the list:
```
DOCKERHUB_TOKEN
Added X seconds ago
```

## Step 3: Test the Workflow

### 3.1 Trigger a Build

The workflow will automatically run when you push to main, but you can test it manually:

1. Go to: https://github.com/mmaudet/TheWhisper-api/actions
2. Select **"Build and Push Docker Images"** workflow
3. Click **"Run workflow"**
4. Select branch: `main`
5. Click **"Run workflow"**

### 3.2 Monitor the Build

1. Click on the running workflow
2. Watch the build progress in real-time
3. Both jobs should complete successfully:
   - ✅ Build faster-whisper Image
   - ✅ Build TheStageAI Image

### 3.3 Expected Output

In the logs, you should see:
```
✅ Docker images built and pushed successfully to Docker Hub!

📦 Images available at:
  - https://hub.docker.com/r/mmaudet/thewhisper-api

🐳 Pull commands:
  docker pull mmaudet/thewhisper-api:latest
  docker pull mmaudet/thewhisper-api:cuda
  docker pull mmaudet/thewhisper-api:thestage
```

## Step 4: Verify Images on Docker Hub

### 4.1 Check Your Repository

Visit: https://hub.docker.com/r/mmaudet/thewhisper-api

You should see:
- ✅ Repository exists
- ✅ Multiple tags (latest, cuda, thestage, etc.)
- ✅ Recent push timestamp
- ✅ README updated from GitHub

### 4.2 Test Pull

```bash
docker pull mmaudet/thewhisper-api:cuda
```

Should succeed without errors.

## Troubleshooting

### Secret Not Found

**Error in logs:**
```
Error: Username and password required
```

**Solution:**
1. Verify secret name is exactly `DOCKERHUB_TOKEN` (case-sensitive)
2. Check secret exists in repository settings
3. Re-create the secret if needed

### Invalid Credentials

**Error in logs:**
```
Error: Cannot perform an interactive login from a non TTY device
```

**Solution:**
1. Regenerate Docker Hub access token
2. Update GitHub secret with new token
3. Ensure token has **Read & Write** permissions

### Rate Limits

**Error:**
```
Error: toomanyrequests: You have reached your pull rate limit
```

**Solution:**
- Wait 6 hours for rate limit reset
- Or upgrade Docker Hub account

### Build Fails

**First build failing?**

Check:
1. Dockerfile syntax is valid
2. Base images are accessible
3. No syntax errors in workflow YAML

## Security Best Practices

### ✅ Do

- ✅ Use access tokens (not passwords)
- ✅ Set minimal required permissions (Read & Write)
- ✅ Name tokens descriptively
- ✅ Rotate tokens periodically (e.g., every 6 months)
- ✅ Delete unused tokens

### ❌ Don't

- ❌ Share tokens in code or commits
- ❌ Use Docker Hub password directly
- ❌ Give tokens more permissions than needed
- ❌ Use the same token across multiple projects

## Token Management

### Rotating Tokens

To rotate your access token:

1. Generate new token on Docker Hub
2. Update GitHub secret `DOCKERHUB_TOKEN`
3. Trigger a test build to verify
4. Delete old token on Docker Hub

### Revoking Access

To revoke CI/CD access:

1. Go to Docker Hub → Settings → Security
2. Find the token: `GitHub Actions - TheWhisper-api`
3. Click **"Delete"**
4. Confirm deletion

GitHub Actions will fail until a new token is configured.

## Additional Configuration

### Multiple Repositories

If you have multiple GitHub repos pushing to Docker Hub:

1. Create separate access tokens for each
2. Use descriptive names: `GitHub Actions - Repo1`, `GitHub Actions - Repo2`
3. Add respective secrets to each repository

### Organization Secrets

For organization-wide access:

1. Go to Organization → Settings → Secrets → Actions
2. Add organization secret
3. Select which repositories can access it

## Workflow Configuration

The workflow is already configured to use your credentials:

```yaml
env:
  DOCKERHUB_USERNAME: mmaudet  # Your Docker Hub username
  IMAGE_NAME: thewhisper-api   # Your image name

- name: Log in to Docker Hub
  uses: docker/login-action@v3
  with:
    username: ${{ env.DOCKERHUB_USERNAME }}
    password: ${{ secrets.DOCKERHUB_TOKEN }}  # GitHub secret
```

## Next Steps

After setup is complete:

1. ✅ Verify builds are working
2. ✅ Check images on Docker Hub
3. ✅ Test pulling images
4. ✅ Deploy to production using `docker-compose.prod.yml`

## Support

If you encounter issues:

1. Check GitHub Actions logs
2. Verify Docker Hub token is valid
3. Review this setup guide
4. Check Docker Hub status: https://status.docker.com/

## Summary Checklist

Before pushing to trigger the CI/CD:

- [ ] Docker Hub access token created
- [ ] Token has **Read & Write** permissions
- [ ] GitHub secret `DOCKERHUB_TOKEN` added
- [ ] Secret name is exactly `DOCKERHUB_TOKEN`
- [ ] Workflow file updated with username `mmaudet`
- [ ] Ready to test build

Once these are done, push to main and watch the magic happen! 🚀
