## What does this PR do?

Describe the changes in this PR.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] New backend (adds a deployment backend plugin)
- [ ] Breaking change (fix or feature that would break existing functionality)
- [ ] Documentation update

## Checklist

- [ ] I've tested the full pipeline end-to-end (`python deploy.py --config deploy_config.yaml`)
- [ ] My code follows the project's style guide (typed Python, structured logging)
- [ ] I've added docstrings to new public functions and classes
- [ ] No secrets or credentials are hardcoded
- [ ] I've updated the CHANGELOG.md (if applicable)

## Backend Checklist (if adding a new backend)

- [ ] Implements all 5 abstract methods from `DeploymentBackend`
- [ ] Uses `@register_backend` decorator
- [ ] Sets `name`, `description`, and `capabilities`
- [ ] Generates Terraform under `generated/terraform/`
- [ ] Generates serving code under `generated/serving_code/`
- [ ] Returns observability instructions

## Related Issues

Closes #
