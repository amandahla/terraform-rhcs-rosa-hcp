SHELL := bash

LOCALBIN ?= $(CURDIR)/bin
LOCALBIN_ABS := $(abspath $(LOCALBIN))
MODULE_REGISTRY := terraform-redhat/rosa-hcp/rhcs

# renovate: datasource=github-releases depName=google/addlicense
ADDLICENSE_VERSION ?= v1.2.0
# renovate: datasource=github-releases depName=terraform-docs/terraform-docs
TERRAFORM_DOCS_VERSION ?= v0.24.0
# renovate: datasource=github-releases depName=terraform-linters/tflint
TFLINT_VERSION ?= v0.62.1
# renovate: datasource=github-releases depName=vale-cli/vale
VALE_VERSION ?= v3.14.2
# renovate: datasource=github-releases depName=aquasecurity/trivy
TRIVY_VERSION ?= v0.70.0

TRIVY_CONFIG ?= trivy.yaml
TRIVY_IMAGE_CONFIG ?= trivy-image.yaml
SECURITY_CHECK_IMAGE ?= terraform-rhcs-rosa-hcp-clients:ci

ifeq ($(shell go env GOOS 2>/dev/null),windows)
	BIN_EXT=.exe
else
	BIN_EXT=
endif

ADDLICENSE := $(LOCALBIN)/addlicense$(BIN_EXT)
TERRAFORM_DOCS := $(LOCALBIN)/terraform-docs$(BIN_EXT)
TFLINT := $(LOCALBIN)/tflint$(BIN_EXT)
VALE := $(LOCALBIN)/vale$(BIN_EXT)
TRIVY := $(LOCALBIN)/trivy$(BIN_EXT)

export PATH := $(LOCALBIN_ABS):$(PATH)

$(LOCALBIN):
	mkdir -p "$(LOCALBIN)"

$(ADDLICENSE): | $(LOCALBIN)
	bash hack/install-release-tool.sh addlicense "$(ADDLICENSE_VERSION)" "$(LOCALBIN_ABS)"

$(VALE): | $(LOCALBIN)
	bash hack/install-release-tool.sh vale "$(VALE_VERSION)" "$(LOCALBIN_ABS)"

$(TFLINT): | $(LOCALBIN)
	bash hack/install-release-tool.sh tflint "$(TFLINT_VERSION)" "$(LOCALBIN_ABS)"

$(TERRAFORM_DOCS): | $(LOCALBIN)
	bash hack/install-release-tool.sh terraform-docs "$(TERRAFORM_DOCS_VERSION)" "$(LOCALBIN_ABS)"

$(TRIVY): | $(LOCALBIN)
	bash hack/install-release-tool.sh trivy "$(TRIVY_VERSION)" "$(LOCALBIN_ABS)"

.PHONY: tools addlicense vale tflint terraform-docs-bin license-check-bin security-check-bin trivy
tools: $(ADDLICENSE) $(VALE) $(TFLINT) $(TERRAFORM_DOCS)

addlicense: $(ADDLICENSE)
vale: $(VALE)
tflint: $(TFLINT)
terraform-docs-bin: $(TERRAFORM_DOCS)
license-check-bin: $(ADDLICENSE)
security-check-bin: $(TRIVY)
trivy: $(TRIVY)

# Merge gate: verify, verify-gen, lint, unit-tests, license-check, docs-lint (fail-fast).
# Intended single OpenShift Prow presubmit after openshift/release switches from verify + verify-gen.
.PHONY: pre-push-checks
pre-push-checks: tools
	@$(MAKE) --no-print-directory verify
	@$(MAKE) --no-print-directory verify-gen
	@$(MAKE) --no-print-directory lint
	@$(MAKE) --no-print-directory unit-tests
	@$(MAKE) --no-print-directory license-check
	@$(MAKE) --no-print-directory docs-lint

# Prow today (until consolidated): verify-format → make verify, verify-gen → make verify-gen.
# https://github.com/openshift/release/tree/master/ci-operator/config/terraform-redhat/terraform-rhcs-rosa-hcp
.PHONY: verify
verify:
	@set -euo pipefail; \
	for d in examples/*/; do \
		echo "!! Validating $$d !!"; \
		( cd "$$d" && rm -rf .terraform .terraform.lock.hcl && terraform init -backend=false -input=false && terraform validate ); \
	done

.PHONY: verify-gen
verify-gen: terraform-docs
	scripts/verify-gen.sh

.PHONY: lint
lint: $(TFLINT)
	terraform fmt -check -recursive
	terraform init -backend=false -input=false
	"$(TFLINT)" --init
	"$(TFLINT)" --recursive \
		--minimum-failure-severity=error \
		--disable-rule=terraform_required_providers \
		--disable-rule=terraform_unused_declarations \
		--disable-rule=terraform_unused_required_providers

.PHONY: unit-tests
unit-tests:
	@set -e; \
	for submodule in modules/*; do \
	  echo "== $$submodule =="; \
	  cd "$$submodule/tests" 2>/dev/null || continue; \
	  echo "== running tests for $$submodule =="; \
	  (cd .. && terraform init -backend=false -input=false && terraform test); \
	  cd ../../..; \
	done

.PHONY: license-check
license-check: $(ADDLICENSE)
	@ADDLICENSE_BIN="$(ADDLICENSE)" ADDLICENSE_VERSION="$(ADDLICENSE_VERSION)" bash scripts/add-license-header.sh -check

.PHONY: license-add
license-add: $(ADDLICENSE)
	@ADDLICENSE_BIN="$(ADDLICENSE)" ADDLICENSE_VERSION="$(ADDLICENSE_VERSION)" bash scripts/add-license-header.sh

.PHONY: docs-lint
docs-lint: $(VALE)
	@echo "Note: README and module docs are generated with 'make terraform-docs'; fix descriptions in *.tf, then run 'make verify-gen'."
	@docs=$$(find . -name '*.md' \
		-not -path './.vale/*' \
		-not -path '*/.terraform/*' \
		-not -path './.terraform-docs-cache/*' \
		-not -path './bin/*'); \
	if [ -z "$$docs" ]; then \
		echo "No Markdown files found for docs-lint"; \
		exit 1; \
	fi; \
	"$(VALE)" --minAlertLevel=error $$docs

# Security (not in pre-push-checks): IaC misconfig via trivy.yaml; image CVEs via trivy-image.yaml.
.PHONY: security-check security-check-image
security-check: $(TRIVY)
	"$(TRIVY)" config --config "$(TRIVY_CONFIG)" .

security-check-image: $(TRIVY)
	docker build -t "$(SECURITY_CHECK_IMAGE)" .
	"$(TRIVY)" image --config "$(TRIVY_IMAGE_CONFIG)" "$(SECURITY_CHECK_IMAGE)"

.PHONY: terraform-docs
terraform-docs: $(TERRAFORM_DOCS)
	@TERRAFORM_DOCS_BIN="$(TERRAFORM_DOCS)" TERRAFORM_DOCS_VERSION="$(TERRAFORM_DOCS_VERSION)" bash scripts/terraform-docs.sh

.PHONY: commits/check
commits/check:
	@./hack/commit-msg-verify.sh

# OpenShift Prow example jobs (rhcs-module-run-example-hcp): make run-example EXAMPLE_NAME=...
.PHONY: run-example
run-example:
	bash scripts/run-example.sh $(EXAMPLE_NAME)

# Maintainer utilities (not part of pre-push-checks).
.PHONY: dev-environment registry-environment change-ocp-version change-module-version
dev-environment:
	find . -type f -name "versions.tf" -exec sed -i -e "s/terraform-redhat\/rhcs/terraform.local\/local\/rhcs/g" -- {} +

registry-environment:
	find . -type f -name "versions.tf" -exec sed -i -e "s/terraform.local\/local\/rhcs/terraform-redhat\/rhcs/g" -- {} +

change-ocp-version:
	find . -type f -name "variables.tf" -exec sed -i -e 's/default = "$(OLD_VER)"/default = "$(NEW_VER)"/g' -- {} +

change-module-version:
	find ./examples -type f -name '*.tf' -exec sed -i 's^source\s*= "\.\./\.\./"^source = "$(MODULE_REGISTRY)"\n  version = "$(MODULE_VERSION)"^g' -- {} +
	find ./examples -type f -name '*.tf' -exec sed -E -i 's^source\s*= "\.\./\.\./modules/([^"]+)"^source = "$(MODULE_REGISTRY)//modules/\1"\n  version = "$(MODULE_VERSION)"^g' -- {} +

.PHONY: tests
tests:
	sh tests.sh
