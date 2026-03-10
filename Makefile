.PHONY: install test test-cov lint format docker-build docker-run docker-stop clean terraform-init terraform-plan terraform-apply

# ─── Development ─────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html -v

lint:
	black --check src/ tests/
	flake8 src/ tests/ --max-line-length=100
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/ --ignore-missing-imports

# ─── Docker ──────────────────────────────────────────────────────────────────

docker-build:
	docker build -f docker/Dockerfile -t sagemaker-mlops-pipeline .

docker-run:
	docker-compose -f docker/docker-compose.yml up -d

docker-stop:
	docker-compose -f docker/docker-compose.yml down -v

# ─── Terraform ───────────────────────────────────────────────────────────────

terraform-init:
	cd terraform && terraform init

terraform-plan:
	cd terraform && terraform plan -var-file=environments/$(ENV).tfvars

terraform-apply:
	cd terraform && terraform apply -var-file=environments/$(ENV).tfvars -auto-approve

terraform-destroy:
	cd terraform && terraform destroy -var-file=environments/$(ENV).tfvars

# ─── Cleanup ─────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache coverage.xml .coverage
