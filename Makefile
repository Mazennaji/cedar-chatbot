.PHONY: help install dev api dashboard ui test lint format docker clean

help: 
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt
	pip install black ruff mypy pytest pytest-cov pre-commit


api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	cd dashboard && python manage.py migrate && python manage.py runserver 8001

ui:
	streamlit run ui/app.py

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -m "not slow"

test-normalizer:
	pytest tests/test_normalizer.py -v

test-api:
	pytest tests/test_api.py -v

lint:
	ruff check src/ api/ tests/

format:
	black src/ api/ tests/ scripts/ ui/
	ruff check --fix src/ api/ tests/

typecheck:
	mypy src/ api/ --ignore-missing-imports

train-reward:
	python scripts/train_reward_model.py --epochs 5

fine-tune:
	python scripts/fine_tune.py --epochs 3

migrate:
	cd dashboard && python manage.py makemigrations && python manage.py migrate

superuser:
	cd dashboard && python manage.py createsuperuser

docker:
	docker-compose up --build

docker-api:
	docker-compose up --build api

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	find . -name .coverage -delete 2>/dev/null || true
	rm -rf dist/ build/