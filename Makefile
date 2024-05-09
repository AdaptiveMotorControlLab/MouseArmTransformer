DOCKERNAME := mausspaun/lifting-transformer

docker:
	docker build -t ${DOCKERNAME} .

test:
	PYTHONPATH=. python3 -m pytest -vvv tests/

test_docker: docker build
	docker run -v $(shell pwd)/tests:/app/tests:ro \
		-w /app -u $(shell id -u) \
		-it ${DOCKERNAME} \
		python3 -m pytest -vvv /app/tests

test_contents:
	tar tf dist/lifting_transformer-*.tar.gz | sort | diff tests/contents.tgz.lst -
	unzip -lqq dist/lifting_transformer-*.whl | sed -e 's/  \+/@/g' | cut -d@ -f4 | sort | diff tests/contents.whl.lst -

tests/contents.tgz.lst:
	tar tf dist/lifting_transformer-0.2.0.tar.gz | sort > tests/contents.tgz.lst

tests/contents.whl.lst:
	unzip -lqq dist/lifting_transformer-*.whl | sed -e 's/  \+/@/g' | cut -d@ -f4 | sort > tests/contents.whl.lst

build:
	docker run -v $(shell pwd):/app \
		-w /app -u $(shell id -u) \
		--tmpfs /.local \
		--tmpfs /.cache \
		-it python:3.10 \
		bash -c "(pip install build && python -m build)"

.PHONY: docker test_docker test build
