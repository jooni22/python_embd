.DEFAULT_GOAL := run

run:
	docker compose up -d

build:
	docker compose up -d --build

reload:
	docker compose up -d --force-recreate

clear_logs:
	echo > logs/embedding_service.log
	echo > logs/reranking_service.log
	echo > logs/splade_doc_service.log
	echo > logs/splade_query_service.log

benchmark:
	chmod +x benchmark.sh
	./benchmark.sh	




#run: build
#	RUST_LOG=info cargo run -rv