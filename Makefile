run-api:
	uvicorn api:app --port 8080

docker:
	docker build -t tools/jarvis -f ./deployment/Dockerfile .
	
docker-compose-up:
	docker compose -f ./deployment/docker-compose.yaml up
