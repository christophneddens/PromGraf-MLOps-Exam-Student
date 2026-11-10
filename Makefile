# Start all services (more services are added in the next steps)
all:
	docker compose up --build -d bike-api prometheus node-exporter grafana

# Stop all services
stop:
	docker compose down

# Wait until the API answers, then send the 3 February weeks to /evaluate
evaluation:
	until curl -s http://localhost:8080/ > /dev/null; do echo "Waiting for the API ..."; sleep 5; done
	docker compose up --build evaluation
