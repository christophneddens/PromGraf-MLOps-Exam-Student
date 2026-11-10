# Start all services: bike-api, Prometheus, node-exporter and Grafana
all:
	docker compose up --build -d bike-api prometheus node-exporter grafana

# Stop all services
stop:
	docker compose down

# Wait until the API answers, then send the 3 February weeks to /evaluate
evaluation:
	until curl -s http://localhost:8080/ > /dev/null; do echo "Waiting for the API ..."; sleep 5; done
	docker compose up --build evaluation

# Send 100 prediction requests (every 10th one invalid) to fill the API dashboard
traffic:
	bash scripts/generate_traffic.sh

# Fire the Grafana alert "High model RMSE": /trigger-drift evaluates a deliberately
# distorted batch, so model_rmse_score jumps far above the threshold of 100.
fire-alert:
	curl -s -X POST http://localhost:8080/trigger-drift
	@echo ""
	@echo "RMSE is now far above 100. In Grafana: Alerting > Alert rules > High model RMSE (Pending, then Firing after 30 s)."
	@echo "To return to normal values: make evaluation"
