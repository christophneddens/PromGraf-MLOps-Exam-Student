#!/bin/bash
# Sends prediction requests to the API to simulate real users.
# Every 10th request is incomplete on purpose (error 422), so that the
# error-rate panel in Grafana has something to show.
API_URL="http://localhost:8080/predict"
COUNT=${1:-100}

for i in $(seq 1 "$COUNT"); do
  HOUR=$((i % 24))
  if [ $((i % 10)) -eq 0 ]; then
    DATA='{"temp": 0.5}'
  else
    DATA="{\"temp\": 0.3, \"atemp\": 0.3, \"hum\": 0.6, \"windspeed\": 0.2, \"mnth\": 2, \"hr\": $HOUR, \"weekday\": 2, \"season\": 1, \"holiday\": 0, \"workingday\": 1, \"weathersit\": 1, \"dteday\": \"2011-02-01\"}"
  fi
  curl -s -o /dev/null -w "%{http_code} " -X POST "$API_URL" \
       -H "Content-Type: application/json" -d "$DATA"
  sleep 0.5
done
echo
echo "$COUNT requests sent to $API_URL"
