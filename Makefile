up:
	docker compose up   --remove-orphans

down:
	docker compose down

test:
	docker compose run -w /pyapp python python test.py  --remove-orphans
