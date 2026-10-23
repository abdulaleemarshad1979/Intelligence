# Deploying the casework / review build

Deploys branch `claude/face-gait-workflow-scope-a66769` (PR #1). Uses the repo's `Dockerfile` and `docker-compose.yml`.

## 1. Server prerequisites
- Linux host with Docker and Docker Compose v2, 4+ CPU cores, 8 GB RAM, 50 GB+ disk (case media and clips accumulate in the `cctv_data` volume).
- Network route from the host to the camera network (`10.243.x.x`, `10.244.x.x`) on RTSP port 554. Without it live search and live tiles cannot get frames.
- Camera RTSP credentials, and the legal authority reference you will enter on each live search job.

## 2. Get the code
```bash
git clone https://github.com/abdulaleemarshad1979/Intelligence.git
cd Intelligence
git checkout claude/face-gait-workflow-scope-a66769
```

## 3. Models (not in git)
Face matching needs `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx` in `models/`.
```bash
python3 scripts/download_models.py --all        # or copy the files into ./models
```
`docker-compose.yml` does not mount `models/`. Add this under `cctv-app.volumes`:
```yaml
      - ./models:/workspace/models:ro
```

## 4. Configuration
Create `.env` next to `docker-compose.yml` (never commit it) and pass it to `cctv-app` with `env_file: .env`:
```
CCTV_RTSP_TEMPLATE=rtsp://{ip}:554/stream0     # per-camera URL; {ip} comes from the camera sheet
CCTV_RTSP_USER=<camera user>
CCTV_RTSP_PASSWORD=<camera password>
CCTV_MODELS_DIR=/workspace/models
```
Before exposing the server:
- Change the Postgres password in `docker-compose.yml` (`postgres`/`postgres` is a placeholder) and stop publishing `5432`, `9997` to the network (remove those `ports:` lines).
- Put the app behind HTTPS (nginx/Caddy reverse proxy) and restrict access to the police network. The app listens on port 8000 over plain HTTP.

## 5. Start
```bash
docker compose build cctv-app
docker compose up -d
docker compose ps            # all three services healthy
docker compose logs -f cctv-app
```

## 6. Load the camera inventory (once)
The grid reads `config/camera_inventory.json` (already in the repo, 544 cameras). The casework database also needs the sheet for search and clip candidates:
```bash
docker compose cp "Matrix 544 DT. 10.09.2026.xlsx" cctv-app:/tmp/matrix.xlsx
docker compose exec cctv-app python scripts/import_camera_inventory.py /tmp/matrix.xlsx
```
Camera serial 255 has bad coordinates in the sheet; correct the row and re-import (the import is safe to repeat).

## 7. Test after deploy
1. `curl -s localhost:8000/cameras/page?page_size=2` returns 544 total and names like `Thadithota-1`.
2. `curl -s localhost:8000/api/notifications` returns `[]` with no suspects enrolled.
3. Open the dashboard: 24 tiles per page, paging works, search finds a camera by IP (`10.243.0.4`).
4. Enroll a volunteer with **Enroll Suspect** (photo or video). Name is optional.
5. Walk the volunteer past one real camera. Expect one alert, a saved image, and a video clip link in the alert card. Walk away: no new alerts.
6. Casework API (`/docs`): create a case, upload a photo, crop and enhance it, view the lineage, record accept/reject/defer on a candidate.
7. Face-similarity search: create a RECORDED job on a clip of the volunteer. Only that clip should produce review items. Then a LIVE job on 1-2 cameras; confirm it expires and stops.
8. Run the accuracy harness on labelled clips from your own cameras before relying on any match threshold: `python scripts/evaluate_face_accuracy.py --manifest manifest.json`.

## 8. Roll back
```bash
docker compose down
git checkout main && docker compose build cctv-app && docker compose up -d
```
Data in the `cctv_data` and `pgdata` volumes is kept.

## Known limits at this build
- Live search has only been tested with fake frames, not on a real RTSP camera.
- Match scores are unvalidated on CCTV; every result is a lead for a person to review.
- The grid still shows placeholder people counts and risk numbers for each camera.
- Live search jobs stop if the server restarts.
