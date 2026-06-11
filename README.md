# TrollFish Local GPS Matcher (Phase 1)

This is the first implemented pipeline slice:

- Upload many video and CSV files in a browser UI
- Project creation and file upload controls are in the `Data Manager` menu (not in the main map interface)
- Main map tab is now `Viewer`, with a top quick project switch menu
- Process status and recent jobs are shown in `Data Manager`
- Click map tracks to open a synchronized video player at the selected point with scrub bar and live boat arrow
- Playback auto-continues to the next linked video when track segments continue
- Process files in a background worker
- Parse GoPro GPS tracks from MP4 via ExifTool
- Parse variable CSV schemas with timestamp normalization
- Match each video to best-fitting CSV track using multi-point GPS + time scoring
- Display all tracks and best matches on a map

## Run

1. Create and activate virtual environment (if needed):
   - `python -m venv .venv`
2. Install dependencies:
   - `.venv\Scripts\python.exe -m pip install -r requirements.txt`
3. Start app:
   - `.\run_local.ps1`
4. Open:
   - `http://127.0.0.1:8000`

## Requirements

- Python 3.12
- ExifTool installed locally (`exiftool` in PATH or default fallback path)

## API endpoints currently implemented

- `POST /api/projects`
- `GET /api/projects`
- `DELETE /api/projects/{project_id}`
- `POST /api/projects/{project_id}/uploads`
- `POST /api/projects/{project_id}/process`
- `GET /api/projects/{project_id}/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/projects/{project_id}/map-data`
- `GET /api/projects/{project_id}/files/{file_id}`
- `GET /api/projects/{project_id}/videos/{file_id}/track`
