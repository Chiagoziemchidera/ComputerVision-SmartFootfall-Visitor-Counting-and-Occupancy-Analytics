# Computer Vision Footfall and Visitor Analytics System

## Project Overview

This project is a computer vision pilot for measuring visitor traffic at a physical location using two CCTV cameras.

The system uses one camera for people entering and another camera for people leaving. It detects and tracks each person, creates a temporary mathematical representation of the person's face, and applies defined counting rules.

The pilot is designed to report:

- Daily unique footfall
- Total physical entries
- Total physical exits
- Current occupancy
- Peak occupancy
- Repeat entries
- Qualified repeat visits after a two-hour absence
- Entry and exit timestamps
- Unresolved entries where a usable face was not captured

The system does not need to know a visitor's name or store visitors image. Facial information (mathematical representation of the person's face) is used only to determine whether the same person has already been seen during that day.

## Project Purpose

Normal CCTV footage shows what happened, but it does not automatically convert visitor movement into useful business information.

This project was developed to turn entrance and exit footage into measurable traffic data that can support decisions such as:

- Identifying peak and low-traffic periods
- Measuring the number of unique visitors per day
- Measuring how many people are inside at a particular time
- Tracking repeat visits separately from daily unique footfall
- Comparing traffic patterns by hour or day
- Supporting staffing, space planning and operational decisions

## Business Definitions

The project separates visitor traffic into different measurements because each one answers a different business question.

### Daily unique footfall

A person is counted once as daily unique footfall on their first valid entry of the day.

If the same person leaves and returns later that day, daily unique footfall does not increase again.

### Physical entries

Every valid entry increases the physical-entry count, including repeat entries.

### Physical exits

Every valid exit increases the physical-exit count.

### Current occupancy

Current occupancy is calculated as:

```text
Physical entries - Physical exits
```

Every valid entry increases occupancy by one. Every valid exit reduces occupancy by one.

### Repeat entry

A repeat entry occurs when a person who has already entered that day returns to the location.

### Qualified repeat visit

A repeat entry becomes a qualified repeat visit when the person has been outside the location for at least two hours.

The two-hour rule is calculated from the person's most recent exit time, not from their first entry time.

### Example

| Time | Activity | Daily unique | Physical entry | Occupancy change | Repeat entry | Qualified repeat |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 9:00 a.m. | First entry | 1 | 1 | +1 | 0 | 0 |
| 10:00 a.m. | Exit | 0 | 0 | -1 | 0 | 0 |
| 10:30 a.m. | Returns after 30 minutes | 0 | 1 | +1 | 1 | 0 |
| 11:00 a.m. | Exit | 0 | 0 | -1 | 0 | 0 |
| 4:00 p.m. | Returns after five hours | 0 | 1 | +1 | 1 | 1 |

For this example, the person contributes:

- 1 daily unique visitor
- 3 physical entries
- 2 repeat entries
- 1 qualified repeat visit

## How the System Works

The application follows this process:

1. Open the entry and exit video sources.
2. Read the footage frame by frame.
3. Detect people in each selected frame.
4. Assign a temporary tracking number to each detected person.
5. Follow the person's movement toward the counting line.
6. Detect a visible face and create a facial embedding.
7. Detect when the person crosses the virtual counting line.
8. Compare the embedding with the temporary identities already created that day.
9. Apply the entry, exit, unique-footfall, occupancy and repeat-visit rules.
10. Save permanent traffic events and temporary daily identities in separate databases.
11. Delete previous-day facial identities when a new day begins.

## Computer Vision Models and Tools

### YOLOX-Nano

YOLOX-Nano detects people in the footage.

It receives a video frame and returns the position of each person it finds. It does not recognise the person's identity and it does not perform the business counting by itself.

Model file:

```text
models/yolox_nano.onnx
```

Official download:

[Download YOLOX-Nano ONNX](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx)

The downloaded model uses a 416 by 416 input size.

### ByteTrack

ByteTrack follows a detected person across consecutive video frames.

For example, if YOLOX detects the same person in 20 frames, ByteTrack helps the application understand that these detections belong to one moving person rather than 20 different people.

In this project, ByteTrack is supplied through the Python `supervision` package. It does not require a separate ONNX model file.

### YuNet

YuNet locates faces in the video frame.

It returns the face position and facial landmarks used to align the face before creating an embedding.

Model file:

```text
models/face_detection_yunet.onnx
```

Official download:

[Download YuNet](https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx?download=true)

After downloading it, rename the file to:

```text
face_detection_yunet.onnx
```

### SFace

SFace converts an aligned face into a facial embedding.

A facial embedding is a list of numbers representing facial features. The application compares this numerical representation with other temporary embeddings created during the same day.

The embedding is not a visitor's name, but it is still biometric information and must be protected.

Model file:

```text
models/face_recognition_sface.onnx
```

Official download:

[Download SFace](https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx?download=true)

After downloading it, rename the file to:

```text
face_recognition_sface.onnx
```

## Technologies Used

| Technology | Purpose |
| --- | --- |
| Python | Runs the complete application |
| OpenCV | Reads videos, detects faces and handles image operations |
| ONNX Runtime | Runs the downloaded YOLOX model |
| Supervision | Provides the ByteTrack implementation |
| NumPy | Handles model outputs and facial vectors |
| PyYAML | Reads the settings in `config.yaml` |
| SQLite | Stores traffic events and temporary daily identities |
| ZoneInfo | Applies the timezone |
| VS Code | Recommended program for editing and running the project |

## Recommended Windows Project Structure

Create one folder called `footfall-pilot` inside the Windows Documents folder.

```text
C:\Users\YourName\Documents\footfall-pilot\
│
├── footfall_pilot.py
├── config.yaml
├── requirements.txt
├── README.md
├── .gitignore
│
├── models\
│   ├── yolox_nano.onnx
│   ├── face_detection_yunet.onnx
│   └── face_recognition_sface.onnx
│
├── videos\
│   ├── entry.mp4
│   └── exit.mp4
│
└── data\
```

### Meaning of each file and folder

| Item | Meaning |
| --- | --- |
| `footfall_pilot.py` | The main Python application |
| `config.yaml` | The editable settings used by the application |
| `requirements.txt` | The list of Python packages required by the project |
| `README.md` | The project documentation shown on GitHub |
| `.gitignore` | Prevents private, temporary and large files from being uploaded |
| `models` | Contains the three downloaded trained models |
| `videos` | Contains entry and exit test footage |
| `data` | Contains the databases created when the application runs |

The folders are not stored inside the Python file. They are separate items inside the main `footfall-pilot` folder.


## Windows Installation

### 1. Install Python

Install Python from [python.org](https://www.python.org/downloads/).

During installation, select:

```text
Add Python to PATH
```

### 2. Install VS Code

Install [Visual Studio Code](https://code.visualstudio.com/).

Open the `footfall-pilot` folder in VS Code using:

```text
File > Open Folder
```

### 3. Open the VS Code terminal

In VS Code, select:

```text
Terminal > New Terminal
```

Confirm that the terminal is inside the project folder.

### 4. Create the virtual environment

```powershell
python -m venv .venv
```

A virtual environment keeps this project's Python packages separate from packages used by other projects.

Windows PowerShell may block the activation script. Activation is not required. The environment's Python program can be used directly.

### 5. Install the packages

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Requirements File

The `requirements.txt` file tells Python which additional packages the project needs.

```text
numpy>=1.26,<3.0
opencv-contrib-python-headless>=4.10
onnxruntime>=1.18
supervision>=0.24
PyYAML>=6.0
```

Running the installation command reads this file and installs the listed packages inside `.venv`.

## Configuration File

The `config.yaml` file allows important settings to be changed without editing the Python code.

Example:

```yaml
location:
  id: PILOT-LOCATION-01
  timezone: Africa/Lagos

operation:
  test_mode: true
  business_start: "08:00"
  business_end: "21:00"
  repeat_visit_hours: 2
  face_match_threshold: 0.42
  face_detection_threshold: 0.85
  minimum_face_width: 60
  process_every_nth_frame: 2
  face_every_nth_frame: 3
  show_preview: false

models:
  person_detector: models/yolox_nano.onnx
  face_detector: models/face_detection_yunet.onnx
  face_recognizer: models/face_recognition_sface.onnx
  person_input_width: 416
  person_input_height: 416
  person_confidence: 0.35
  person_nms_threshold: 0.45

database:
  events: data/footfall_events.sqlite
  identities: data/daily_identities.sqlite

cameras:
  - id: ENTRY-CAMERA-01
    role: entry
    source: videos/entry.mp4
    counting_line: [0.10, 0.60, 0.90, 0.60]
    allowed_crossing: any

  - id: EXIT-CAMERA-01
    role: exit
    source: videos/exit.mp4
    counting_line: [0.10, 0.60, 0.90, 0.60]
    allowed_crossing: any
```

### Important configuration settings

| Setting | Simple explanation |
| --- | --- |
| `location.id` | Unique name for the pilot location |
| `timezone` | Uses Nigerian local time |
| `test_mode: true` | Processes recorded videos immediately |
| `test_mode: false` | Uses the configured business hours for live processing |
| `business_start` | Live counting begins at 8:00 a.m. |
| `business_end` | Live counting stops at 9:00 p.m. |
| `repeat_visit_hours` | Minimum absence required for a qualified repeat visit |
| `face_match_threshold` | Controls how similar two embeddings must be to match |
| `minimum_face_width` | Rejects faces that are too small for dependable matching |
| `process_every_nth_frame` | Reduces processing work by skipping some frames |
| `show_preview` | Controls whether a video window is displayed |
| `person_input_width` | Width expected by the YOLOX-Nano model |
| `person_input_height` | Height expected by the YOLOX-Nano model |
| `person_confidence` | Minimum confidence for accepting a person detection |
| `counting_line` | Position of the invisible entrance or exit line |
| `allowed_crossing` | Controls which movement direction can trigger a count |

`allowed_crossing: any` is suitable only for early testing. After reviewing the camera movement, it should be changed to `negative_to_positive` or `positive_to_negative` so that each camera counts only the intended direction.

## Running the Pilot

Before running the application, confirm that:

- The three ONNX models are inside the `models` folder.
- The test videos are called `entry.mp4` and `exit.mp4`.
- Both videos are inside the `videos` folder.
- `config.yaml` uses an input width and height of 416.
- The required Python packages have been installed.

Run:

```powershell
.\.venv\Scripts\python.exe footfall_pilot.py --config config.yaml
```

This command starts the complete process.


### `main`

The `main` function connects every part of the application. It loads the configuration, models and databases, starts both camera workers, displays progress in the terminal and closes the application safely when processing ends.

## Output Data

### Permanent traffic database

```text
data/footfall_events.sqlite
```

This database contains non-biometric operational results.

Main tables:

| Table | Contents |
| --- | --- |
| `counters` | Daily totals and current/peak occupancy |
| `crossing_events` | Each entry or exit event and its timestamp |
| `occupancy_snapshots` | Occupancy after each recorded movement |


## Viewing the Results

The easiest graphical option is [DB Browser for SQLite](https://sqlitebrowser.org/dl/).

Open:

```text
data/footfall_events.sqlite
```

Then select `Browse Data` and choose a table.

The tables can also be exported to CSV and analysed in Excel or Power BI.

## Moving from Recorded Videos to Live Cameras

For recorded-video testing:

```yaml
test_mode: true
source: videos/entry.mp4
```

For a live IP camera, the source changes to an RTSP stream and test mode changes to false.

Example:

```yaml
test_mode: false
source: ${ENTRY_CAMERA_RTSP}
```

Camera passwords should be stored in environment variables or another protected secrets file. They should not be written directly inside `config.yaml` or uploaded to GitHub.

The live application should run on a dedicated local mini PC at the location. The laptop can be used for development and recorded-video testing.

## Privacy and Data Protection

This project processes facial embeddings for temporary same-day matching. These embeddings should be treated as biometric personal data even though they do not contain the visitor's name.

The intended privacy controls are:

- Process footage locally at the location.
- Do not upload face photographs or embeddings to the central database.
- Do not save face crops unless specifically required for an approved test.
- Keep temporary identities separate from permanent traffic events.
- Delete previous-day embeddings and temporary visitor IDs.
- Restrict access to the local computer and databases.
- Do not upload test footage, databases or credentials to GitHub.
- Display an appropriate CCTV and visitor-analytics notice.
- Complete a data-protection impact assessment before live deployment.
- Confirm the lawful basis and other compliance requirements before processing visitors' biometric data.

Daily deletion reduces retention risk, but it does not remove the responsibilities associated with biometric processing.

## Testing the Pilot

The system should be tested against manual counts.

Recommended test process:

1. Use staged footage with volunteers before using public visitor footage.
2. Record the manual number of entries and exits.
3. Run the same footage through the application.
4. Compare physical entries, exits, unique visitors and repeat visits.
5. Review unresolved identities.
6. Adjust the counting line and crossing direction.
7. Adjust the person and face confidence settings where necessary.
8. Calibrate the face-match threshold using footage from the actual entrance.
9. Test people walking side by side, turning around, waiting near the line and returning later.
10. Run a longer pilot before relying on the figures for business decisions.

## Current Limitations

This is a pilot application, not a finished production product.

Known limitations include:

- Poor lighting, face coverings and unsuitable camera angles can reduce face-matching accuracy.
- People entering side by side may be harder to detect and match.
- A person without a usable face can be counted physically but cannot be confirmed as a unique visitor.
- Facial similarity thresholds require location-specific testing.
- Occupancy will be incorrect if the application starts after people are already inside.
- The system cannot reliably distinguish staff from customers without another staff-identification method.
- In the current pilot, recorded video events use the processing time. Historical-video analysis needs an added recording start time to reproduce the original event timestamps.
- The current SQLite setup is suitable for a pilot. Production use requires stronger encryption, backup, access control and deletion verification.
- Camera disconnections, power loss and computer restarts require monitoring and recovery procedures.

## Recommended Production Improvements

Possible next steps include:

- Add a secure HTTPS API for sending only non-biometric results to the central database.
- Add an offline upload queue when internet access is unavailable.
- Encrypt the temporary identity database.
- Add camera-health and processing-health alerts.
- Add automatic startup using Windows Task Scheduler, Docker or a system service.
- Add a dashboard for daily footfall, occupancy and peak-period reporting.
- Add recorded-video timestamps for historical analysis.
- Add automated accuracy reports against manual counts.
- Add tests for counting rules, database retention and identity matching.
- Separate the single Python script into smaller modules as the application grows.
- Introduce a staff QR code, access card or another approved method if staff must be excluded.


## Project Status

This repository represents a one-location pilot and proof of concept.

The immediate objective is to test whether the selected computer vision models, camera positions and counting rules can produce dependable daily unique footfall, occupancy and repeat-visit measurements before a wider rollout.

## Disclaimer

This project is for controlled pilot testing and learning. It should not be used as a production biometric-surveillance system without technical validation, security controls, documented governance and applicable legal review.

