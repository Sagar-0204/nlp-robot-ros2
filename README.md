# Natural-language commands for a ROS 2 robot (LLM + Nav2)

Type *"go to the red box, then come back"* and a TurtleBot3 does it in Gazebo.
An LLM turns the sentence into a **validated JSON plan** made of 8 safe actions, Nav2
carries it out, and a camera + LiDAR detector finds "the red box". The LLM never touches
motor commands, and every plan is checked before the robot moves.

**Stack:** Ubuntu 24.04 · ROS 2 Jazzy · Gazebo Harmonic · Nav2 · TurtleBot3 waffle · Python ·
OpenCV · Ollama (local LLM: Qwen2.5 7B / 3B) or a cloud LLM (Google Gemini, Anthropic, any OpenAI-compatible API)

## Results

| Language layer (26 test commands) | Correct plans | Correct refusals | Time per command |
|---|---|---|---|
| Regex baseline | 14/26 | 5/5 | 0 s |
| Qwen2.5-7B, first prompt (v1) | 20/25 | 4/5 | - |
| **Qwen2.5-7B, final (v3)** | **26/26** | **5/5** | **2.9 s** (RTX 3050 Laptop GPU, 4 GB) |
| Qwen2.5-3B, final (v3) | 20/26 | 3/5 | 0.5 s |
| **Gemini 3.5 Flash-Lite (cloud), final (v3)** | **26/26** | **5/5** | **1.4 s** |

<p>
  <img src="results/figures/accuracy_by_version.png" width="49%" alt="Accuracy by version">
  <img src="results/figures/speed_vs_accuracy.png" width="49%" alt="Speed vs accuracy">
</p>

- Box detection error in simulation: **0.03–0.24 m**. A box moved by hand is found again
  at its new position.
- Estimating distance from the camera where the 3.5 m LiDAR can't reach cut
  "go to the red box" from **132 s to 19 s**.
- Ctrl+C stops the robot mid-command; refused commands ("bring me a coffee") never move it.
- Local or cloud is one launch argument (`llm_backend:=ollama` / `gemini`). The free-tier
  cloud ran out of quota mid-evaluation, and any backend failure falls back to a rule parser
  instead of leaving the robot without a plan.

Full write-up with failure analysis, safety design and limitations:
**[docs/RESULTS.md](docs/RESULTS.md)**

## How it works

```
"go to the red box,     ┌────────────────────┐ JSON ┌─────────────────────┐ valid ┌───────────────┐
 then come back"  ────► │ LLM (Ollama/cloud) │ ───► │ validator:          │ plan  │ commander     │ ──► Nav2
                        │ structured output  │ ◄─── │ 8 actions, limits,  │ ────► │ action server │     (path, spin,
                        └────────────────────┘ 1 fix│ targets, directions │       └───────────────┘      drive)
                          ▲ grounding: places, seen boxes, robot pose, last 3 commands        ▲
                          │                                                                   │ box positions
                        ┌─┴────────────────────────────────────────────────────────────────┐  │
                        │ colour detector: camera (direction) + LiDAR/camera (distance) ───┼──┘
                        └──────────────────────────────────────────────────────────────────┘
```

## Quick start

```bash
sudo apt install ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-turtlebot3 \
     ros-jazzy-turtlebot3-simulations ros-jazzy-ros-gz python3-opencv python3-requests
curl -fsSL https://ollama.com/install.sh | sh && ollama pull qwen2.5:7b-instruct

git clone https://github.com/Sagar-0204/nlp-robot-ros2.git ~/nlp_robot_ws && cd ~/nlp_robot_ws
colcon build --symlink-install && source install/setup.bash

ros2 launch nlp_robot bringup.launch.py        # terminal 1: Gazebo + Nav2 + RViz + LLM commander
ros2 run nlp_robot nlp_cli                     # terminal 2: type commands
```
Without an LLM: `ros2 launch nlp_robot bringup.launch.py llm_backend:=rules`. The language
layer alone, with no ROS: `python3 src/nlp_robot/tools/plan_shell.py --backend ollama`.

## How it was built

The project was built in eight stages, each tested before the next. The sections below
follow the same order.

| # | Stage | Status |
|---|---|---|
| 1 | Workspace + custom interfaces | ✅ |
| 2 | Room world + TurtleBot3 in Gazebo | ✅ |
| 3 | Nav2 on a saved map | ✅ |
| 4 | Language layer: plan schema, validation, LLM backends | ✅ |
| 5 | Commander action server (plan → Nav2) | ✅ |
| 6 | CLI client | ✅ |
| 7 | Colour-box detection | ✅ |
| 8 | Evaluation | ✅ |

## Packages

| Package | Type | What it holds |
|---|---|---|
| `nlp_robot_interfaces` | ament_cmake | `ExecuteCommand.action`, `DetectedObject(Array).msg` |
| `nlp_robot` | ament_python | All the Python nodes and the language layer |

## The room

`nlp_robot/world_spec.py` is the single source of truth for the room. It is an 8 m × 6 m
room with two partition walls, which make a "reading nook" in the north-west and a
"storage" bay in the south-east, and a coloured 0.30 m cube in each corner.
`tools/generate_world.py` turns it into **both** the Gazebo world (`worlds/nlp_room.sdf`)
and the matching Nav2 map (`maps/nlp_room.pgm` + `.yaml`). The simulation and the map
therefore always agree, and no SLAM mapping step is needed. The boxes are left out of
the map, so you can move them in Gazebo and the robot has to find them with its sensors.

```
 y=+3 ┌────────┬──────────────────────────┐
      │ blue ■ │                    ■ red │
      │ (nook) │                          │
      │        │                          │
      │                 ● home (0,0)      │     x: -4 … +4 m
      │                       │           │     y: -3 … +3 m
      │                       │ (storage) │
      │ ■ yellow              │   green ■ │
 y=-3 └───────────────────────┴───────────┘
     x=-4    x=-1.5          x=+1.5     x=+4
```

To change the room, edit `world_spec.py`, run `python3 tools/generate_world.py`, and rebuild.

## Build and test

```bash
cd ~/nlp_robot_ws
colcon build --symlink-install
source install/setup.bash
ros2 run nlp_robot check_interfaces                       # stage 1
cd src/nlp_robot && python3 -m pytest test/ && cd ../..   # unit tests (no ROS needed)
```

## Run the simulation (stage 2)

```bash
# Terminal 1
export ROS_DOMAIN_ID=42
cd ~/nlp_robot_ws && source install/setup.bash
bash src/nlp_robot/tools/clean.sh          # stop leftovers from other projects
bash src/nlp_robot/tools/check_setup.sh    # pre-flight check
ros2 launch nlp_robot sim.launch.py

# Terminal 2: drive it (w/a/s/d/x, space = stop)
export ROS_DOMAIN_ID=42 TURTLEBOT3_MODEL=waffle
cd ~/nlp_robot_ws && source install/setup.bash
ros2 run turtlebot3_teleop teleop_keyboard
```

Use the same `ROS_DOMAIN_ID` in every terminal. It keeps this project's topics separate
from anything else running on the PC.

## Navigate (stage 3)

```bash
# Terminal 1: Gazebo + robot + Nav2 + RViz
export ROS_DOMAIN_ID=42
cd ~/nlp_robot_ws && source install/setup.bash
bash src/nlp_robot/tools/clean.sh
ros2 launch nlp_robot bringup.launch.py          # gui:=false hides the Gazebo window

# Terminal 2: send it to a named place
export ROS_DOMAIN_ID=42 && cd ~/nlp_robot_ws && source install/setup.bash
ros2 run nlp_robot goto --list
ros2 run nlp_robot goto kitchen                  # Ctrl+C cancels the goal
ros2 run nlp_robot goto "the dock"               # aliases work
```

Or click **2D Goal Pose** in RViz and then click and drag on the map.

How it fits together:
- **Map server** loads `maps/nlp_room.yaml`.
- **AMCL** works out where the robot is on that map by matching the LiDAR scan against
  the walls. It starts at the spawn pose, so no "2D Pose Estimate" click is needed.
- The **planner** finds a path across the global costmap, which is the map with walls
  inflated by the robot's radius.
- The **controller** follows that path while avoiding anything the LiDAR sees.
- The **behaviour server** provides `spin`, `backup` and `drive_on_heading`. The
  stage-5 commander uses these for "turn around" and "back up 30 cm".

Named places live in `config/places.yaml`, with an `[x, y, yaw]` pose and aliases for
each. `test/test_places.py` checks that each place is inside the room and clear of the
walls and boxes.

## The language layer (stage 4)

This is the generative-AI part. A sentence goes in and a **validated plan** comes out. The
LLM never touches motor commands.

```
"go to the red box,     ┌─────────────────────┐   JSON    ┌──────────────────────┐  valid plan
 then come back"   ───► │ prompt + LLM        │ ───────►  │ plan_schema.validate │ ──────────►  robot
                        │ (structured output) │ ◄───────  │ (8 actions, limits,  │  (stage 5)
                        └─────────────────────┘  errors,  │  targets, signs)     │
                          ▲ grounding: places,   one retry └──────────────────────┘
                          │ seen objects, robot pose, last 3 commands
```

| File | Role |
|---|---|
| `nlp_robot/plan_schema.py` | The contract: 8 actions, a JSON schema, target resolution ("the dock" → `home`, "it" → the previous step's target), range limits, and the direction check |
| `nlp_robot/llm_backends.py` | `ollama` (local), `anthropic`, `openai` (any compatible server) and `rules` (regex, no LLM). Uses only `requests`, with no vendor SDKs |
| `nlp_robot/command_parser.py` | The system prompt, the grounding message, validation, one repair round, and the fallback backend |
| `tools/plan_shell.py` | Type sentences and see the plan (`--show-prompt` shows what the LLM sees) |
| `tools/eval_parser.py` | A 26-command benchmark of any backend, written to `results/*.csv` |

**Actions:** `navigate_to(place | <colour>_box)`, `find(<colour>_box)`, `go_to_pose(x, y, yaw)`,
`return_to_start`, `rotate(angle_deg)`, `move(distance_m)`, `wait(seconds)`, `say(text)`.

### Why the robot can trust it

1. **A closed set of actions.** The LLM can only choose from 8 actions. It cannot output
   velocities or code.
2. **Structured output.** The JSON schema is given to the model itself (Ollama `format`,
   Anthropic forced tool call, OpenAI `json_schema`), so the reply is valid JSON in the
   right shape.
3. **Validation before anything moves.** Plans are rejected if they have unknown places or
   boxes, points outside the map, distances over 3 m, more than 12 steps, or a turn or move
   whose sign contradicts the user's own words ("turn right" given as +45°).
4. **Bounded self-repair.** The validation errors go back to the LLM once. If the second
   answer also fails, nothing moves.
5. **Clarify instead of guessing.** An empty plan with a question ("Where is the garage?")
   is a valid answer.
6. **Grounding.** The prompt lists only the places that exist and the objects actually seen,
   so the model has no reason to invent coordinates.
7. **Graceful degradation.** If the LLM is down or times out, the `rules` parser still
   handles simple commands.

### Try it (no ROS needed)

```bash
cd ~/nlp_robot_ws/src/nlp_robot
python3 tools/plan_shell.py                          # rules
python3 tools/plan_shell.py --backend ollama         # local LLM
python3 tools/plan_shell.py --backend ollama --show-prompt
```

### Evaluate

Run this with the simulation **closed**, so the timings aren't affected by Gazebo:

```bash
cd ~/nlp_robot_ws/src/nlp_robot
python3 tools/eval_parser.py --backend rules
python3 tools/eval_parser.py --backend ollama --model qwen2.5:7b-instruct --label v3 \
    --compare ../../results/eval_ollama_qwen2.5-7b-instruct_v2.csv
python3 tools/eval_parser.py --backend ollama --model qwen2.5:3b-instruct --label v3
GEMINI_API_KEY=... python3 tools/eval_parser.py --backend gemini --model gemini-3.5-flash-lite --label v3 --delay 6
python3 tools/summarize_results.py            # table + figures from every run in results/
```

The v1 result, from an earlier version of the project, is kept as the baseline in
`results/eval_ollama_qwen2.5-7b-instruct_v1.csv`.

| Backend | Plans it can carry out | Correct refusals | Median latency |
|---|---|---|---|
| `rules` (regex baseline) | 9/21 | 5/5 | 0 s |
| `qwen2.5:7b-instruct`, v1 prompt (earlier version) | 16/20 | 4/5 | 15.3 s |
| `qwen2.5:7b-instruct`, v2 (CPU) | 19/21 | 4/5 | 5.7 s |
| `qwen2.5:7b-instruct`, v3 (CPU) | **21/21** | **5/5** | 5.1 s |
| `qwen2.5:7b-instruct`, v3 (53 % GPU) | **21/21** | **5/5** | 2.9 s |
| `qwen2.5:3b-instruct`, v3 (100 % GPU) | 17/21 | 3/5 | **0.5 s** |
| `gemini-3.5-flash-lite`, v3 (cloud) | **21/21** | **5/5** | 1.4 s |
| `gemini-3.8-flash`, v3 (cloud) | 17/17 answered; 9 hit the free-tier quota | — | 6.0 s |

Hardware: Intel Alder Lake laptop, RTX 3050 Laptop GPU (4 GB). The 7B model (5.2 GB) does
not fit in 4 GB, so Ollama splits it between GPU and CPU; the 3B (1.9 GB) fits entirely.

### Model size: what the 3B gets wrong

The 3B is 6× faster but makes mistakes the validator **cannot** catch, because the plans
are valid, just not what was asked:

| Command | 3B plan | Kind of error |
|---|---|---|
| `bring me a coffee` | `navigate_to kitchen` ("to get a coffee") | substitutes a possible action for an impossible one |
| `go to the purple box` | `navigate_to home` ("let's go home instead") | substitutes instead of asking |
| `go to coordinates x 1.5, y -0.5` | `move 1.5; move -0.5` | wrong action for the meaning |
| `turn around and go back 30 cm` | `rotate -90; move -0.3` | repair fixed the move but broke the turn |
| `find the green box and go to it` / `visit … then the green box` | `find green_box` only | drops "go to it" |

**Takeaway for the report:** validation guarantees a plan is *executable and inside the
safety limits*; it cannot guarantee the plan *means what the user said*. That second
property comes from model capability. The 7B gets both right at 2.9 s, so it is the
default; the 3B is the option when latency matters more than reliability.

The 3B run also exposed a UI bug: after a repair the plan was `rotate -45` but the model's
reply still said "Turning left 45 degrees". Since then the robot's reply is always
generated from the validated plan (`summarize_plan`); the model's own wording is kept
only in the CSV's `llm_reply` column.

The v1 failures led to the v2 changes:

| v1 failure | Cause | v2 change |
|---|---|---|
| `turn right by 45 degrees` → +45, which turns **left** | sign convention | direction check plus repair; sign rule and example in the prompt |
| `take a look in the nook` → added `find red_box` | invented a goal | rule: only search for objects the user names |
| `return to the kitchen` → `return_to_start` + `navigate_to` | "return" taken literally | rule: "return to <place>" means `navigate_to` |
| `fly to the ceiling` → a spin | substituted an action instead of refusing | rule: refuse, and never substitute |
| `where is the yellow cube? go there` → `find` + `navigate_to` | grading, not the model | the evaluation now treats these as equivalent |

The v2 failures (23/26) led to the v3 changes:

| v2 failure | Cause | v3 change |
|---|---|---|
| `drive 50 metres forward` → `move 2.0` | the repair message gave the limit, so the model **shortened** the move to fit it | the error now says "do not shorten, refuse"; a new check compares the distances in the command with the limit |
| `return to the kitchen` → still added `return_to_start` | the prompt rule alone wasn't enough | a new check rejects `return_to_start` when the command says "return to <place>" |
| `find the blue box` → `find` + `navigate_to` | did more than it was asked | prompt rule and example: `find` alone means search only |

A lesson worth putting in the report: the wording of the error that goes back to the model
matters as much as the prompt. An error that only states a limit invites the model to
fit inside it.

## The commander (stage 5)

`nlp_robot/commander_node.py` is the action server `/execute_command`. For each command it
asks the language layer for a validated plan, then runs the steps one at a time as Nav2
actions, publishing feedback as it goes:

| Plan step | Nav2 action |
|---|---|
| `navigate_to <place>`, `go_to_pose`, `return_to_start` | `NavigateToPose` |
| `rotate` | `Spin` (behaviour server) |
| `move` forward / backward | `DriveOnHeading` / `BackUp` |
| `navigate_to <box>`, `find <box>` | spin and visit the lookout places until the detector sees it, then `NavigateToPose` 0.6 m short of it (from stage 7) |
| `wait`, `say` | handled in the commander |

Design points:
- **Only one command at a time.** A second goal is rejected while one is running, instead of
  two plans fighting over the robot.
- **Cancelling cascades.** Cancelling `/execute_command` cancels whichever Nav2 action is
  running. A goal Nav2 accepts just after the cancel is cancelled too.
- **"Come back" means the start of the last movement command.** The commander records the
  pose when a command that moves begins.
- **Warm start.** At launch the commander loads the model and sends one throwaway command,
  so the first real command doesn't pay for the model load and the first prompt.
- **Results are recorded.** Every result carries `plan_json`, `backend` and
  `parse_time_s`, so each live run doubles as data for the evaluation.

```bash
# Terminal 1
ros2 launch nlp_robot bringup.launch.py                      # Ollama, qwen2.5:7b-instruct
ros2 launch nlp_robot bringup.launch.py llm_backend:=rules   # no LLM

# Terminal 2 (the stage-6 CLI makes this nicer)
ros2 action send_goal /execute_command nlp_robot_interfaces/action/ExecuteCommand \
  "{command: 'go to the kitchen, then come back'}" --feedback
```

## Talk to it (stage 6)

```bash
# Terminal 1
ros2 launch nlp_robot bringup.launch.py            # gui:=false makes the LLM faster (less CPU contention)

# Terminal 2
export ROS_DOMAIN_ID=42 && cd ~/nlp_robot_ws && source install/setup.bash
ros2 run nlp_robot nlp_cli
```

```
you> go to the kitchen, then come back
  ... understanding the command
robot> I'll go to the kitchen, then come back.
  > [1/2] navigate to kitchen
robot> Reached the kitchen.
  > [2/2] return to start
robot> Back where I started.
DONE - Reached the kitchen. Back where I started. (2 steps, 58.8 s)
  plan: [{"action": "navigate_to", "target": "kitchen"}, {"action": "return_to_start"}]
  [ollama:qwen2.5:7b-instruct, planned in 3.1 s]

you> turn around and back up 30 cm
you> do that again                 # uses the recent-command history
you> bring me a coffee            # refused politely; nothing moves
you> dry go to the office         # show the plan without moving
```

- **Ctrl+C while a command runs** stops the robot. It cancels the command and the Nav2 goal,
  and you get the prompt back. Ctrl+C at the prompt, or `quit`, exits.
- **One-shot mode:** `ros2 run nlp_robot nlp_cli "go home"`, which exits with 0 on success
  and 1 on failure, so it can be used in scripts.
- If planning takes more than 5 s, the CLI prints `... still thinking (N s)`, so a slow
  model doesn't look like a hang.

## Finding "the red box" (stage 7)

`nlp_robot/color_detector_node.py`, with the pure-Python core in `color_detection.py`,
publishes `/detected_objects` (`DetectedObjectArray`, in the map frame) at 2 Hz:

1. **Colour → direction (camera).** The image is shrunk to 480 px and thresholded in HSV,
   one range per colour, and the largest blob of each colour is kept. Its left and right
   edges give two rays, transformed into the LiDAR frame through TF.
2. **Distance, from the LiDAR when it can see that far.** The LiDAR returns between those two
   rays are the box's front face. Pushing back by half the box size gives its centre.
3. **Distance, from the camera beyond 3.5 m.** The TurtleBot3 LiDAR stops at 3.5 m, and every
   box is about 4.2 m from `home`. Further out, the distance is estimated from the box's
   known height: distance ≈ focal length (px) × 0.30 m / blob height (px). Height is used
   rather than width because a cube's visible width changes as it turns. The estimate is
   coarser, so it is flagged `range_source: camera` and the RViz detector view marks it `cam`.
4. **Tracking.** Each sighting is converted to map coordinates and averaged with the previous
   ones. A box is only *confirmed* after 3 consistent sightings, so one noisy frame never
   sends the robot anywhere. If a box is moved, 3 sightings at the new place move the
   estimate there.

**The commander uses it like this:**
- **`find X`** spins 360°, then visits the `lookout` places nearest-first, spinning at each,
  and stops the moment X is confirmed.
- **`navigate_to X`** first searches if needed. If the box is more than 3 m away (probably a
  camera estimate), the robot drives in two stages: first to 1.5 m short of the box, where
  the LiDAR takes over the distance, then to 0.6 m short, facing the box.

**Ground truth (evaluation only).** Gazebo publishes each box's true pose. The
`ground_truth` node draws it in RViz as a wire-frame cube labelled with the detector's
error, and logs `red_box: true (…), detected (…), error 0.12 m`. The robot never reads it.

In an earlier version of the project, adding the camera-based distance cut "go to the red box" from home
from **132 s to 19 s**. Set `camera_range_fallback: false` in `config/nlp_robot.yaml` to
reproduce the slower search as an ablation.

**Measured in simulation** (true position from Gazebo against the detector's estimate):

| Box | Position error | Notes |
|---|---|---|
| yellow | 0.03 m | |
| red | 0.08 m | |
| blue | 0.09 m | after being **moved by hand** next to the red box; the tracker picked up the new position |
| green | 0.24 m | in the storage bay, seen at an angle past the partition |

`find the green box` from home took 118 s. The green box is behind the storage partition,
so it can't be seen from home, the kitchen or the nook. Searching is a nearest-first tour of
the lookout places with no prior about where the box might be. Once a box has been seen,
later commands use the remembered position and need no search.

```
you> find the green box
you> go to the red box, then come back
you> find the yellow box and go to it
you> is there a blue box in the reading nook?
```

## Evaluation write-up (stage 8)

[`docs/RESULTS.md`](docs/RESULTS.md) is the results section for the report. It covers the
setup, the language layer from regex to v3, model size and hardware, end-to-end runs,
detection accuracy, the safety design, limitations and future work.

```bash
cd ~/nlp_robot_ws/src/nlp_robot
python3 tools/summarize_results.py     # rebuilds the table + results/figures/*.png from results/*.csv
```
Figures need matplotlib: `sudo apt install python3-matplotlib`.
