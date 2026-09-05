# Chassis Lab

A working Python/MuJoCo experiment for a **VEX-inspired four-wheel differential-drive chassis**. It validates a small JSON design, derives component masses and inertias, generates MJCF, physically drives an ordered course, saves metrics, and compares an initial design with one revision. AI mode uses OpenAI Responses structured output. Offline mode uses explicitly handwritten samples through the identical simulation and reporting pipeline.

The original chassis commands use a synthetic driving benchmark. The new mechanism commands below run **simulation benchmarks based on official Override field resources**, with an arm, claw or roller acquisition tool. Neither subsystem establishes official legality, calibrated hardware performance, or manufacturing readiness. No RL, hidden chassis forces, web server, GPU or database is required.

## Quick standing chassis assembly (no AI)

For the simplest parts-and-assembly test:

```sh
/private/tmp/chassis-lab-venv/bin/python -m chassis_lab assembly
# Live desktop viewer on macOS; leave open to inspect, close window to exit:
/private/tmp/chassis-lab-venv/bin/mjpython -m chassis_lab assembly --viewer
```

This assembles a fixed V5-style chassis with six perforated C-channels, four
4-inch wheels, four 1/8-inch square axles, eight bearing flats, eight shaft
collars, visible #8-32 screws/nuts, eight 1.5-inch standoffs, four motor cases,
and mounted battery/brain envelopes. Crossmembers are raised on standoffs to
clear the wheels. Each axle has two parallel bearing supports. Named parts
and connection relationships are saved in `assemblies/standard_v5/assembly.json`;
the complete MuJoCo model is `chassis.xml` in that directory. `chassis.png`,
`top.png`, and `axles.png` show the assembly when rendering is available.

The command actually simulates three seconds of gravity with no drive commands
and saves a four-wheel contact/stability check in `standing_check.json`.
`--no-render` works without graphics; the live viewer uses ongoing physics.
No API key or AI operation is involved. This command intentionally overwrites
its fixed output folder for quick iteration; use `--output` to keep a variant.

This is a **nominal part-level representation**, not imported manufacturer CAD.
Shaft size, bearing support and fastener conventions follow VEX's
[shaft guide](https://kb.vex.com/hc/en-us/articles/360035591372-Using-V5-Shafts),
[chassis guide](https://kb.vex.com/hc/en-us/articles/360035953131-Designing-a-V5-Chassis),
and [fastener guide](https://kb.vex.com/hc/en-us/articles/360035952791-Using-V5-Fasteners).
Case dimensions, masses and hole clearances are simplified. Bolted connections
are rigid child bodies, while axles use hinge joints. Screw threads, bearing
contact clearances and fastener stresses are not individually simulated.

## Install and run

Python 3.11+ is required. From this project directory:

```sh
python3 -m venv /private/tmp/chassis-lab-venv
source /private/tmp/chassis-lab-venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m chassis_lab demo --offline
python -m pytest -q
```

The external environment path is intentional: this workspace name contains `:`, which Python refuses in a virtual-environment path. On Linux use `/tmp/chassis-lab-venv`; on Windows use a normal path such as `.venv`. An environment in `/private/tmp` is temporary and may need recreation after cleanup. A durable environment outside this workspace is also fine. The lock records the exact tested dependencies; for other Python/platform combinations, `python -m pip install -e '.[test]'` resolves compatible versions from `pyproject.toml` instead.

The exact command using the environment created during development is:

```sh
/private/tmp/chassis-lab-venv/bin/python -m chassis_lab demo --offline
```

Each run prints an absolute path to `report.html`. Open that file in a browser. It contains aggregate comparisons, model/design links, MuJoCo screenshots when graphics are available, and an elapsed-time slider showing both recorded trajectories. No server is required.

AI mode (with an authorized, working `OPENAI_API_KEY` in the process environment or an ignored `.env.local`/`.env` file):

```sh
python -m chassis_lab experiment --ai --generations 2
python -m chassis_lab experiment --ai --generations 2 --model gpt-5.6-sol
```

Do not paste keys into source, commands, reports, or chat. The program never prints keys and never includes environment contents or arbitrary HTTP error text in records. It reads local dotenv files only in AI mode, without overriding existing process variables.

Configuration:

| Setting | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | none | API authentication; unnecessary for offline mode |
| `CHASSIS_LAB_MODEL` | `gpt-5.6-sol` | Model selection; `--model` overrides it |
| `--output` | `runs` | Parent directory for unique experiment records |
| `--no-render` | false | Skip screenshot subprocesses, retaining physics and HTML playback |
| `--generations` | 2 | Only 2 is accepted: initial proposal plus one revision |

## Replay

```sh
python -m chassis_lab replay <run-directory> --candidate 0
python -m chassis_lab replay <run-directory> --candidate 1 --seed 211
python -m chassis_lab replay <run-directory> --headless
```

On macOS the native viewer requires MuJoCo's launcher:

```sh
/private/tmp/chassis-lab-venv/bin/mjpython -m chassis_lab replay runs/20260905T171537Z_offline_4b27bd --candidate 1
```

The viewer plays saved generalized positions at their recorded times, supports MuJoCo camera controls, and exits at the end or when closed. It does not claim replay is a fresh physics trial. Physics evaluation only sets the root pose once at initialization, then calls `mj_step` with four wheel motor torques. `--headless` verifies recorded-state playback without opening a window. Select development trials with `--split development --seed 11`.

## Design schema and components

`chassis_lab/schema.py` defines the Pydantic design, proposal, and trial models. Machine-readable `design.schema.json` and `proposal.schema.json` are exported in every run; cross-field geometry checks are implemented in the Pydantic validator because JSON Schema bounds alone do not express them.

All lengths are meters. Robot-local axes are +x forward, +y left, +z up. The origin is at frame center and axle height. Only these design values are variable:

| Field | Range / options | Definition |
|---|---|---|
| `frame_width_m` | 0.18–0.32 | Left/right outer frame edges, excludes tires |
| `frame_length_m` | 0.25–0.42 | Front/rear outer frame edges |
| `wheelbase_m` | 0.14–0.32 | Front/rear wheel-center separation |
| `track_width_m` | 0.23–0.40 | Left/right wheel-center separation |
| `wheel_diameter_m` | 0.08255, 0.1016, 0.104775 | Nominal 3.25, 4, 4.125 inches |
| `gearing_rpm` | 100, 200, 600 | Approximate unloaded geared motor output speed |
| `battery` | `{x_m, y_m}` | Center of fixed battery box on frame |
| `ballast` | null or `{x_m, y_m}` | Omit ballast, or position a fixed 0.30 kg block |

Example handwritten baseline:

```json
{
  "frame_width_m": 0.25,
  "frame_length_m": 0.34,
  "wheelbase_m": 0.27,
  "track_width_m": 0.30,
  "wheel_diameter_m": 0.1016,
  "gearing_rpm": 200,
  "battery": {"x_m": 0.0, "y_m": 0.0},
  "ballast": null
}
```

Validation requires track ≥ frame width + 0.035 (25 mm tire width plus 5 mm clearance on each side), wheelbase ≥ diameter + 0.020, and wheelbase ≤ frame length − 0.040. Overall footprint is `max(frame_length, wheelbase + diameter)` by `track + 0.025`, each at most 0.4572 m. Vertical size is derived, never selected: the fixed frame/component layout stays below 0.12 m nominal height for all wheel options, well inside the same 0.4572 m engineering envelope. Battery and ballast fit inside frame edges with 5 mm margins and do not overlap; a 5 mm separation is required. Extra fields and nonfinite numeric values are rejected.

Every chassis has four motors, one per wheel, with no gearing stages outside the selected motor cartridge approximation. Fixed component masses are four 0.28 kg motor surrogates, four 0.12 kg solid-cylinder tire surrogates, and a 0.35 kg battery. Frame/support/electronics mass is `0.65 + 2*(length+width)` kg, distributed as a 25 mm thick box. Optional ballast adds 0.30 kg. MuJoCo computes COM and inertia from these component primitives, including their positions. Neither total mass, inertia, speed, torque nor traction is freely chosen by the AI. Motor and frame boxes are inertia surrogates, not detailed physical part volumes.

All these masses, dimensions and motor curves are **engineering assumptions**, not verified VEX specifications. The nominal diameter and rpm options are VEX-inspired. Hardware calibration is future work.

## Physics and controller

`physics.py` builds a free root body, four independently rotating cylinder tires on parallel lateral hinge axes, collision geometry, a ground plane, obstacles and enclosing walls. Battery, frame, motors and optional ballast are fixed mass contributions to the root. MuJoCo uses gravity 9.81 m/s², `implicitfast`, 2 ms steps, 60 solver iterations and elliptic friction cones. Wheel joints have damping 0.001 N m s/rad and armature 0.0001 kg m².

The isotropic Coulomb tire contact (`condim=3`) opposes both longitudinal and lateral slip. Turning therefore creates physical four-wheel scrub. It is intentionally a conventional skid-steer architecture, not an omni-wheel chassis. No extra steering force, kinematic root control, waypoint teleportation, or hidden traction multiplier is applied.

For each motor, `omega0 = rpm*2*pi/60` and `stall = 1.05*200/rpm` N m. The controller requests angular wheel velocities from linear/yaw commands using each design's radius and track. A voltage-like speed servo applies:

```text
u = clip(omega_target/omega0 + 0.8*(omega_target - omega), -1, 1)
torque = clip(stall * (u - omega/omega0), -stall, stall)
```

The back-EMF term reduces torque at speed; the symmetric current cap bounds torque including braking. No battery sag, current sharing, motor heat, gearbox efficiency map or V5 firmware behavior is modeled. High feedback gain and simplified rigid contacts can cause wheel-speed chatter, saturation and variable stopping time; those affect measured performance and are reported rather than hidden.

All designs use the same deterministic SI-unit waypoint controller: maximum requested forward speed 0.85 m/s, yaw rate 1.8 rad/s; forward motion is suppressed as heading error approaches 0.4 rad. Requested wheel speed is scaled to the selected free speed. The controller was developed on the handwritten baseline using development seeds, then frozen before AI experiments. There is no candidate-specific tuning budget or hidden optimization of the controller.

## Benchmark, metrics and ranking

The enclosure has interior limits x = −0.75…6.45 m and y = −1.75…1.75 m. Initial pose is near (0,0), heading 0. Ordered targets are:

1. (2,0): straight acceleration followed by a required braking stop.
2. (2.65,0.60), (3.45,−0.60), (4.25,0.60): slalom.
3. (5.1,0), then (5.65,0): approach and parking with heading 0 rad.

Three 0.14 m radius obstacles stand at (2.65,−0.30), (3.45,0.30), (4.25,−0.30). Waypoints must be reached in order; intermediate tolerance is 0.11 m. The straight target requires proximity under 0.09 m and speed under 0.045 m/s for 0.35 s. Final completion requires parking position error under 0.085 m, heading error under 0.10 rad, linear speed under 0.045 m/s and yaw rate under 0.10 rad/s continuously for 0.5 s. Time limit is 45 s.

Seeds 11, 23, 37 are development trials. Seeds 101, 211, 307 are held out from AI feedback and used after both proposals are fixed. Each seed draws independent initial x/y offsets uniformly ±0.025 m, heading ±0.04 rad, then friction uniformly 0.55–0.80. Every design receives the same paired conditions. Final seeds are reserved from tuning; they are a small fixed test set, not a guarantee of generalization.

Trial JSON records completion and time, sequential waypoint progress, collisions, time-sampled RMS distance to the current path segment, parking errors, instability/failure summaries, per-wheel absolute travel, voltage saturation fraction, root distance traveled, peak speed and braking stop duration. A new collision episode with a given wall/post requires more than 0.25 s since its last contact; continuous or brief chattering contact counts once. Floor contacts are excluded. Tilt over 35°, root height over 0.30 m, excessive velocity, nonfinite values or numerical solver warnings terminate the trial as unstable.

The aggregate ranking lexicographically minimizes:

```text
[-clean completions, -completions, unstable runs, collision episodes,
 -mean ordered progress, mean completion time (45 s for failures), mean RMS error]
```

A stopped or incomplete robot cannot outrank a completed course. Progress is capped below 1 until all stages and parking checks pass. Comparison treats time differences ≤0.10 s and progress/RMS differences ≤0.001 as ties before comparing the next metric. Reports say improvement, regression or inconclusive; this is a descriptive three-trial comparison, not statistical significance. Tracking and speed are lower priority than successful clean completion.

## AI integration and limits

The implementation was checked against the current [OpenAI structured output guide](https://developers.openai.com/api/docs/guides/structured-outputs) and [gpt-5.6-sol model page](https://developers.openai.com/api/docs/models/gpt-5.6-sol), and the installed SDK signature. It calls `client.responses.parse(..., text_format=WireProposal)` and consumes `output_parsed`. Cross-field validation follows transport parsing so token usage can be preserved when geometry needs repair.

The AI receives allowed fields, engineering assumptions, the benchmark, and for the revision the original proposal plus compact numerical development results/failures. Both proposals must include a rationale, predicted benefit and testable hypothesis. No image input is needed. The model receives no tools and cannot change code, scoring, seeds or assumptions. Proposals are data; XML is generated only from validated parameters.

Limits: exactly two proposals; at most one invalid-output repair for each; at most four API calls total per experiment; 2,500 output tokens per call; low reasoning effort; 60 s SDK timeout; SDK automatic retries disabled; `store=False`. Authentication/transport failures stop the AI run immediately. Validation failures and refusals are recorded. No fixture fallback masquerades as AI output. Available response IDs, model, token usage, input prompts and proposal text are saved. If the API fails before usage is returned, usage is unavailable, not zero.

## Experiment files

Each uniquely named run contains:

```text
manifest.json             seeds, settings, versions, source hashes, status, rendering status
assumptions.json           complete modeling assumptions
design.schema.json        machine-readable field schema
proposal.schema.json      proposal schema
source/                   executed Python source snapshot
ai_calls.json             AI runs only: bounded attempts, prompts, outputs, usage, safe errors
candidates.json           proposals, parent relationships and results
comparison.json           aggregate results and verdict
report.html               self-contained comparison and canvas playback
candidate_0/, candidate_1/
  design.json, proposal.json, model.xml, results.json
  development_11.json/xml/npz ...
  final_101.json/xml/npz ...
  start.png, slalom.png, finish.png, robot.png  (when rendering succeeds)
```

Each trial's XML includes its actual friction; the seed and initial pose are recorded in metrics. NPZ stores 50 Hz generalized-position and time histories for replay. `model.xml` uses nominal friction 0.7. JSON uses finite numbers only. Source snapshots and dependency versions make the experiment auditable; native simulation is not promised bit-identical across CPUs or MuJoCo releases.

## Verification and actual results

The baseline first passed separate straight-drive and turning checks using wheel torques. The focused suite then passed **15 tests**, covering invalid geometry, dimensions/mass/COM/inertia, finite dynamics, actual translation and turning versus an unactuated control, motor bounds/back EMF, full-course completion, anti-shortcut ranking, repeatability, and bounded AI repairs with preserved usage. Same-process repeated seeded trajectories agree within 1e-10 m/rad. Across platforms, compare completion/collision outcomes and allow roughly 0.1 s / 1 mm metric differences; discontinuous contact/threshold changes may require investigation instead of assuming that tolerance always holds.

The complete offline run is `runs/20260905T171537Z_offline_4b27bd`. Its held-out results are:

| Handwritten sample | Clean finishes | Mean completion | Mean RMS tracking | Collisions | Unstable runs |
|---|---:|---:|---:|---:|---:|
| Initial, 0.27 m wheelbase | 3/3 | 36.239 s | 0.06073 m | 0 | 0 |
| Revision, 0.20 m wheelbase | 3/3 | 28.331 s | 0.05172 m | 0 | 0 |

The sample revision is **21.82% faster** on these three trials. It changes only wheelbase, with the same 3.780 kg mass, frame, battery and gearing. This is an offline fixture comparison, not evidence of AI design quality. Per-seed final times are initial 35.100 / 36.702 / 36.914 s and revision 28.408 / 27.952 / 28.634 s. Development also finished 3/3 clean for each sample.

An authorized live experiment was attempted. The initial sandbox attempt recorded `APIConnectionError`; the network-enabled retry recorded HTTP 401 `AuthenticationError` before any proposal or usage was returned. Those runs remain separately labeled failed/inconclusive. The Responses integration has been implemented and locally tested, but the end-to-end AI proposal/revision path is **not verified** with this rejected credential. No AI results are fabricated.

MuJoCo generated representative screenshots for both samples after macOS graphics access was granted. Headless playback verified 1,756 saved baseline frames. The native `mjpython` replay command launched and exited successfully; automated visual inspection of that window was unavailable because Accessibility/Screen Recording permissions were pending. The local HTML report was generated and checked structurally; automated browser inspection was blocked by the browser's local-file URL policy.

## Troubleshooting and limitations

- **401 AuthenticationError:** the configured key did not authenticate. Use secure API key setup to configure a working authorized credential; do not print the rejected key. A successful connection is necessary before model access can be verified.
- **APIConnectionError:** check network access. In a sandbox, permit the actual experiment's outbound API request. Transport failures do not establish invalid credentials or quota exhaustion.
- **Other API failures:** `ai_calls.json` records exception class and HTTP status without raw error text. An unavailable model is not automatically substituted; set `--model` explicitly if needed.
- **macOS `invalid CoreGraphics connection`:** graphics access was blocked in the sandbox during development. Rendering succeeded with desktop graphics access. Use `--no-render` for pure headless runs. Physics and HTML canvas playback still work.
- **Viewer cannot launch:** use `mjpython` on macOS; use `--headless` for recorded-state verification. On Linux a display or supported offscreen OpenGL backend is needed for screenshots, but not physics. Rendering is isolated in a 30-second subprocess so a graphics failure does not invalidate a completed experiment.
- **Poor candidate performance:** inspect ordered progress, saturation, wheel travel, collisions and parking error. The fixed controller is part of this experiment; results do not measure a chassis's best achievable performance under an optimized controller.

This version omits structural flexibility, detailed tire deformation, real motor electrical/thermal behavior, manufacturing detail, gear backlash, uneven terrain and battery discharge. Frame and component primitive inertia approximations are crude. The original chassis-only experiment searches one architecture with one revision. The mechanism search supports multiple generations and structural edits; both small benchmark sets invite overfitting. Physical validation and independent benchmarks are required before using results for real hardware decisions.

## Override mechanism search (new subsystem)

The image-inspired seed has a wheel-driven chassis, powered pivoting arm,
counter-rotating wrist and two powered claw fingers. A generic edit can replace
those fingers with powered rollers and add a passive guide. These are generated
from a versioned component graph, not selected from complete robot templates.
The existing chassis benchmark and standing assembly commands remain available.

Use the Python environment installed earlier, or follow the installation section
above. The following commands need no API access:

```sh
python -m chassis_lab field
python -m chassis_lab seed-robot
python -m chassis_lab mechanism-benchmark --seed 41
python -m chassis_lab search --offline --generations 3 --candidates 3 --trials 2 --tuning 2 --wall-seconds 600 --no-render
python -m chassis_lab mechanism-benchmark --design <run>/g1_c1/design.json --task cup
python -m chassis_lab compare <run>
python -m chassis_lab resume <run> --no-render
python -m chassis_lab mechanism-replay <run>/g1_c1/development_41.xml --headless
```

`--offline` uses explicitly **handwritten development mutations**, never AI.
Each generation selects the measured champion, a distinct mechanism family when
available, and the original seed; later fixture mutations use those selected
parents. The fixture policy is intentionally simple and is not an AI substitute.

For live structured-output proposals, using an already configured credential:

```sh
python -m chassis_lab search --ai --model gpt-5.6-sol --reasoning low --generations 3 --candidates 3 --trials 2 --api-calls 12 --output-tokens 3000 --wall-seconds 600
```

`OPENAI_API_KEY` is read from the environment, then `.env.local` / `.env` without
overriding an existing environment value. Do not put credentials in design JSON
or command arguments. `CHASSIS_LAB_MODEL` sets the default model. The existing
configured credential returned **HTTP 401 AuthenticationError** on the actual
search run. Consequently no AI-generated mechanism candidates or token usage
were available, and multi-generation live AI behavior remains unverified.
There is no silent offline fallback inside an AI experiment.

The integration uses OpenAI Python `client.responses.parse(...,
text_format=Proposal)` with Pydantic structured output, configurable reasoning,
`store=False`, a 45-second request timeout, no SDK retries, one repair attempt
per proposal and a global call budget. Numerical failure summaries and up to
two rendered failure images accompany parent specifications. Candidate output
is data only: no generated Python, evaluator edits, seed changes, or tools.
An unsupported generator request is quarantined as a validation error for
separate implementation and review; the current search cannot automatically
admit arbitrary new generators. See the official
[structured outputs guide](https://platform.openai.com/docs/guides/structured-outputs).

### Resources and reconstruction

The authoritative pinned resource is **Override Game Manual v2.0, September 3,
2026**, obtained through the
[official current-game page](https://www.vexrobotics.com/v5/competition/vrc-current-game)
and downloaded from the
[official PDF](https://content.vexrobotics.com/docs/2026-2027/override/files/override-2.0.pdf).
`resources/override_v2/manifest.json` records URLs, download time and SHA-256;
`traceability.json` maps geometry/rules to PDF pages. A copy is included in every
search archive. The official CAD and assembly download endpoints returned 403;
manufacturer field CAD and complete assembly instructions were **not imported**.
`tools/fetch_override.py` is the resource acquisition script, not part of search.

The Pin is 165 mm tall with 35.6 mm tips, 59.6 mm taper bases and an 80.3 mm
circumscribed flange (PDF p100/A5). Its rounded triangular profile is simplified
to circular tapers/flange. The Cup is 164.5 mm tall, with 80.2 mm outer rims and
59 mm neck (p101/A6); a 2 mm wall and central divider are assumptions. Cups and
Goals use separate convex wall sectors so their interiors are physically hollow.
The Goal opening is 60.1 mm, with 82.5/146.5/222.7 mm heights (p102/A7). Goal
outer details and the straight internal bore are simplified.

The inspection scene includes nine Goals at dimensioned coordinates, perimeter,
four revolute Toggles, and representative loose Pins/Cups. Toggle span is
660.2 mm (p103/A8); pivot, damping, stops and inertia are approximations. The
3.5664 m inner field width and Goal coordinates follow p105/A10. Loaders, exact
starting inventory and full match reset are absent. This is not a complete
competition field or match simulator.

`field` runs positive/negative contact fixtures before writing the inspection
scene. Tests include a falling Cup, a Pin in a Goal, a stable Pin–Cup–Pin stack,
an outside Pin, and above-rim/horizontal rejection. Scoring is a conservative
**upright nesting subset of SC2**, requiring exactly one Pin tip inside the
opening below its rim. It is not general tilted-overlap adjudication. SC3 color
visibility, yellow ownership, Toggle ownership, full expansion, autonomous
bonus/AWP, protected zones, robot interaction and match phases are unimplemented.
`official_match_score` is always null. A Goal collision fails this benchmark;
it is not automatically a game-rule violation. Single-object tasks cannot
exceed the one-Pin/one-Cup possession limit, but no general possession classifier
is claimed.

### Design, assembly and physical assumptions

`assembly-1` is a rooted, ordered component graph. Each instance specifies an
ID, parent, catalog primitive, attachment (`origin`, beam `tip`, tool-mount
`tool`), meter offsets and full XYZ dimensions, joint, axis, limits, gear
reduction and controller role. Rotation is in radians. `chassis` coordinates
use X forward, Y left, Z up relative to the chassis body. Beam `tip` is at its
full X length; `tool` is 30 mm beyond the tool mount. Roller X is axial length
and Y/Z are equal diameters. Collision geometry, body inertia and rendering are
compiled from the same graph. No independent AI mass or torque fields exist.

Rigid parts are fixed body transforms. Revolute and passive prismatic joints
compile to MuJoCo joints. **Powered prismatic mechanisms are rejected** until a
linear transmission/controller extension is validated. The supported controller
requires a lift, wrist, and paired claw or roller roles. Arbitrary linkages and
closed-loop kinematic mechanisms are not yet supported. Edits can add/remove
parts or replace an instance's geometry, mounting, joint or reduction.

The reused chassis supplies four motorized wheel/axle hinges and visible
bearing/fastener conventions. The arm has two bearing towers and mounted motor
mass contributions. Hidden image connections, exact arm dimensions, motor
mounts, holes and transmission details are inferred. The visible reference is
Clawbot instructions **276-6009-750**, not a dimensioned engineering drawing.
The seed is a plausible mechanism reference, not an exact Clawbot reconstruction.
Raised chassis crossmembers are removed to clear the arm; their rigidity is
represented by the fixed chassis subassembly. Detailed replacement bracing,
screw-fit checks and mechanism shaft/bearing manufacturability still need
engineering review. Motor cases are mass contributions with disabled collision;
this can conceal packaging conflicts and is a significant limitation.

Checks reject unknown/disconnected/duplicate parts, invalid axes/limits, unsupported
gearing/catalog entries, unsupported powered slides, initial penetrations over
4 mm, and an initial 18-inch cube violation (1 mm numerical allowance). They
screen declared total motor power at 88 W, with 44 W drive allocation. These
are **partial construction checks**, not proof of physical buildability or
competition legality. Those statuses are separately recorded.

Approximate motors use a bounded linear torque-speed law: 1.05 N m and 200 rpm
for each 11 W direct output; claw actuators use half torque. Ratios 1/3/5
multiply torque and divide speed with ideal efficiency. No thermal model,
backlash or shaft stress is simulated. Wheels use ordinary frictional contact,
including lateral scrub, without hidden chassis forces. Fixed component mass
assumptions are not verified VEX specifications. Pin mass is 0.116 kg; Cup mass
is 0.108 kg. MuJoCo timestep is 2 ms, contact solref is 0.01 s / damping ratio 1.
Floor contact friction varies 0.5–0.8; explicit floor priority prevents MuJoCo's
maximum-coefficient mixing from masking low-friction tests. Gripper coefficient
is fixed at 1.1. Contact compliance, mass, geometry and motor calibration still
need hardware validation; only floor friction sensitivity is exercised here.

### Benchmark and search interpretation

The primary task acquires an upright Pin, lifts it through actual mechanism
contact, drives to an alliance-height Goal at X=0.8 m, lowers and releases it.
The cropped scene preserves the Goal geometry. A Cup extension tries placing a
Cup over a preplaced Pin in that Goal. This is already a combined drive/acquire/
transport/place sequence; multi-object cycles and full-field navigation remain
future work.

Object starts at X=0.36 m plus/minus 8 mm and Y plus/minus 8 mm. Floor friction
is uniformly sampled from 0.5–0.8. Robot initial pose is fixed for these mechanism
tests. The state controller sees exact robot/object poses and encoders—privileged
simulator observations, not a controller ready for a V5 Brain. Bounded wheel
speed feedback and lift/wrist/grip or roller actuation implement approach,
acquire, lift, transport, lower, release and one recovery attempt. No object
welds, teleportation, root pose writes or hidden gripping forces are used during
simulation. Root states are assigned only for recorded visual playback.

Each candidate gets the same tuning schedule (default two closure settings on
training seeds 17 and 29); selection uses the task ranking. Development seeds
are 41 and 53; held-out seeds are 1009 and 1013 (1019 reserved for future checks).
Finalists are the seed, development champion, and a distinct architecture where
available. Final seeds do not influence parent selection. Default duration is
32 simulated seconds. A release must remain nested without robot contact for
0.75 seconds, after actual lifting, without tipping or Goal collisions.
One sustained Goal contact counts one episode, with 0.25 s separation needed
for a new episode. Metrics include pickup error/time, placement error, drops,
jams/recovery attempts, state-transition times, saturation and wheel travel.

Lexicographic ranking prioritizes success count, then fewer unstable trials,
fewer collision episodes, lower mean time (failures receive the full timeout),
then pickup count. Partial/stationary runs cannot outrank a successful run.
Small samples support engineering comparisons, not statistical significance.
The HTML report compares parents on a shared evaluation split and includes
recorded trajectory scrubbing. Every candidate preserves its parent and edits.

Each search stores schemas, source snapshots/hashes, manual/resource snapshots,
dependency versions, seeds, limits, proposals/API metadata, XML, time-indexed
NPZ trajectories, per-trial JSON, controller settings and candidate/state JSON.
Checkpoints are atomic; cached trials are reused on resume. Resume refuses
changed source or manual hashes. Wall-clock budget is cumulative and checked
between trials/calls; one in-flight trial/request may finish after the threshold.
An exhausted budget remains exhausted on resume; start a new run with a larger
budget. Image rendering failure is recorded separately from physics failure.

### Viewing and verification

On macOS, use `mjpython` for the native viewer:

```sh
/private/tmp/chassis-lab-venv/bin/mjpython -m chassis_lab field --viewer
/private/tmp/chassis-lab-venv/bin/mjpython -m chassis_lab seed-robot --viewer
/private/tmp/chassis-lab-venv/bin/mjpython -m chassis_lab mechanism-replay <run>/g1_c1/development_41.xml
```

Recorded playback loops until the window closes. It visualizes saved physical
states; it does not rerun a controller. Open `<run>/report.html` directly with
no server. For screenshots and a short animated replay:

```sh
python -m chassis_lab.override.visual <run>/g1_c1/development_41.xml
python -m pytest -q
python -m tools.verify_mechanism
```

`--no-render` keeps the entire physics/search pipeline headless. Rendering needs
an available OpenGL context; sandboxed macOS graphics may require desktop
permission. Tests cover invalid assemblies, derived robot mass/envelope, actual
wheel/arm/claw actuation, nesting/stack contact fixtures, physical roller pickup
and placement, successful-vs-stationary ranking, and repeatability within 1e-9
absolute qpos tolerance on the same installed MuJoCo/platform. Bitwise or 1e-9
agreement across different MuJoCo versions/architectures is not promised.

### Executed results, September 5, 2026

Completed archive: `mechanism_runs/20260905T191807Z_offline_d4e41`.
This is **offline handwritten search**, with one seed plus nine proposals across
three generations, 40 tuning/development trials and six finalist Pin/Cup trials.
It completed in approximately 227 seconds before report rendering; a resume
check reused saved trials successfully.

| Design | Development Pin success | Mean successful time | Held-out Pin success | Mean successful time |
|---|---:|---:|---:|---:|
| Image-inspired claw seed | 0/2 | — | 0/2 | — |
| `g1_c1`, rollers + passive guide | 2/2 | 25.51 s | 2/2 | 24.89 s |
| Shorter-arm roller descendants | 2/2 each | 27.25 s | Not selected | — |

All claw variants failed acquisition. Both finalists failed the single Cup
extension trial (0/1 each). Successful Pin trials had no Goal collision episodes
or instability. The structural change improved this simulated task; later
length mutations regressed on time. Several handwritten mutations are duplicates,
so nine proposals do not mean nine unique engineering designs. No statistical
or real-world reliability claim follows from two held-out trials.

`assemblies/mechanism_verification/verification.json` records the actuated seed
moving 0.488 m in two seconds, turning 0.358 rad in two seconds, and reaching
residual planar speeds below 0.006 m/s after three seconds of braking. This
controller's turning response is sluggish because of wheel scrub. Separate
roller trials at floor friction 0.50/0.65/0.80 succeeded in 24.84/25.93/26.04 s.
The test suite passed **29 tests** after the contact-priority correction.

Actual AI attempt: `mechanism_runs/20260905T191720Z_ai_e0e28`, status **blocked**,
`AuthenticationError`, HTTP **401**, zero returned candidate proposals. Its
preflight seed trials used the earlier contact-mixing version recorded in that
archive; they are not used for the final offline comparison. An earlier partial
offline run `20260905T190351Z_offline_7026b` was interrupted to correct contact
priority and is superseded. Live structured proposal parsing, repair behavior
against actual model output, and multiple AI generations remain unverified.

Useful artifacts in the completed archive:

- `report.html`: comparison, parent hypotheses, every candidate image and XY replay scrubber.
- `summary.json` and each candidate's `aggregate.json`: derived numerical summaries.
- `g1_c1/final_1009.gif`: held-out successful physical manipulation.
- `seed/development_41.gif`: failed claw attempt.
- `g1_c1/cup_1009.gif`: failed Cup extension.
- `g1_c1/final_1009.xml` / `.npz`: interactive MuJoCo replay.
- `resources/manual.pdf`: pinned official manual; other geometry is reconstructed.

To regenerate aggregate JSON after a new run:
`python -m tools.summarize_override <run>`. The normal report already calculates
aggregate comparison values from saved trials.
