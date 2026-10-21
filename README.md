# 3D AI Rocket Landing Simulator

**A physics-based 3D rocket simulation where a Reinforcement Learning (RL) agent learns to land a custom rocket under changing environmental conditions.**

<!-- Replace this section with a demo GIF or video once available. -->
<!-- ![Rocket Landing Demo](assets/demo.gif) -->

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Reinforcement Learning](https://img.shields.io/badge/AI-Reinforcement%20Learning-purple)
![Physics](https://img.shields.io/badge/Physics-3D%20Dynamics-orange)
![Visualization](https://img.shields.io/badge/Visualization-Godot-478CBF?logo=godot-engine)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview

The **3D AI Rocket Landing Simulator** is a Python-based simulation and reinforcement learning project designed to teach an AI agent to control a custom rocket and achieve safe, stable landings.

The project combines physical simulation, environmental modeling, machine learning, 3D visualization, and experimental analysis into a modular system.

Rather than relying on a predefined sequence of movements, the agent learns a control policy by interacting with the simulated environment, receiving rewards or penalties, and improving through repeated training episodes.

The long-term objective is to develop an agent capable of landing under different initial conditions, wind patterns, fuel levels, atmospheric conditions, and other sources of uncertainty.

> **Development status:** This repository describes the planned architecture and implementation roadmap. Features are not considered implemented until their code, tests, and results are available. Update the status badges, commands, screenshots, and results as development progresses.

## Project Objectives

- Build a custom 3D rocket with configurable physical properties.
- Simulate gravity, thrust, mass variation, fuel consumption, rotation, and landing contact.
- Gradually introduce wind, atmospheric density, temperature, pressure, and aerodynamic forces.
- Create a custom Gymnasium environment for reinforcement learning.
- Train a landing policy using Proximal Policy Optimization (PPO).
- Support fast, headless training without rendering every frame.
- Visualize trained flights in a real-time 3D viewer.
- Record flight telemetry and landing outcomes for every episode.
- Evaluate reliability across both familiar and previously unseen conditions.
- Compare the learned policy with a baseline controller.
- Make the project reproducible, testable, and open source.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main simulation and training language |
| MuJoCo | Rigid-body dynamics and contact physics |
| NumPy | Numerical calculations and vector operations |
| Gymnasium | Reinforcement learning environment interface |
| Stable-Baselines3 | PPO training and evaluation utilities |
| PyTorch | Neural network computation through Stable-Baselines3 |
| Godot Engine | Planned real-time 3D visualization |
| Pandas | Telemetry processing and experiment analysis |
| Matplotlib | Trajectory, reward, and performance graphs |
| PyYAML | External configuration files |
| Pytest | Automated testing |

### Why this stack?

**MuJoCo** provides rigid-body simulation and contact dynamics. Rocket-specific propulsion, fuel behavior, aerodynamics, and environmental forces will be implemented and validated as part of this project.

**Gymnasium** provides a standard interface for observations, actions, rewards, resets, and episode termination.

**Stable-Baselines3** provides a PPO implementation that can train an agent to produce continuous control actions.

**Godot** will provide a dedicated 3D visualization layer without making the training loop dependent on graphical rendering.

References:

- [MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/)
- [Gymnasium documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/)
- [Godot documentation](https://docs.godotengine.org/en/stable/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)

## System Architecture

The project is divided into six main subsystems.

### 1. Physics Engine

Responsible for calculating the rocket's physical state over time.

Responsibilities:

- Translational motion in three dimensions.
- Rotational motion and angular velocity.
- Gravity and thrust.
- Rocket mass and fuel depletion.
- Engine limits and actuator response.
- Aerodynamic forces and wind interaction.
- Landing gear and ground contact.
- Collision and failure detection.

### 2. Environmental Model

Responsible for the conditions in which the rocket operates.

Planned features:

- Configurable gravity.
- Wind speed, direction, and gusts.
- Atmospheric pressure and density.
- Ambient temperature.
- Aerodynamic drag.
- Terrain elevation and landing-pad slope.
- Randomized environmental conditions.
- Sensor noise and model uncertainty.

### 3. Reinforcement Learning Environment

Exposes the simulation to the agent through a Gymnasium-compatible interface.

The environment defines:

- Observation space.
- Continuous action space.
- Reward function.
- Episode reset behavior.
- Success and failure conditions.
- Episode metrics and diagnostic information.

### 4. AI Training System

Uses PPO to learn a policy for controlling the rocket.

Responsibilities:

- Training on repeated simulation episodes.
- Saving model checkpoints.
- Tracking rewards and episode statistics.
- Evaluating the policy on separate scenarios.
- Comparing trained and baseline controllers.

### 5. Telemetry and Evaluation

Records data from every simulation episode and calculates performance metrics.

### 6. 3D Visualization

A separate Godot application will display the rocket, terrain, environment, trajectory, and telemetry.

The viewer will consume simulation state through a defined communication interface. It will not independently calculate a second version of the rocket physics.

## Planned Repository Structure

```text
rocket-landing-ai/
│
├── README.md
├── LICENSE
├── NOTICE
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── rocket.yaml
│   ├── physics.yaml
│   ├── environment.yaml
│   └── training.yaml
│
├── src/
│   └── rocket_sim/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── physics/
│       │   ├── rocket_model.py
│       │   ├── dynamics.py
│       │   ├── propulsion.py
│       │   ├── aerodynamics.py
│       │   ├── atmosphere.py
│       │   ├── wind.py
│       │   └── landing_gear.py
│       │
│       ├── environment/
│       │   ├── gym_env.py
│       │   ├── observations.py
│       │   ├── actions.py
│       │   ├── rewards.py
│       │   └── termination.py
│       │
│       ├── agents/
│       │   ├── train_ppo.py
│       │   ├── evaluate.py
│       │   └── baseline_controller.py
│       │
│       ├── telemetry/
│       │   ├── recorder.py
│       │   ├── metrics.py
│       │   └── reports.py
│       │
│       └── viewer_bridge/
│           ├── server.py
│           └── state.py
│
├── models/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── data/
│   └── .gitkeep
│
├── tests/
│   ├── test_physics.py
│   ├── test_environment.py
│   └── test_landing.py
│
├── scripts/
│   ├── run_simulation.py
│   ├── train.py
│   └── watch_landing.py
│
└── viewer3d/
    └── Godot project files
```

This is the intended structure. Files and entry points will be added as their corresponding features are implemented.

## Installation and Setup

### Prerequisites

- Windows 10/11, Linux, or macOS.
- Python 3.12 recommended for the initial development environment.
- Git.
- A code editor such as Visual Studio Code.
- Godot Engine for the planned 3D viewer.
- Sufficient CPU resources for repeated simulation and training.

Python 3.12 is a starting recommendation, not a claim that other supported Python versions cannot work. Confirm package compatibility when establishing the project's pinned dependencies.

### Step 1: Clone the repository

Once the repository has been created on GitHub:

```bash
git clone https://github.com/YOUR_USERNAME/rocket-landing-ai.git
cd rocket-landing-ai
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Create a virtual environment

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks virtual environment activation, you can run the environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

On Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install the Python dependencies

After activating the environment:

```bash
python -m pip install --upgrade pip

python -m pip install mujoco gymnasium "stable-baselines3[extra]" numpy pandas matplotlib pytest pyyaml
```

MuJoCo's Python package includes the bindings needed to access its simulation engine. A separate manual MuJoCo binary installation is generally unnecessary for standard Python usage.

### Step 4: Verify the installation

Run:

```bash
python -c "import mujoco, gymnasium, stable_baselines3, numpy, pandas, matplotlib, yaml; print('Core dependencies imported successfully')"
```

Then verify Pytest:

```bash
python -m pytest --version
```

### Step 5: Install the 3D viewer

Download and install Godot from:

https://godotengine.org/download/

The viewer is a separate component. Its project files and integration code will be added during the visualization milestone.

### Step 6: Install the project itself

Once `pyproject.toml` and the Python package structure have been implemented, install the project in editable mode:

```bash
python -m pip install -e .
```

This command is intended for the completed package structure and is not guaranteed to work before those files exist.

## How to Work With the Project

Development is divided into incremental stages. Complete and test each stage before depending on it in the next one.

### Stage 1 — Build the physical rocket

Implement the initial rocket model with:

- Configurable dry mass and fuel mass.
- Gravity.
- Main engine thrust.
- Fuel consumption.
- Position and velocity.
- Orientation and angular velocity.
- Basic ground contact.

The first version should use simplified atmospheric conditions and a flat landing surface.

**Validation goals:**

- The rocket accelerates correctly under gravity.
- Thrust acts in the expected direction.
- Fuel consumption reduces the available fuel.
- Total mass decreases consistently as fuel burns.
- The simulation detects ground contact.
- Repeated runs with the same initial conditions are reproducible.

Do not begin training until these tests pass.

### Stage 2 — Add the environment

Introduce wind, atmosphere, drag, and variable starting conditions.

Start with simplified models rather than full computational fluid dynamics.

Examples of randomized conditions:

- Initial altitude.
- Horizontal position.
- Initial descent velocity.
- Remaining fuel.
- Wind speed and direction.
- Ambient temperature.
- Atmospheric density.

Use fixed random seeds when debugging so that the same conditions can be reproduced.

### Stage 3 — Create the Gymnasium environment

Implement the custom environment with the standard Gymnasium interface:

```python
reset(seed=None, options=None)
step(action)
```

The reset operation initializes a new episode and returns the initial observation and information dictionary.

The step operation applies an action, advances the simulation, and returns the observation, reward, termination status, truncation status, and information dictionary.

The environment should also define valid observation and action spaces.

Once implemented, run the Gymnasium environment checker and automated tests before starting RL training.

### Stage 4 — Define the observation space

The initial observation can contain:

| Observation | Description |
|---|---|
| Relative position | Rocket position relative to the target |
| Linear velocity | Velocity along the three axes |
| Orientation | Rocket attitude relative to the desired landing attitude |
| Angular velocity | Rotation rate around the three axes |
| Fuel fraction | Remaining usable fuel |
| Engine state | Available thrust or actuator state |
| Environmental data | Wind and atmospheric information, if observable |

All observations must use consistent units and sensible numerical ranges.

Later, the agent can receive noisy sensor measurements rather than perfect simulated state information.

### Stage 5 — Define the action space

Use continuous-valued actions.

The initial action design may include:

- Main engine throttle.
- Two-axis attitude control.
- Optional engine gimbal control.

For example, actions could be normalized to the interval `[-1, 1]` and mapped to valid actuator commands.

The simulation itself must enforce actuator limits, fuel availability, and engine constraints. The agent should not be able to bypass physical limits by generating an invalid action.

### Stage 6 — Design the reward function

The reward function guides the agent toward a stable landing.

The planned reward components include:

- Progress toward the landing target.
- Reduction of unsafe descent speed.
- Reduced horizontal drift.
- Stable orientation.
- Low angular velocity at touchdown.
- Successful landing.
- Fuel efficiency.
- Penalties for collisions, tipping, and fuel exhaustion.

A successful landing should receive a substantial positive terminal reward. A crash or other unrecoverable failure should receive a substantial negative reward.

Reward weights must be tested empirically. Poorly designed rewards can encourage behavior that achieves a high score without producing a genuinely safe landing.

### Stage 7 — Train the AI

Use PPO from Stable-Baselines3 as the initial algorithm.

Training should begin in a simple environment:

1. Start with a fixed gravity and flat terrain.
2. Use predictable initial conditions.
3. Train the policy to land successfully.
4. Evaluate its performance separately from training.
5. Save checkpoints.
6. Introduce progressively more difficult scenarios.
7. Track success rate, crash rate, and fuel efficiency.

The training implementation will eventually support configuration-based settings for total timesteps, random seeds, checkpoint frequency, and evaluation frequency.

Training timesteps are not the same as successful landings. A large training run does not by itself prove that the agent has learned a reliable policy.

### Stage 8 — Build the 3D viewer

The planned Godot viewer will include:

- A custom rocket model.
- Landing legs and a landing surface.
- Terrain and environmental visuals.
- A camera that follows the rocket.
- Flight trajectory visualization.
- Altitude, velocity, and fuel indicators.
- Wind direction and speed indicators.
- Live landing status.
- Pause, restart, and replay controls.

Python will remain responsible for the simulated state. Godot will display that state and provide the interactive visual interface.

### Stage 9 — Evaluate the trained policy

Evaluate the agent on scenarios it did not encounter during training.

Measure:

- Stable landing success rate.
- Crash rate.
- Horizontal landing error.
- Vertical and horizontal touchdown speed.
- Tilt at touchdown.
- Fuel consumed.
- Time to land.
- Performance under changing wind and starting conditions.

Run repeated trials with recorded seeds. Keep evaluation results separate from training statistics.

## Physics and Environmental Modeling

The simulation will use SI units throughout:

- Distance: metres (`m`)
- Mass: kilograms (`kg`)
- Time: seconds (`s`)
- Velocity: metres per second (`m/s`)
- Acceleration: metres per second squared (`m/s²`)
- Force and thrust: newtons (`N`)
- Pressure: pascals (`Pa`)
- Temperature: kelvin (`K`)

### Gravity

Gravity is a configurable acceleration vector.

For a simple vertical environment, the initial model can use a constant gravitational acceleration. Later versions may support different celestial bodies or altitude-dependent gravity.

### Mass and fuel

The rocket's total mass includes its dry mass, remaining fuel, and any payload.

As fuel is consumed, the mass and potentially the center of mass and inertia change.

The simulation must account for these changes consistently. Fuel consumption and thrust should also be physically compatible with the selected engine model.

### Propulsion

The engine model will eventually include:

- Maximum thrust.
- Throttle range.
- Fuel consumption.
- Response delay.
- Startup and shutdown behavior.
- Optional gimbal limits.
- Optional engine failures or performance variation.

The initial engine model will be simplified and deterministic.

### Wind and aerodynamics

Wind will be represented as a vector field that can vary with position and time.

Aerodynamic drag depends on factors including relative air velocity, air density, drag coefficient, and reference area.

A simplified drag model can be used initially. More advanced versions may include lift, changing aerodynamic coefficients, and more detailed atmospheric profiles.

### Temperature and pressure

Temperature and pressure will influence the environmental model through explicitly defined relationships.

Temperature alone will not automatically change the engine's performance unless that behavior is implemented in the propulsion model. Similarly, atmospheric density should be calculated from a consistent atmospheric model or configured explicitly.

### Landing contact

The landing system will consider:

- Landing surface elevation and slope.
- Landing gear geometry.
- Vertical and horizontal touchdown velocity.
- Rocket orientation.
- Angular velocity.
- Contact forces and tipping.
- Structural load estimates.

An impact-force estimate from a simplified model must not be presented as a validated structural stress analysis.

## Reinforcement Learning Design

### Algorithm

The initial algorithm is Proximal Policy Optimization (PPO).

PPO is suitable for experimenting with continuous control policies and offers a straightforward starting point for training the rocket controller.

Other algorithms can be considered later if PPO's learning efficiency or stability is insufficient.

### Curriculum learning

Training complexity will increase gradually:

1. Calm conditions and simple landings.
2. Variable starting position and descent velocity.
3. Changing fuel levels.
4. Moderate wind.
5. More variable atmospheric conditions.
6. Sensor noise and actuator delays.
7. Previously unseen combinations of conditions.

The aim is to develop a robust policy rather than one that memorizes a small set of landing scenarios.

### Baseline controller

A simple rule-based or conventional controller should be implemented as a comparison baseline.

This allows us to determine whether the trained agent actually improves landing performance instead of merely comparing its reward with zero.

### Reproducibility

Each experiment should record:

- Random seed.
- Configuration file.
- Training timestep count.
- Algorithm and hyperparameters.
- Model checkpoint.
- Evaluation conditions.
- Performance metrics.

This makes comparisons between experiments more meaningful.

## Telemetry and Experiment Data

Each episode should produce a structured record of the rocket's flight.

### Planned telemetry fields

| Category | Example fields |
|---|---|
| Identification | Episode ID, seed, timestamp |
| Initial state | Initial position, velocity, mass and fuel |
| Motion | Position, linear velocity, orientation, angular velocity |
| Propulsion | Throttle, thrust, fuel remaining |
| Environment | Wind vector, temperature, pressure, density |
| Landing | Touchdown velocity, target error, tilt, contact status |
| Outcome | Success, crash, tipping, fuel exhaustion |
| Training | Episode reward, duration, model version |

Telemetry can be stored as CSV for straightforward analysis and JSON for structured episode metadata.

### Planned analysis outputs

- Training reward curve.
- Landing success rate over time.
- Crash rate.
- Fuel consumption distribution.
- Landing position distribution.
- Touchdown speed distribution.
- Success rate by wind condition.
- Comparison between baseline and trained controller.

No performance numbers should be reported until they have been measured in actual evaluation runs.

## Testing

Automated tests are required before trusting training results.

### Physics tests

- Gravity-only free fall.
- Thrust direction and magnitude.
- Fuel consumption and mass changes.
- State updates and coordinate conventions.
- Deterministic simulation under fixed conditions.
- Collision detection.
- Numerical stability at the chosen timestep.

### Environment tests

- Valid observation shapes and data types.
- Valid action handling.
- Correct reset behavior.
- Correct termination and truncation behavior.
- No invalid fuel or mass states.
- Correct reward and terminal conditions.

### AI tests

- The agent can interact with the environment.
- Training checkpoints can be saved and loaded.
- Evaluation is separate from training.
- A baseline controller can be evaluated.
- Results can be reproduced using recorded seeds and configurations.

Run the tests with:

```bash
python -m pytest
```

The command becomes useful once the test suite has been implemented.

## Evaluation Metrics

The principal metric is **stable landing success rate**.

For a set of evaluation episodes:

`Success rate = Successful landings / Total evaluation episodes`

The definition of success must be specified before testing. It should include acceptable limits for landing position, touchdown speed, orientation, angular velocity, and ground stability.

Additional metrics:

| Metric | Purpose |
|---|---|
| Crash rate | Measures catastrophic failures |
| Landing error | Measures accuracy |
| Touchdown velocity | Measures impact severity |
| Fuel consumption | Measures efficiency |
| Time to land | Measures duration |
| Stability | Measures attitude and motion at touchdown |
| Robustness | Measures performance under unseen conditions |

Use a separate evaluation set and report the number of trials alongside the results. A high success rate on a small number of attempts is not sufficient evidence of general reliability.

## Development Roadmap

- [x] **Phase 1:** Repository setup and dependency verification.
- [x] **Phase 2:** Core rocket dynamics and gravity.
- [x] **Phase 3:** Fuel consumption and propulsion.
- [ ] **Phase 4:** Ground contact and landing conditions.
- [ ] **Phase 5:** Automated physics tests.
- [ ] **Phase 6:** Gymnasium environment.
- [ ] **Phase 7:** Observations, actions, rewards, and termination.
- [ ] **Phase 8:** PPO training and checkpointing.
- [ ] **Phase 9:** Baseline controller and evaluation.
- [ ] **Phase 10:** Wind and atmospheric modeling.
- [ ] **Phase 11:** Telemetry and performance reports.
- [ ] **Phase 12:** Godot 3D viewer.
- [ ] **Phase 13:** Simulation-to-viewer integration.
- [ ] **Phase 14:** Robustness testing and final documentation.
- [ ] **Phase 15:** Public release and demonstration video.

## Troubleshooting

### Python package installation fails

Check that the intended Python interpreter is active:

```bash
python --version
python -m pip --version
```

Install packages through `python -m pip` to avoid accidentally installing them into a different Python environment.

### MuJoCo cannot be imported

Try:

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

If this fails, verify the environment and the installation output before attempting to build MuJoCo from source.

### Training is too slow

- Disable rendering during training.
- Reduce unnecessary per-step logging.
- Profile the simulation before optimizing it.
- Consider parallel environments after the environment is stable.
- Save checkpoints at reasonable intervals.

### The AI receives a high reward but crashes

Inspect the reward function and terminal conditions. A reward function that values proximity to the ground more than safe touchdown can encourage dangerous behavior.

### The rocket behaves incorrectly

Verify units, coordinate frames, force application points, mass properties, timestep, and action scaling before changing the reward function.

### The 3D viewer disagrees with the simulation

Check the coordinate-system conversion, units, rotation representation, and synchronization between Python and Godot. The viewer should display the simulation's authoritative state rather than independently integrate the rocket.

## Git and Artifact Management

Large model checkpoints, training logs, generated datasets, caches, and temporary files should not be committed indiscriminately.

A suitable `.gitignore` should include entries such as:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Virtual environments
.venv/
venv/

# Training artifacts
models/*
!models/.gitkeep

logs/*
!logs/.gitkeep

data/*
!data/.gitkeep

# Test and tooling caches
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Local configuration
.env
.env.*

# IDE
.vscode/
.idea/

# Generated viewer and editor caches
.godot/
```

Review these patterns before committing. Track configuration files, source code, tests, documentation, and small reproducible examples.

If a trained model is small enough for the repository, consider publishing it as a release artifact rather than committing every checkpoint.

## License

This project is intended to use the **MIT License** for original project code.

The MIT License permits reuse, modification, distribution, and commercial use subject to its terms, including preserving applicable copyright and permission notices.

If code is reused from Gymnasium or another third-party project:

- Preserve the original copyright and license notices where required.
- Document the upstream source and any modifications.
- Include the relevant third-party license text.
- Review the licenses of all bundled models, textures, fonts, and other assets.

An example upstream source used for reference is:

[Gymnasium LunarLander source](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py)

Do not claim that third-party code is entirely your original work. The repository's MIT license does not erase the licensing obligations attached to reused material.

## Limitations and Disclaimer

This is an educational and research-oriented simulation project.

The simulator will use progressively more detailed physical models, but a visually convincing simulation is not automatically a validated engineering model.

Potential limitations include:

- Simplified propulsion and fuel models.
- Approximate atmospheric and aerodynamic effects.
- Incomplete structural and thermal analysis.
- Idealized sensors and actuator behavior.
- Numerical and model uncertainty.
- Reinforcement learning policies that fail outside their training distribution.

The project is not intended to certify real-world flight safety or replace professional aerospace analysis and validation.

A trained agent should be evaluated systematically, and successful simulated landings should not be interpreted as proof that a real rocket could land safely.

## Future Improvements

Potential extensions include:

- More detailed rocket engine and throttle-response models.
- Multiple rocket configurations.
- More sophisticated atmospheric profiles.
- Terrain-aware landing-site selection.
- Sensor-based state estimation.
- Fault injection and uncertainty modeling.
- Alternative RL algorithms.
- Conventional guidance and control baselines.
- Parallelized training.
- Interactive scenario configuration.
- Flight replay and comparison tools.
- Exportable evaluation reports.

## Contributing

Contributions, suggestions, bug reports, and improvements are welcome.

For significant changes:

1. Open an issue describing the proposed feature or problem.
2. Explain the physics or learning assumptions behind the change.
3. Add tests for new behavior.
4. Document any configuration or API changes.
5. Report evaluation results where relevant.

## Author

**Mathan Michael Raj. A**

GitHub: mathanmichaelraj-byte(https://github.com/mathanmichaelraj-byte)

Update this section with your actual GitHub username and any relevant project links.

---

**Project principle:** Build the physics, validate the environment, train the agent, measure the results, and only then claim reliable landing performance.

