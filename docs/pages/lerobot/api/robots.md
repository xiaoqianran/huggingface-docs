# Robots

Every robot in LeRobot implements the `Robot` interface: connect, read an observation, send an action,
disconnect. Writing a policy or a recording script against that interface means it works with any supported
arm without change.

This page is the generated reference. For wiring, calibration and first-run instructions, start with the
hardware guides — [SO-101](../so101), [LeKiwi](../lekiwi), [Hope Jr](../hope_jr), [Reachy 2](../reachy2),
[OpenArm](../openarm) — or [Imitation Learning for Robots](../il_robots) for the end-to-end workflow. For
community-maintained robots outside the core set, see [Third-Party Robots & Teleoperators](../third_party_robots).
To add a robot of your own, see [Bring Your Own Hardware](../integrate_hardware).

## Robot[[lerobot.robots.Robot]]

The abstract base class. Subclasses implement every method below; the contract described here is what a
policy or recording loop can rely on.

#### lerobot.robots.Robot[[lerobot.robots.Robot]]

```python
lerobot.robots.Robot(config: RobotConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/robot.py#L33)

**Parameters:**

config_class (RobotConfig) : The expected configuration class for this robot.

name (str) : The unique robot name used to identify this robot type.

The base abstract class for all LeRobot-compatible robots.

This class provides a standardized interface for interacting with physical robots.
Subclasses must implement all abstract methods and properties to be usable.

#### connect[[lerobot.robots.Robot.connect]]

```python
connect(calibrate: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/robot.py#L136)

**Parameters:**

calibrate (bool) : If True, automatically calibrate the robot after connecting if it's not calibrated or needs calibration (this is hardware-dependant).

Establish communication with the robot.

#### disconnect[[lerobot.robots.Robot.disconnect]]

```python
disconnect()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/robot.py#L237)

Disconnect from the robot and perform any necessary cleanup.

#### configure[[lerobot.robots.Robot.configure]]

```python
configure()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/robot.py#L202)

Apply any one-time or runtime configuration to the robot.
This may include setting motor parameters, control modes, or initial state.

#### calibrate[[lerobot.robots.Robot.calibrate]]

```python
calibrate()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/robot.py#L153)

Calibrate the robot if applicable. If not, this should be a no-op.

This method should collect any necessary data (e.g., motor offsets) and update the
:pyattr:`calibration` dictionary accordingly.

#### get_observation[[lerobot.robots.Robot.get_observation]]

```python
get_observation()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/robot.py#L210)

**Returns:** `RobotObservation`

A flat dictionary representing the robot's current sensory state. Its structure
should match :pymeth:`observation_features`.

Retrieve the current observation from the robot.

#### send_action[[lerobot.robots.Robot.send_action]]

```python
send_action(action: dict)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/robot.py#L222)

**Parameters:**

action (RobotAction) : Dictionary representing the desired action. Its structure should match :pymeth:`action_features`.

**Returns:** `RobotAction`

The action actually sent to the motors potentially clipped or modified, e.g. by
safety limits on velocity.

Send an action command to the robot.

#### observation_features[[lerobot.robots.Robot.observation_features]]

```python
observation_features()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/robot.py#L100)

A dictionary describing the structure and types of the observations produced by the robot.
Its structure (keys) should match the structure of what is returned by :pymeth:`get_observation`.
Values for the dict should either be:
- The type of the value if it's a simple value, e.g. `float` for single proprioceptive value (a joint's position/velocity)
- A tuple representing the shape if it's an array-type value, e.g. `(height, width, channel)` for images

Note: this property should be able to be called regardless of whether the robot is connected or not.

#### action_features[[lerobot.robots.Robot.action_features]]

```python
action_features()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/robot.py#L114)

A dictionary describing the structure and types of the actions expected by the robot. Its structure
(keys) should match the structure of what is passed to :pymeth:`send_action`. Values for the dict
should be the type of the value if it's a simple value, e.g. `float` for single proprioceptive value
(a joint's goal position/velocity)

Note: this property should be able to be called regardless of whether the robot is connected or not.

#### is_connected[[lerobot.robots.Robot.is_connected]]

```python
is_connected()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/robot.py#L127)

Whether the robot is currently connected or not. If `False`, calling :pymeth:`get_observation` or
:pymeth:`send_action` should raise an error.

#### is_calibrated[[lerobot.robots.Robot.is_calibrated]]

```python
is_calibrated()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/robot.py#L147)

Whether the robot is currently calibrated or not. Should be always `True` if not applicable

## RobotConfig[[lerobot.robots.RobotConfig]]

#### lerobot.robots.RobotConfig[[lerobot.robots.RobotConfig]]

```python
lerobot.robots.RobotConfig(id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/config.py#L23)

## make_robot_from_config[[lerobot.robots.make_robot_from_config]]

#### lerobot.robots.make_robot_from_config[[lerobot.robots.make_robot_from_config]]

```python
lerobot.robots.make_robot_from_config(config: RobotConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/utils.py#L28)

## SO-100 and SO-101 followers[[lerobot.robots.so_follower.SOFollower]]

`SO100Follower` and `SO101Follower` are aliases of the same `SOFollower` class; the two arms differ in their
configuration, not their control code. `SO100FollowerConfig` and `SO101FollowerConfig` are likewise aliases
of `SOFollowerRobotConfig`.

#### lerobot.robots.so_follower.SOFollower[[lerobot.robots.so_follower.SOFollower]]

```python
lerobot.robots.so_follower.SOFollower(config: SOFollowerRobotConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/so_follower.py#L54)

Generic SO follower base implementing common functionality for SO-100/101/10X.
Designed to be subclassed with a per-hardware-model `config_class` and `name`.

#### connect[[lerobot.robots.so_follower.SOFollower.connect]]

```python
connect(calibrate: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/so_follower.py#L110)

We assume that at connection time, arm is in a rest position,
and torque can be safely disabled to run calibration.

#### send_action[[lerobot.robots.so_follower.SOFollower.send_action]]

```python
send_action(action: dict)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/so_follower.py#L187)

**Returns:** `RobotAction`

the action sent to the motors, potentially clipped.

**Raises:** ``RobotDeviceNotConnectedError``

- ``RobotDeviceNotConnectedError`` -- if robot is not connected.

Command arm to move to a target joint configuration.

The relative action magnitude may be clipped depending on the configuration parameter
`max_relative_target`. In this case, the action sent differs from original action.
Thus, this function always returns the action actually sent.

#### lerobot.robots.so_follower.SOFollowerRobotConfig[[lerobot.robots.so_follower.SOFollowerRobotConfig]]

```python
lerobot.robots.so_follower.SOFollowerRobotConfig(port: str, disable_torque_on_disconnect: bool = True, max_relative_target: float | dict[str, float] | None = None, cameras: dict = <factory>, use_degrees: bool = True, position_p_coefficient: int = 16, position_i_coefficient: int = 0, position_d_coefficient: int = 32, num_read_retries: int = 2, id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/config_so_follower.py#L59)

## BiSOFollower[[lerobot.robots.bi_so_follower.BiSOFollower]]

Two SO followers driven as one bimanual robot.

#### lerobot.robots.bi_so_follower.BiSOFollower[[lerobot.robots.bi_so_follower.BiSOFollower]]

```python
lerobot.robots.bi_so_follower.BiSOFollower(config: BiSOFollowerConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/bi_so_follower/bi_so_follower.py#L31)

[Bimanual SO Follower Arms](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio

#### lerobot.robots.bi_so_follower.BiSOFollowerConfig[[lerobot.robots.bi_so_follower.BiSOFollowerConfig]]

```python
lerobot.robots.bi_so_follower.BiSOFollowerConfig(left_arm_config: SOFollowerConfig, right_arm_config: SOFollowerConfig, cameras: dict = <factory>, id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/bi_so_follower/config_bi_so_follower.py#L27)

Configuration class for Bi SO Follower robots.

## KochFollower[[lerobot.robots.koch_follower.KochFollower]]

#### lerobot.robots.koch_follower.KochFollower[[lerobot.robots.koch_follower.KochFollower]]

```python
lerobot.robots.koch_follower.KochFollower(config: KochFollowerConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/koch_follower/koch_follower.py#L33)

- [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow
  expansion, developed by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
- [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss

#### connect[[lerobot.robots.koch_follower.KochFollower.connect]]

```python
connect(calibrate: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/koch_follower/koch_follower.py#L89)

We assume that at connection time, arm is in a rest position,
and torque can be safely disabled to run calibration.

#### send_action[[lerobot.robots.koch_follower.KochFollower.send_action]]

```python
send_action(action: dict)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/koch_follower/koch_follower.py#L176)

**Parameters:**

action (RobotAction) : The goal positions for the motors.

**Returns:** `RobotAction`

The action sent to the motors, potentially clipped.

Command arm to move to a target joint configuration.

The relative action magnitude may be clipped depending on the configuration parameter
`max_relative_target`. In this case, the action sent differs from original action.
Thus, this function always returns the action actually sent.

#### lerobot.robots.koch_follower.KochFollowerConfig[[lerobot.robots.koch_follower.KochFollowerConfig]]

```python
lerobot.robots.koch_follower.KochFollowerConfig(port: str, disable_torque_on_disconnect: bool = True, max_relative_target: float | dict[str, float] | None = None, cameras: dict = <factory>, use_degrees: bool = False, id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/koch_follower/config_koch_follower.py#L24)

## LeKiwi[[lerobot.robots.lekiwi.LeKiwi]]

`LeKiwi` runs on the robot itself. `LeKiwiClient` is the host-side proxy that talks to it over the network
and presents the same `Robot` interface.

#### lerobot.robots.lekiwi.LeKiwi[[lerobot.robots.lekiwi.LeKiwi]]

```python
lerobot.robots.lekiwi.LeKiwi(config: LeKiwiConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/lekiwi/lekiwi.py#L36)

The robot includes a three omniwheel mobile base and a remote follower arm.
The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
forwarded to the remote follower arm (after applying a safety clamp).
In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.

#### send_action[[lerobot.robots.lekiwi.LeKiwi.send_action]]

```python
send_action(action: dict)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/lekiwi/lekiwi.py#L339)

**Returns:** `RobotAction`

the action sent to the motors, potentially clipped.

**Raises:** ``RobotDeviceNotConnectedError``

- ``RobotDeviceNotConnectedError`` -- if robot is not connected.

Command lekiwi to move to a target joint configuration.

The relative action magnitude may be clipped depending on the configuration parameter
`max_relative_target`. In this case, the action sent differs from original action.
Thus, this function always returns the action actually sent.

#### lerobot.robots.lekiwi.LeKiwiConfig[[lerobot.robots.lekiwi.LeKiwiConfig]]

```python
lerobot.robots.lekiwi.LeKiwiConfig(port: str = '/dev/ttyACM0', disable_torque_on_disconnect: bool = True, max_relative_target: float | dict[str, float] | None = None, cameras: dict = <factory>, use_degrees: bool = True, num_read_retries: int = 2, id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/lekiwi/config_lekiwi.py#L47)

#### lerobot.robots.lekiwi.LeKiwiClient[[lerobot.robots.lekiwi.LeKiwiClient]]

```python
lerobot.robots.lekiwi.LeKiwiClient(config: LeKiwiClientConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/lekiwi/lekiwi_client.py#L40)

#### connect[[lerobot.robots.lekiwi.LeKiwiClient.connect]]

```python
connect()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/lekiwi/lekiwi_client.py#L129)

Establishes ZMQ sockets with the remote mobile robot

#### disconnect[[lerobot.robots.lekiwi.LeKiwiClient.disconnect]]

```python
disconnect()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/lekiwi/lekiwi_client.py#L351)

Cleans ZMQ comms

#### get_observation[[lerobot.robots.lekiwi.LeKiwiClient.get_observation]]

```python
get_observation()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/lekiwi/lekiwi_client.py#L269)

Capture observations from the remote robot: current follower arm positions,
present wheel speeds (converted to body-frame velocities: x, y, theta),
and a camera frame. Receives over ZMQ, translate to body-frame vel

#### send_action[[lerobot.robots.lekiwi.LeKiwiClient.send_action]]

```python
send_action(action: dict)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/lekiwi/lekiwi_client.py#L323)

**Parameters:**

action (RobotAction) : array containing the goal positions for the motors.

**Returns:** `np.ndarray`

the action sent to the motors, potentially clipped.

**Raises:** ``RobotDeviceNotConnectedError``

- ``RobotDeviceNotConnectedError`` -- if robot is not connected.

Command lekiwi to move to a target joint configuration. Translates to motor space + sends over ZMQ

#### lerobot.robots.lekiwi.LeKiwiClientConfig[[lerobot.robots.lekiwi.LeKiwiClientConfig]]

```python
lerobot.robots.lekiwi.LeKiwiClientConfig(remote_ip: str, port_zmq_cmd: int = 5555, port_zmq_observations: int = 5556, teleop_keys: dict = <factory>, cameras: dict = <factory>, polling_timeout_ms: int = 15, connect_timeout_s: int = 5, id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/lekiwi/config_lekiwi.py#L87)

## OpenArmFollower[[lerobot.robots.openarm_follower.OpenArmFollower]]

#### lerobot.robots.openarm_follower.OpenArmFollower[[lerobot.robots.openarm_follower.OpenArmFollower]]

```python
lerobot.robots.openarm_follower.OpenArmFollower(config: OpenArmFollowerConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/openarm_follower/openarm_follower.py#L40)

OpenArms Follower Robot which uses CAN bus communication to control 7 DOF arm with a gripper.
The arm uses Damiao motors in MIT control mode.

#### calibrate[[lerobot.robots.openarm_follower.OpenArmFollower.calibrate]]

```python
calibrate()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/openarm_follower/openarm_follower.py#L164)

Run calibration procedure for OpenArms robot.

The calibration procedure:
1. Disable torque
2. Ask user to position arms in hanging position with grippers closed
3. Set this as zero position
4. Record range of motion for each joint
5. Save calibration

#### configure[[lerobot.robots.openarm_follower.OpenArmFollower.configure]]

```python
configure()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/openarm_follower/openarm_follower.py#L215)

Configure motors with appropriate settings.

#### connect[[lerobot.robots.openarm_follower.OpenArmFollower.connect]]

```python
connect(calibrate: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/openarm_follower/openarm_follower.py#L130)

Connect to the robot and optionally calibrate.

We assume that at connection time, the arms are in a safe rest position,
and torque can be safely disabled to run calibration if needed.

#### disconnect[[lerobot.robots.openarm_follower.OpenArmFollower.disconnect]]

```python
disconnect()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/openarm_follower/openarm_follower.py#L345)

Disconnect from robot.

#### get_observation[[lerobot.robots.openarm_follower.OpenArmFollower.get_observation]]

```python
get_observation()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/openarm_follower/openarm_follower.py#L226)

Get current observation from robot including position, velocity, and torque.

Reads all motor states (pos/vel/torque) in one CAN refresh cycle
instead of 3 separate reads.

#### send_action[[lerobot.robots.openarm_follower.OpenArmFollower.send_action]]

```python
send_action(action: dict, custom_kp: dict[str, float] | None = None, custom_kd: dict[str, float] | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/openarm_follower/openarm_follower.py#L266)

**Parameters:**

action : Dictionary with motor positions (e.g., "joint_1.pos", "joint_2.pos")

custom_kp : Optional custom kp gains per motor (e.g., {"joint_1": 120.0, "joint_2": 150.0})

custom_kd : Optional custom kd gains per motor (e.g., {"joint_1": 1.5, "joint_2": 2.0})

**Returns:**

The action actually sent (potentially clipped)

Send action command to robot.

The action magnitude may be clipped based on safety limits.

#### lerobot.robots.openarm_follower.OpenArmFollowerConfig[[lerobot.robots.openarm_follower.OpenArmFollowerConfig]]

```python
lerobot.robots.openarm_follower.OpenArmFollowerConfig(port: str, side: str | None = None, can_interface: str = 'socketcan', use_can_fd: bool = True, can_bitrate: int = 1000000, can_data_bitrate: int = 5000000, disable_torque_on_disconnect: bool = True, use_velocity_and_torque: bool = False, max_relative_target: float | dict[str, float] | None = None, cameras: dict = <factory>, motor_config: dict = <factory>, position_kp: list = <factory>, position_kd: list = <factory>, joint_limits: dict = <factory>, id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/openarm_follower/config_openarm_follower.py#L125)

## BiOpenArmFollower[[lerobot.robots.bi_openarm_follower.BiOpenArmFollower]]

#### lerobot.robots.bi_openarm_follower.BiOpenArmFollower[[lerobot.robots.bi_openarm_follower.BiOpenArmFollower]]

```python
lerobot.robots.bi_openarm_follower.BiOpenArmFollower(config: BiOpenArmFollowerConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/bi_openarm_follower/bi_openarm_follower.py#L31)

Bimanual OpenArm Follower Arms

#### lerobot.robots.bi_openarm_follower.BiOpenArmFollowerConfig[[lerobot.robots.bi_openarm_follower.BiOpenArmFollowerConfig]]

```python
lerobot.robots.bi_openarm_follower.BiOpenArmFollowerConfig(id: str | None = 'bi_openarm_follower', calibration_dir: pathlib.Path | None = None, left_arm_config: OpenArmFollowerConfigBase, right_arm_config: OpenArmFollowerConfigBase, cameras: dict = <factory>)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/bi_openarm_follower/config_bi_openarm_follower.py#L27)

Configuration class for Bi OpenArm Follower robots.

## OmxFollower[[lerobot.robots.omx_follower.OmxFollower]]

#### lerobot.robots.omx_follower.OmxFollower[[lerobot.robots.omx_follower.OmxFollower]]

```python
lerobot.robots.omx_follower.OmxFollower(config: OmxFollowerConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/omx_follower/omx_follower.py#L33)

- [OMX](https://github.com/ROBOTIS-GIT/open_manipulator),
  expansion, developed by Woojin Wie and Junha Cha from [ROBOTIS](https://ai.robotis.com/)

#### connect[[lerobot.robots.omx_follower.OmxFollower.connect]]

```python
connect(calibrate: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/omx_follower/omx_follower.py#L89)

For OMX robots that come pre-calibrated:
- If default calibration from package doesn't match motors, read from motors and save
- This allows using pre-calibrated robots without manual calibration
- If no calibration file exists, use factory default values (homing_offset=0, range_min=0, range_max=4095)

#### send_action[[lerobot.robots.omx_follower.OmxFollower.send_action]]

```python
send_action(action: dict)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/omx_follower/omx_follower.py#L190)

**Parameters:**

action (RobotAction) : The goal positions for the motors.

**Returns:** `RobotAction`

The action sent to the motors, potentially clipped.

Command arm to move to a target joint configuration.

The relative action magnitude may be clipped depending on the configuration parameter
`max_relative_target`. In this case, the action sent differs from original action.
Thus, this function always returns the action actually sent.

#### lerobot.robots.omx_follower.OmxFollowerConfig[[lerobot.robots.omx_follower.OmxFollowerConfig]]

```python
lerobot.robots.omx_follower.OmxFollowerConfig(port: str, disable_torque_on_disconnect: bool = True, max_relative_target: float | dict[str, float] | None = None, cameras: dict = <factory>, use_degrees: bool = False, id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/omx_follower/config_omx_follower.py#L24)

## Reachy2Robot[[lerobot.robots.reachy2.Reachy2Robot]]

#### lerobot.robots.reachy2.Reachy2Robot[[lerobot.robots.reachy2.Reachy2Robot]]

```python
lerobot.robots.reachy2.Reachy2Robot(config: Reachy2RobotConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/reachy2/robot_reachy2.py#L75)

[Reachy 2](https://www.pollen-robotics.com/reachy/), by Pollen Robotics.

#### lerobot.robots.reachy2.Reachy2RobotConfig[[lerobot.robots.reachy2.Reachy2RobotConfig]]

```python
lerobot.robots.reachy2.Reachy2RobotConfig(max_relative_target: float | None = None, ip_address: str | None = 'localhost', port: int = 50065, disable_torque_on_disconnect: bool = False, use_external_commands: bool = False, with_mobile_base: bool = True, with_l_arm: bool = True, with_r_arm: bool = True, with_neck: bool = True, with_antennas: bool = True, with_left_teleop_camera: bool = False, with_right_teleop_camera: bool = False, with_torso_camera: bool = False, camera_width: int = 640, camera_height: int = 480, cameras: dict = <factory>, id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/reachy2/configuration_reachy2.py#L25)

## UnitreeG1[[lerobot.robots.unitree_g1.UnitreeG1]]

#### lerobot.robots.unitree_g1.UnitreeG1[[lerobot.robots.unitree_g1.UnitreeG1]]

```python
lerobot.robots.unitree_g1.UnitreeG1(config: UnitreeG1Config)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/unitree_g1/unitree_g1.py#L155)

#### lerobot.robots.unitree_g1.UnitreeG1Config[[lerobot.robots.unitree_g1.UnitreeG1Config]]

```python
lerobot.robots.unitree_g1.UnitreeG1Config(kp: list[float] | None = None, kd: list[float] | None = None, default_positions: list = <factory>, control_dt: float = 0.004, is_simulation: bool = True, end_effector: G1EndEffector = <G1EndEffector.DEX1: 'dex1'>, sim_publish_images: bool = True, sim_camera_port: int = 5555, sim_onscreen: bool | None = None, robot_ip: str = '192.168.123.164', cameras: dict = <factory>, gravity_compensation: bool = False, controller: str | None = None, embodiment: str = 'g1_29', id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/unitree_g1/config_unitree_g1.py#L30)

## Hope Jr[[lerobot.robots.hope_jr.HopeJrArm]]

The Hope Jr humanoid is exposed as two independent robots, an arm and a hand.

#### lerobot.robots.hope_jr.HopeJrArm[[lerobot.robots.hope_jr.HopeJrArm]]

```python
lerobot.robots.hope_jr.HopeJrArm(config: HopeJrArmConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/hope_jr/hope_jr_arm.py#L34)

#### connect[[lerobot.robots.hope_jr.HopeJrArm.connect]]

```python
connect(calibrate: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/hope_jr/hope_jr_arm.py#L88)

We assume that at connection time, arm is in a rest position,
and torque can be safely disabled to run calibration.

#### lerobot.robots.hope_jr.HopeJrArmConfig[[lerobot.robots.hope_jr.HopeJrArmConfig]]

```python
lerobot.robots.hope_jr.HopeJrArmConfig(port: str, disable_torque_on_disconnect: bool = True, max_relative_target: float | dict[str, float] | None = None, cameras: dict = <factory>, id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/hope_jr/config_hope_jr.py#L42)

#### lerobot.robots.hope_jr.HopeJrHand[[lerobot.robots.hope_jr.HopeJrHand]]

```python
lerobot.robots.hope_jr.HopeJrHand(config: HopeJrHandConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/hope_jr/hope_jr_hand.py#L58)

#### lerobot.robots.hope_jr.HopeJrHandConfig[[lerobot.robots.hope_jr.HopeJrHandConfig]]

```python
lerobot.robots.hope_jr.HopeJrHandConfig(port: str, side: str, disable_torque_on_disconnect: bool = True, cameras: dict = <factory>, id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/hope_jr/config_hope_jr.py#L26)

## RebotB601Follower[[lerobot.robots.rebot_b601_follower.RebotB601Follower]]

#### lerobot.robots.rebot_b601_follower.RebotB601Follower[[lerobot.robots.rebot_b601_follower.RebotB601Follower]]

```python
lerobot.robots.rebot_b601_follower.RebotB601Follower(config: RebotB601FollowerRobotConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py#L62)

Seeed Studio reBot B601 follower arm (6-DOF + gripper, CAN motors).

Motor communication is handled by the `motorbridge` package over a CAN bus,
reached either through a Damiao serial bridge or MotorBridge's native CAN transport.

#### configure[[lerobot.robots.rebot_b601_follower.RebotB601Follower.configure]]

```python
configure()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py#L224)

Set each motor's control mode before enabling torque.

#### disable_torque[[lerobot.robots.rebot_b601_follower.RebotB601Follower.disable_torque]]

```python
disable_torque()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py#L245)

Disable motor torque so the arm can be moved by hand (read-only debugging).

#### disconnect[[lerobot.robots.rebot_b601_follower.RebotB601Follower.disconnect]]

```python
disconnect()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py#L444)

Disconnect motors and cameras from the robot.

#### send_action[[lerobot.robots.rebot_b601_follower.RebotB601Follower.send_action]]

```python
send_action(action: dict)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py#L313)

Command the arm to a target joint configuration.

Positions are expressed in degrees. The relative action magnitude may be
clipped depending on `max_relative_target`, so the action actually sent is
always returned.

#### lerobot.robots.rebot_b601_follower.RebotB601FollowerRobotConfig[[lerobot.robots.rebot_b601_follower.RebotB601FollowerRobotConfig]]

```python
lerobot.robots.rebot_b601_follower.RebotB601FollowerRobotConfig(port: str, motor_family: MotorFamily = <MotorFamily.DM: 'dm'>, can_adapter: str | None = None, dm_serial_baud: int = 921600, disable_torque_on_disconnect: bool = True, max_relative_target: float | dict[str, float] | None = None, cameras: dict = <factory>, motor_can_ids: dict[str, tuple[int, int]] | None = None, control_mode: ArmControlMode = <ArmControlMode.MIT: 'mit'>, gripper_control_mode: lerobot.robots.rebot_b601_follower.motor_family.GripperControlMode | None = None, mit_kp: float | dict[str, float] | None = None, mit_kd: float | dict[str, float] | None = None, pos_vel_velocity: float | dict[str, float] | None = None, gripper_torque_ratio: float | None = None, gripper_torque_limit: float | None = None, gripper_hold_torque_limit: float | None = None, joint_limits: dict[str, tuple[float, float]] | None = None, id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/rebot_b601_follower/config_rebot_b601_follower.py#L119)

Registered configuration for the reBot B601 follower robot.

## BiRebotB601Follower[[lerobot.robots.bi_rebot_b601_follower.BiRebotB601Follower]]

#### lerobot.robots.bi_rebot_b601_follower.BiRebotB601Follower[[lerobot.robots.bi_rebot_b601_follower.BiRebotB601Follower]]

```python
lerobot.robots.bi_rebot_b601_follower.BiRebotB601Follower(config: BiRebotB601FollowerConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/bi_rebot_b601_follower/bi_rebot_b601_follower.py#L28)

Bimanual Seeed Studio reBot B601 follower.

Composes two single-arm `RebotB601Follower` instances. Observation and
action keys of each arm are namespaced with a `left_` / `right_` prefix.

#### lerobot.robots.bi_rebot_b601_follower.BiRebotB601FollowerConfig[[lerobot.robots.bi_rebot_b601_follower.BiRebotB601FollowerConfig]]

```python
lerobot.robots.bi_rebot_b601_follower.BiRebotB601FollowerConfig(left_arm_config: RebotB601FollowerConfig, right_arm_config: RebotB601FollowerConfig, cameras: dict = <factory>, id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/bi_rebot_b601_follower/config_bi_rebot_b601_follower.py#L27)

Configuration class for the bimanual reBot B601 follower robot.

## EarthRoverMiniPlus[[lerobot.robots.earthrover_mini_plus.EarthRoverMiniPlus]]

#### lerobot.robots.earthrover_mini_plus.EarthRoverMiniPlus[[lerobot.robots.earthrover_mini_plus.EarthRoverMiniPlus]]

```python
lerobot.robots.earthrover_mini_plus.EarthRoverMiniPlus(config: EarthRoverMiniPlusConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/earthrover_mini_plus/robot_earthrover_mini_plus.py#L73)

**Parameters:**

config : Robot configuration

sdk_base_url : Base URL of the Frodobots SDK server, taken from `config.sdk_url` (default: http://localhost:8000)

EarthRover Mini Plus robot controlled via Frodobots SDK HTTP API.

This robot uses cloud-based control through the Frodobots SDK instead of direct
hardware connection. Cameras stream via WebRTC through Agora cloud, and control
commands are sent via HTTP POST requests.

The robot supports:
- Dual cameras (front and rear) accessed via SDK HTTP endpoints
- Linear and angular velocity control
- Battery and orientation telemetry

#### calibrate[[lerobot.robots.earthrover_mini_plus.EarthRoverMiniPlus.calibrate]]

```python
calibrate()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/earthrover_mini_plus/robot_earthrover_mini_plus.py#L153)

Calibration not needed for SDK-based robot.

#### configure[[lerobot.robots.earthrover_mini_plus.EarthRoverMiniPlus.configure]]

```python
configure()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/earthrover_mini_plus/robot_earthrover_mini_plus.py#L166)

Configure robot (no-op for SDK-based robot).

#### connect[[lerobot.robots.earthrover_mini_plus.EarthRoverMiniPlus.connect]]

```python
connect(calibrate: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/earthrover_mini_plus/robot_earthrover_mini_plus.py#L124)

**Parameters:**

calibrate : Not used for SDK-based robot (kept for API compatibility)

**Raises:** ``DeviceAlreadyConnectedError`` or ``DeviceNotConnectedError``

- ``DeviceAlreadyConnectedError`` -- If robot is already connected
- ``DeviceNotConnectedError`` -- If cannot connect to SDK server

Connect to robot via Frodobots SDK.

#### disconnect[[lerobot.robots.earthrover_mini_plus.EarthRoverMiniPlus.disconnect]]

```python
disconnect()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/earthrover_mini_plus/robot_earthrover_mini_plus.py#L366)

**Raises:** ``DeviceNotConnectedError``

- ``DeviceNotConnectedError`` -- If robot is not connected

Disconnect from robot.

Stops the robot and closes connection to SDK.

#### get_observation[[lerobot.robots.earthrover_mini_plus.EarthRoverMiniPlus.get_observation]]

```python
get_observation()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/earthrover_mini_plus/robot_earthrover_mini_plus.py#L248)

**Returns:** `RobotObservation`

Observation containing:
- front: Front camera image (480, 640, 3) in RGB format
- rear: Rear camera image (480, 640, 3) in RGB format
- speed: float - Current speed (raw SDK value)
- battery_level: float - Battery level (0-100)
- orientation: float - Robot orientation in degrees
- gps_latitude: float - GPS latitude coordinate
- gps_longitude: float - GPS longitude coordinate
- gps_signal: float - GPS signal strength (percentage)
- signal_level: float - Network signal level (0-5)
- vibration: float - Vibration sensor reading
- lamp: float - Lamp state (0=off, 1=on)
- accelerometer_x/y/z: float - Accelerometer axes (raw SDK value)
- gyroscope_x/y/z: float - Gyroscope axes (raw SDK value)
- magnetometer_filtered_x/y/z: float - Magnetometer axes (raw SDK value)
- wheel_rpm_0/1/2/3: float - Wheel RPMs

**Raises:** ``DeviceNotConnectedError``

- ``DeviceNotConnectedError`` -- If robot is not connected

Get current robot observation from SDK.

Camera frames are retrieved from SDK endpoints /v2/front and /v2/rear.
Frames are decoded from base64 and converted from BGR to RGB format.
Robot telemetry is retrieved from /data endpoint.
Sensor arrays (accels, gyros, mags, rpms) each contain entries of
[values..., timestamp]; the latest reading from each array is used.

Note:
Camera frames are retrieved from SDK endpoints /v2/front and /v2/rear.
Frames are decoded from base64 and converted from BGR to RGB format.
Robot telemetry is retrieved from /data endpoint.
All SDK values are normalized to appropriate ranges for dataset recording.

#### send_action[[lerobot.robots.earthrover_mini_plus.EarthRoverMiniPlus.send_action]]

```python
send_action(action: dict)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/earthrover_mini_plus/robot_earthrover_mini_plus.py#L334)

**Parameters:**

action : Action dict with keys: - linear_velocity: Target linear velocity (-1 to 1) - angular_velocity: Target angular velocity (-1 to 1)

**Returns:** `RobotAction`

The action that was sent (matches action_features keys)

**Raises:** ``DeviceNotConnectedError``

- ``DeviceNotConnectedError`` -- If robot is not connected

Send action to robot via SDK.

Note:
Actions are sent to SDK via POST /control endpoint.
SDK expects commands in range [-1, 1].

#### lerobot.robots.earthrover_mini_plus.EarthRoverMiniPlusConfig[[lerobot.robots.earthrover_mini_plus.EarthRoverMiniPlusConfig]]

```python
lerobot.robots.earthrover_mini_plus.EarthRoverMiniPlusConfig(sdk_url: str = 'http://localhost:8000', id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/earthrover_mini_plus/config_earthrover_mini_plus.py#L25)

**Parameters:**

sdk_url : URL of the Frodobots SDK server (default: http://localhost:8000)

Configuration for EarthRover Mini Plus robot using Frodobots SDK.

This robot uses cloud-based control via the Frodobots SDK HTTP API.
Camera frames are accessed directly through SDK HTTP endpoints.

### Datasets
https://huggingface.co/docs/lerobot/v0.7.0/api/datasets.md
