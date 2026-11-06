# Teleoperators

A teleoperator produces actions for a robot to follow — a leader arm, a gamepad, a keyboard, a phone. All of
them implement the `Teleoperator` interface, so a recording script written against it works with any input
device.

See [Phone teleoperation](../phone_teleop) and [Isaac Teleop](../isaac_teleop) for setup guides, and
[Imitation Learning for Robots](../il_robots) for the recording workflow. For community-maintained devices
outside the core set, see [Third-Party Robots & Teleoperators](../third_party_robots). To add a teleoperator
of your own, see [Bring Your Own Hardware](../integrate_hardware).

## Teleoperator[[lerobot.teleoperators.Teleoperator]]

#### lerobot.teleoperators.Teleoperator[[lerobot.teleoperators.Teleoperator]]

```python
lerobot.teleoperators.Teleoperator(config: TeleoperatorConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/teleoperators/teleoperator.py#L32)

**Parameters:**

config_class (RobotConfig) : The expected configuration class for this teleoperator.

name (str) : The unique name used to identify this teleoperator type.

The base abstract class for all LeRobot-compatible teleoperation devices.

This class provides a standardized interface for interacting with physical teleoperators.
Subclasses must implement all abstract methods and properties to be usable.

#### connect[[lerobot.teleoperators.Teleoperator.connect]]

```python
connect(calibrate: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/teleoperators/teleoperator.py#L125)

**Parameters:**

calibrate (bool) : If True, automatically calibrate the teleoperator after connecting if it's not calibrated or needs calibration (this is hardware-dependant).

Establish communication with the teleoperator.

#### disconnect[[lerobot.teleoperators.Teleoperator.disconnect]]

```python
disconnect()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/teleoperators/teleoperator.py#L224)

Disconnect from the teleoperator and perform any necessary cleanup.

#### configure[[lerobot.teleoperators.Teleoperator.configure]]

```python
configure()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/teleoperators/teleoperator.py#L190)

Apply any one-time or runtime configuration to the teleoperator.
This may include setting motor parameters, control modes, or initial state.

#### calibrate[[lerobot.teleoperators.Teleoperator.calibrate]]

```python
calibrate()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/teleoperators/teleoperator.py#L142)

Calibrate the teleoperator if applicable. If not, this should be a no-op.

This method should collect any necessary data (e.g., motor offsets) and update the
:pyattr:`calibration` dictionary accordingly.

#### get_action[[lerobot.teleoperators.Teleoperator.get_action]]

```python
get_action()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/teleoperators/teleoperator.py#L198)

**Returns:** `RobotAction`

A flat dictionary representing the teleoperator's current actions. Its
structure should match :pymeth:`observation_features`.

Retrieve the current action from the teleoperator.

#### send_feedback[[lerobot.teleoperators.Teleoperator.send_feedback]]

```python
send_feedback(feedback: dict)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/teleoperators/teleoperator.py#L209)

**Parameters:**

feedback (dict[str, Any]) : Dictionary representing the desired feedback. Its structure should match :pymeth:`feedback_features`.

**Returns:** dict[str, Any]

The action actually sent to the motors potentially clipped or modified, e.g. by
safety limits on velocity.

Send a feedback action command to the teleoperator.

#### action_features[[lerobot.teleoperators.Teleoperator.action_features]]

```python
action_features()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/teleoperators/teleoperator.py#L90)

A dictionary describing the structure and types of the actions produced by the teleoperator. Its
structure (keys) should match the structure of what is returned by :pymeth:`get_action`. Values for
the dict should be the type of the value if it's a simple value, e.g. `float` for single
proprioceptive value (a joint's goal position/velocity)

Note: this property should be able to be called regardless of whether the robot is connected or not.

#### feedback_features[[lerobot.teleoperators.Teleoperator.feedback_features]]

```python
feedback_features()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/teleoperators/teleoperator.py#L103)

A dictionary describing the structure and types of the feedback actions expected by the robot. Its
structure (keys) should match the structure of what is passed to :pymeth:`send_feedback`. Values for
the dict should be the type of the value if it's a simple value, e.g. `float` for single
proprioceptive value (a joint's goal position/velocity)

Note: this property should be able to be called regardless of whether the robot is connected or not.

#### is_connected[[lerobot.teleoperators.Teleoperator.is_connected]]

```python
is_connected()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/teleoperators/teleoperator.py#L116)

Whether the teleoperator is currently connected or not. If `False`, calling :pymeth:`get_action`
or :pymeth:`send_feedback` should raise an error.

#### is_calibrated[[lerobot.teleoperators.Teleoperator.is_calibrated]]

```python
is_calibrated()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/teleoperators/teleoperator.py#L136)

Whether the teleoperator is currently calibrated or not. Should be always `True` if not applicable

## TeleoperatorConfig[[lerobot.teleoperators.TeleoperatorConfig]]

#### lerobot.teleoperators.TeleoperatorConfig[[lerobot.teleoperators.TeleoperatorConfig]]

```python
lerobot.teleoperators.TeleoperatorConfig(id: str | None = None, calibration_dir: pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/teleoperators/config.py#L23)

## make_teleoperator_from_config[[lerobot.teleoperators.make_teleoperator_from_config]]

#### lerobot.teleoperators.make_teleoperator_from_config[[lerobot.teleoperators.make_teleoperator_from_config]]

```python
lerobot.teleoperators.make_teleoperator_from_config(config: TeleoperatorConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/teleoperators/utils.py#L36)

### Cameras
https://huggingface.co/docs/lerobot/v0.7.0/api/cameras.md
