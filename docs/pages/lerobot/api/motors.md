# Motors

`SerialMotorsBus` is the low-level interface to a chain of Feetech or Dynamixel servos on a serial bus. Robots use it
to read positions and write goal positions; you rarely touch it directly unless you are adding hardware. Each motor's
model is looked up in [rustypot](https://github.com/pollen-robotics/rustypot), which holds the control tables and the
facts that differ between families, and also owns the serial port. `FeetechMotorsBus` and `DynamixelMotorsBus` are
deprecated aliases of it, kept for one release.

### Migrating from the SDK-based buses

The aliases keep the class names, not the old SDK attributes and model tables:

- `bus.model_resolution_table[model]` becomes `bus.resolution(model)`; the other model tables (`model_ctrl_table`,
  `model_number_table`, ...) are now rustypot's servo definitions.
- Operating modes are selected by name, `bus.set_operating_mode("position")`, instead of writing an `OperatingMode`
  value to the `Operating_Mode` register.
- The SDK port and packet handlers are no longer exposed: rustypot owns the serial port.
- `protocol_version` is ignored: each motor's model gives its protocol.

See [Bring Your Own Hardware](../integrate_hardware) for adding a new bus, and
[Updating Feetech Firmware](../feetech) and [Damiao Motors and CAN Bus](../damiao) for device-specific notes.

## SerialMotorsBus[[lerobot.motors.SerialMotorsBus]]

#### lerobot.motors.SerialMotorsBus[[lerobot.motors.SerialMotorsBus]]

```python
lerobot.motors.SerialMotorsBus(port: str, motors: dict[str, Motor], calibration: dict[str, MotorCalibration] | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L156)

Read and write a chain of servos daisy-chained on one serial port, Feetech or Dynamixel.

rustypot handles each model's metadata (control table, resolution, baud rates, operating
modes) and the serial I/O; LeRobot handles normalisation, calibration and the recommended
settings. Motors of several models, such as STS and SCS or XL430 and XL330, share the port.

To find the port, run:
```bash
lerobot-find-port
>>> Finding all available ports for the MotorsBus.
>>> ["/dev/tty.usbmodem575E0032081", "/dev/tty.usbmodem575E0031751"]
>>> Remove the usb cable from your MotorsBus and press Enter when done.
>>> The port of this MotorsBus is /dev/tty.usbmodem575E0031751.
>>> Reconnect the usb cable.
```

Example for a single Feetech sts3215 on the bus:
```python
from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

bus = SerialMotorsBus(
    port="/dev/tty.usbmodem575E0031751",
    motors={"my_motor": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)},
)
bus.connect()

position = bus.read("Present_Position", "my_motor", normalize=False)

# Move a few motor steps as an example
bus.write("Goal_Position", "my_motor", position + 30, normalize=False)

bus.disconnect()
```

#### configure_motors[[lerobot.motors.SerialMotorsBus.configure_motors]]

```python
configure_motors(return_delay_time: int = 0, maximum_acceleration: int = 254, acceleration: int = 254)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L572)

**Parameters:**

return_delay_time (`int`, *optional*, defaults to `0`) : Delay before a motor answers, in units of 2 µs; the motors leave the factory at 250 (500 µs).

maximum_acceleration (`int`, *optional*, defaults to `254`) : Acceleration ceiling, on the servos that have one (Feetech STS), high to speed up acceleration and deceleration.

acceleration (`int`, *optional*, defaults to `254`) : Acceleration, on the servos that have it (Feetech).

Write LeRobot's recommended settings to every motor.

#### connect[[lerobot.motors.SerialMotorsBus.connect]]

```python
connect(handshake: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L396)

**Parameters:**

handshake (`bool`, *optional*, defaults to `True`) : Whether to check that every motor answers with the model number of its model, and that the Feetech motors of one definition run one firmware.

**Raises:** ``DeviceAlreadyConnectedError`` or ``ConnectionError`` or ``RuntimeError``

- ``DeviceAlreadyConnectedError`` -- If the port is already open.
- ``ConnectionError`` -- If the port could not be opened.
- ``RuntimeError`` -- If the handshake fails; the port is closed again.

Open the serial port and check the motors.

#### disable_torque[[lerobot.motors.SerialMotorsBus.disable_torque]]

```python
disable_torque(motors: NameOrID | Sequence[NameOrID] | None = None, num_retry: int = 0)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L650)

**Parameters:**

motors (`str | int | Sequence[str | int]`, *optional*) : Motors to free, by name or id. `None` selects every motor.

num_retry (`int`, *optional*, defaults to `0`) : Additional attempts on each motor after a communication failure.

**Raises:** ``ConnectionError``

- ``ConnectionError`` -- If some motors could not be reached; the message names them.

Disable the torque of the selected motors.

With the torque off, a motor's EEPROM can be written, and a Feetech motor's lock opens
for it. Every selected motor is attempted; failures are reported, and unresponsive
motors may remain torque-enabled.

#### disconnect[[lerobot.motors.SerialMotorsBus.disconnect]]

```python
disconnect(disable_torque: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L440)

**Parameters:**

disable_torque (`bool`, *optional*, defaults to `True`) : Whether to disable the torque of every motor before closing the port, so that no motor is left resisting.

Close the serial port, by default with the torque of every motor off.

Safe to call on a bus that is already disconnected, and the port is released even if
disabling the torque fails.

#### enable_torque[[lerobot.motors.SerialMotorsBus.enable_torque]]

```python
enable_torque(motors: NameOrID | Sequence[NameOrID] | None = None, num_retry: int = 0)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L669)

**Parameters:**

motors (`str | int | Sequence[str | int]`, *optional*) : Motors to hold, by name or id. `None` selects every motor.

num_retry (`int`, *optional*, defaults to `0`) : Additional attempts on each motor after a communication failure.

**Raises:** ``ConnectionError``

- ``ConnectionError`` -- If some motors could not be reached; the message names them.

Enable the torque of the selected motors, and close the lock of Feetech motors.

#### ping[[lerobot.motors.SerialMotorsBus.ping]]

```python
ping(motor: NameOrID, num_retry: int = 0)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L1069)

**Parameters:**

motor (`str | int`) : Motor, by name or id, one of the bus' motors.

num_retry (`int`, *optional*, defaults to `0`) : Additional attempts after a communication failure.

**Returns:** `int | None`

The model number, `None` when the motor does not answer or refuses the read.

The model number a motor answers with.

Reads `Model_Number` rather than sending a ping: presence and identity then cost one
round trip instead of two.

#### read[[lerobot.motors.SerialMotorsBus.read]]

```python
read(data_name: str, motor: str, normalize: bool = True, num_retry: int = 0)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L1090)

**Parameters:**

data_name (`str`) : Register name, e.g. `"Present_Position"`; the case does not matter.

motor (`str`) : Motor name.

normalize (`bool`, *optional*, defaults to `True`) : Whether to scale a position to the range its calibration and norm mode define.

num_retry (`int`, *optional*, defaults to `0`) : Additional attempts after a communication failure.

**Returns:** `int | float`

The raw value, or the normalised one.

**Raises:** ``ConnectionError`` or ``RuntimeError``

- ``ConnectionError`` -- If the motor does not answer.
- ``RuntimeError`` -- If the motor refuses the read. An alarm in its status, as an
  overload, does not raise.

Read a register of one motor.

#### read_calibration[[lerobot.motors.SerialMotorsBus.read_calibration]]

```python
read_calibration()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L744)

**Returns:** `dict[str, MotorCalibration]`

The calibration of each motor, by name.

Read the calibration of every motor.

A motor without a homing offset register (Feetech SCS) reports 0, and one without a
`Drive_Mode` register (Feetech) reports drive mode 0.

#### record_calibration[[lerobot.motors.SerialMotorsBus.record_calibration]]

```python
record_calibration(mode: str, full_turn_motors: Sequence[str], inverted_motors: Sequence[str] = (), motors: Sequence[str] | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L931)

**Parameters:**

mode (`str`) : Operating mode of the motors, e.g. `"position"`.

full_turn_motors (`Sequence[str]`) : Motors given the whole encoder range instead of the range they are moved through.

inverted_motors (`Sequence[str]`, *optional*, defaults to `()`) : Motors turning the other way.

motors (`Sequence[str]`, *optional*) : Motors to calibrate. `None` calibrates every motor.

**Returns:** `dict[str, MotorCalibration]`

The calibration of each motor, in the order of the bus.

**Raises:** ``ValueError``

- ``ValueError`` -- If a motor is not on the bus, or a full-turn or inverted motor is not one
  of the motors to calibrate; no motor has been touched then.

Calibrate motors moved by hand.

The selected motors are freed, put in an operating mode and given their direction: a
motor with a `Drive_Mode` register gets it written, inverted or not, while on the others,
as Feetech servos, an inversion is applied in software when normalising. The user then
holds the joints in the middle of their range of motion, which becomes a half turn of
each motor, and moves each joint through its whole range, recorded until ENTER. A
full-turn motor is not recorded: it spans the whole encoder of its model.

The motors need the operating mode and a homing offset, which the Feetech SCS0009 has
neither. Their homing offsets and position limits are rewritten, the cached
`calibration` is cleared, and their torque is left off. The calibration returned is not
written to the position limits nor saved: `~SerialMotorsBus.write_calibration` writes
it, and the device saves it.

#### record_ranges_of_motion[[lerobot.motors.SerialMotorsBus.record_ranges_of_motion]]

```python
record_ranges_of_motion(motors: NameOrID | Sequence[NameOrID] | None = None, display_values: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L876)

**Parameters:**

motors (`str | int | Sequence[str | int]`, *optional*) : Motors to record, by name or id. `None` selects every motor.

display_values (`bool`, *optional*, defaults to `True`) : Whether to print the live table.

**Returns:** `tuple[dict[str, int | float], dict[str, int | float]]`

The lowest and the highest
raw position of each motor, by name.

**Raises:** ``ValueError``

- ``ValueError`` -- If a motor did not move: its lowest and highest positions are equal.

Record the range each selected motor is moved through by hand, until ENTER.

The joints are moved by hand, with the torque off, while a live table shows the
positions.

#### reset_calibration[[lerobot.motors.SerialMotorsBus.reset_calibration]]

```python
reset_calibration(motors: NameOrID | Sequence[NameOrID] | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L799)

**Parameters:**

motors (`str | int | Sequence[str | int]`, *optional*) : Motors to reset, by name or id. `None` selects every motor.

Restore the factory calibration of the selected motors.

The homing offset is set to 0 on the motors that have one, and the position limits span
the whole encoder. The cached `calibration` is cleared.

#### resolution[[lerobot.motors.SerialMotorsBus.resolution]]

```python
resolution(model: str)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L268)

**Parameters:**

model (`str`) : Model of one of the bus' motors, e.g. `"sts3215"`.

**Returns:** `int`

The steps per turn, e.g. 4096 on a 12-bit encoder.

**Raises:** ``ValueError``

- ``ValueError`` -- If the model does not count encoder steps.

Encoder steps per turn of a motor model.

#### set_baudrate[[lerobot.motors.SerialMotorsBus.set_baudrate]]

```python
set_baudrate(baudrate: int)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L707)

**Parameters:**

baudrate (`int`) : Baud rate, in bits per second.

Change the baud rate of the open port.

The port opens at `default_baudrate` on every connect.

#### set_half_turn_homings[[lerobot.motors.SerialMotorsBus.set_half_turn_homings]]

```python
set_half_turn_homings(motors: NameOrID | Sequence[NameOrID] | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L820)

**Parameters:**

motors (`str | int | Sequence[str | int]`, *optional*) : Motors to home, by name or id. `None` selects every motor.

**Returns:** `dict[str, int]`

The homing offset written to each motor, by name.

**Raises:** ``ValueError``

- ``ValueError`` -- If a selected motor has no homing offset, as the Feetech SCS.

Make the present position of each selected motor a half turn.

The calibration of the motors is reset, then each gets the homing offset that brings its
present position to half its encoder, e.g. 2047 on a 12-bit one. Every selected motor
must have a homing offset, which is checked before any register is written.

#### set_operating_mode[[lerobot.motors.SerialMotorsBus.set_operating_mode]]

```python
set_operating_mode(mode: str, motors: NameOrID | Sequence[NameOrID] | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L600)

**Parameters:**

mode (`str`) : Name of the operating mode, e.g. `"position"`.

motors (`str | int | Sequence[str | int]`, *optional*) : Motors to set, by name or id. `None` selects every motor.

**Raises:** ``ValueError``

- ``ValueError`` -- If a selected motor has no such mode; no motor has changed mode then.

Put the selected motors in an operating mode, given by name.

The modes come from each motor's definition, so they depend on the model, not only on
the family: the Feetech SCS0009 has none. The mode must exist for every selected motor,
which is checked before any motor changes mode. Write it with the torque off.

- `position`: holds a goal position within one turn and the position limits.
- `velocity`: turns endlessly at a goal velocity, as a wheel.
- `pwm`: drives the motor voltage directly, open loop.
- `step` (Feetech): moves by the number of steps written as the goal position.
- `current` (Dynamixel): holds a goal current, that is a torque, whatever the position.
- `extended_position` (Dynamixel): holds a goal position over several turns. The
  `Min_Position_Limit` and `Max_Position_Limit` registers are ignored in this mode.
- `current_based_position` (Dynamixel): holds a goal position over several turns with
  the current capped by the goal current, as a gripper that must not crush what it holds.

#### setup_motor[[lerobot.motors.SerialMotorsBus.setup_motor]]

```python
setup_motor(motor: str, initial_baudrate: int | None = None, initial_id: int | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L471)

**Parameters:**

motor (`str`) : Name of the motor.

initial_baudrate (`int`, *optional*) : Baud rate the motor answers at now.

initial_id (`int`, *optional*) : Id the motor answers at now, used with `initial_baudrate`.

**Raises:** ``RuntimeError`` or ``ConnectionError``

- ``RuntimeError`` -- If the motor is not found, is not of its model, or does not answer as
  configured once set up.
- ``ConnectionError`` -- If communication with the motor fails.

Give a single motor its id and the bus' default baud rate.

The motor must be the only one on the bus. Unless both `initial_baudrate` and
`initial_id` are given, it is found first: at `initial_baudrate` when only that is
given, else across the baud rates of its model; an `initial_id` given alone is replaced
by the id found. rustypot then turns its torque off, opens its EEPROM, and writes the id
and the baud rate.

#### setup_motors[[lerobot.motors.SerialMotorsBus.setup_motors]]

```python
setup_motors(motors: Sequence[str] | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L530)

**Parameters:**

motors (`Sequence[str]`, *optional*) : Motors in the order they are set up. `None` goes from the last motor to the first, from the gripper down on an arm.

Set up the motors one at a time, asking for each to be alone on the bus.

#### sync_read[[lerobot.motors.SerialMotorsBus.sync_read]]

```python
sync_read(data_name: str, motors: NameOrID | Sequence[NameOrID] | None = None, normalize: bool = True, num_retry: int = 0)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L1193)

**Parameters:**

data_name (`str`) : Register name, e.g. `"Present_Position"`; the case does not matter.

motors (`str | int | Sequence[str | int]`, *optional*) : Motors to read, by name or id. `None` reads every motor.

normalize (`bool`, *optional*, defaults to `True`) : Whether to scale positions to the range their calibration and norm mode define.

num_retry (`int`, *optional*, defaults to `0`) : Additional attempts after a communication failure. A retry sends every group again, those that had succeeded included; an unknown motor or register is never retried.

**Returns:** `dict[str, int | float]`

The value of each motor, by name.

**Raises:** ``ConnectionError`` or ``RuntimeError``

- ``ConnectionError`` -- If a motor does not answer.
- ``RuntimeError`` -- If a motor refuses the read. An alarm in a status, as an overload,
  does not raise.

Read a register of several motors.

rustypot sends one Sync Read per group of motors sharing a protocol and the register's
address and size, and reads the motors of a servo without Sync Read, as the Feetech SCS,
one after the other. The values are therefore not one simultaneous snapshot.

#### sync_write[[lerobot.motors.SerialMotorsBus.sync_write]]

```python
sync_write(data_name: str, values: Value | dict[str, Value], normalize: bool = True, num_retry: int = 0)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L1253)

**Parameters:**

data_name (`str`) : Register name, e.g. `"Goal_Position"`; the case does not matter.

values (`int | float | dict[str, int | float]`) : One value for every motor, or a value per motor name.

normalize (`bool`, *optional*, defaults to `True`) : Whether positions are given in the range their calibration and norm mode define.

num_retry (`int`, *optional*, defaults to `0`) : Additional attempts after a failure of the port. A retry sends every group again.

**Raises:** ``ConnectionError``

- ``ConnectionError`` -- If the port fails.

Write a register of several motors, without waiting for an answer.

Unlike `~SerialMotorsBus.write`, no motor answers a sync write: faster, suited to
control loops, but a lost packet goes unnoticed. rustypot sends one Sync Write per group
of motors sharing a protocol and the register's address and size.

#### torque_disabled[[lerobot.motors.SerialMotorsBus.torque_disabled]]

```python
torque_disabled(motors: NameOrID | Sequence[NameOrID] | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L684)

**Parameters:**

motors (`str | int | Sequence[str | int]`, *optional*) : Motors to free, by name or id. `None` selects every motor.

Disable the torque of the selected motors for the duration of a `with` block.

The torque is enabled again on exit, even when the block raises, which suits configuring
motors.

Example:
```python
>>> with bus.torque_disabled():
...     bus.set_operating_mode("position")
```

#### write[[lerobot.motors.SerialMotorsBus.write]]

```python
write(data_name: str, motor: str, value: Value, normalize: bool = True, num_retry: int = 0)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L1146)

**Parameters:**

data_name (`str`) : Register name, e.g. `"Goal_Position"`; the case does not matter.

motor (`str`) : Motor name.

value (`int | float`) : Value to write, normalised or raw as `normalize` says.

normalize (`bool`, *optional*, defaults to `True`) : Whether a position is given in the range its calibration and norm mode define.

num_retry (`int`, *optional*, defaults to `0`) : Additional attempts after a communication failure.

**Raises:** ``ConnectionError`` or ``RuntimeError``

- ``ConnectionError`` -- If the motor does not answer.
- ``RuntimeError`` -- If the motor refuses the write. An alarm in its status, as an
  overload, does not raise.

Write a register of one motor, and wait for its answer.

Unlike `~SerialMotorsBus.sync_write`, the motor acknowledges the write, so a failure is
reported: slower, but suited to configuring motors.

#### write_calibration[[lerobot.motors.SerialMotorsBus.write_calibration]]

```python
write_calibration(calibration_dict: dict[str, MotorCalibration], cache: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L774)

**Parameters:**

calibration_dict (`dict[str, MotorCalibration]`) : Calibration of each motor, by name, as `~SerialMotorsBus.read_calibration` returns it.

cache (`bool`, *optional*, defaults to `True`) : Whether to keep the calibration as the bus' `calibration`, which normalises the positions.

Write a calibration to the motors, and by default cache it.

The position limits are written to every motor, and the homing offset to the motors
that have one. Neither the id nor the drive mode is written: a Feetech's drive mode is
applied in software when normalising, and a Dynamixel's `Drive_Mode` register is set on
its own.

## Motor[[lerobot.motors.Motor]]

#### lerobot.motors.Motor[[lerobot.motors.Motor]]

```python
lerobot.motors.Motor(id: int, model: str, norm_mode: MotorNormMode, motor_type_str: str | None = None, recv_id: int | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L148)

## MotorCalibration[[lerobot.motors.MotorCalibration]]

#### lerobot.motors.MotorCalibration[[lerobot.motors.MotorCalibration]]

```python
lerobot.motors.MotorCalibration(id: int, drive_mode: int, homing_offset: int, range_min: int, range_max: int)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L139)

## MotorNormMode[[lerobot.motors.MotorNormMode]]

#### lerobot.motors.MotorNormMode[[lerobot.motors.MotorNormMode]]

```python
lerobot.motors.MotorNormMode(*values)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L125)

## DriveMode[[lerobot.motors.DriveMode]]

`DriveMode` has two members, `NON_INVERTED = 0` and `INVERTED = 1`. Use `.value` where the integer is expected, in a
`Drive_Mode` register or in `MotorCalibration.drive_mode`:

```python
bus.write("Drive_Mode", "elbow_flex", DriveMode.INVERTED.value)
```

#### lerobot.motors.DriveMode[[lerobot.motors.DriveMode]]

```python
lerobot.motors.DriveMode(*values)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L131)

Value of a Dynamixel `Drive_Mode` register, and of `MotorCalibration.drive_mode`.

### Policies
https://huggingface.co/docs/lerobot/v0.7.0/api/policies.md
