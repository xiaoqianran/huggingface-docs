# Cameras

Cameras supply the image observations a policy sees. Every backend — OpenCV, Intel RealSense, Reachy 2 —
implements the `Camera` interface, so swapping hardware does not change the code that reads frames.

See the [Cameras guide](../cameras) for choosing and configuring a camera, and
[Third-Party Cameras & Sensors](../third_party_sensors) for devices outside the core set.

## Camera[[lerobot.cameras.Camera]]

#### lerobot.cameras.Camera[[lerobot.cameras.Camera]]

```python
lerobot.cameras.Camera(config: CameraConfig)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/cameras/camera.py#L26)

**Parameters:**

fps (float | None) : Configured or measured frames per second

width (int | None) : Frame width in pixels

height (int | None) : Frame height in pixels

Base class for camera implementations.

Defines a standard interface for camera operations across different backends.
Subclasses must implement all abstract methods.

Manages basic camera properties (FPS, resolution) and core operations:
- Connection/disconnection
- Frame capture (sync/async/latest)

#### connect[[lerobot.cameras.Camera.connect]]

```python
connect(warmup: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/cameras/camera.py#L99)

**Parameters:**

warmup : If True (default), captures a warmup frame before returning. Useful for cameras that require time to adjust capture settings. If False, skips the warmup frame.

Establish connection to the camera.

#### disconnect[[lerobot.cameras.Camera.disconnect]]

```python
disconnect()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/cameras/camera.py#L181)

Disconnect from the camera and release resources.

#### read[[lerobot.cameras.Camera.read]]

```python
read()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/cameras/camera.py#L110)

**Returns:** `np.ndarray`

Captured frame as a numpy array.

Capture and return a single frame from the camera synchronously.

This is a blocking call that will wait for the hardware and its SDK.

#### async_read[[lerobot.cameras.Camera.async_read]]

```python
async_read(timeout_ms: float = Ellipsis)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/cameras/camera.py#L121)

**Parameters:**

timeout_ms : Maximum time to wait for a new frame in milliseconds. Defaults to 200ms (0.2s).

**Returns:** `np.ndarray`

Captured frame as a numpy array.

**Raises:** ``TimeoutError``

- ``TimeoutError`` -- If no new frame arrives within `timeout_ms`.

Return the most recent new frame.

This method retrieves the latest frame captured by the background thread.
If a new frame is already available in the buffer (captured since the last call),
it returns it immediately.

It blocks up to `timeout_ms` only if the buffer is empty or if the latest frame
was already consumed by a previous `async_read` call.

Essentially, this method return the latest unconsumed frame, waiting if necessary
for a new one to arrive within the specified timeout.

Usage:
- Ideal for control loops where you want to ensure every processed frame
is fresh, effectively synchronizing your loop to the camera's FPS.
- Causes of a timeout usually include: very low camera FPS, heavy processing load,
or if the camera is disconnected.

#### find_cameras[[lerobot.cameras.Camera.find_cameras]]

```python
find_cameras()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/cameras/camera.py#L89)

**Returns:** List[Dict[str, Any]]

A list of dictionaries,
where each dictionary contains information about a detected camera.

Detects available cameras connected to the system.

## CameraConfig[[lerobot.cameras.CameraConfig]]

#### lerobot.cameras.CameraConfig[[lerobot.cameras.CameraConfig]]

```python
lerobot.cameras.CameraConfig(fps: int | None = None, width: int | None = None, height: int | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/cameras/configs.py#L60)

## make_cameras_from_configs[[lerobot.cameras.make_cameras_from_configs]]

#### lerobot.cameras.make_cameras_from_configs[[lerobot.cameras.make_cameras_from_configs]]

```python
lerobot.cameras.make_cameras_from_configs(camera_configs: dict)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/cameras/utils.py#L25)
