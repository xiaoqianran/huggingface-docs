# Third-Party Cameras & Sensors

The LeRobot ecosystem extends far beyond its natively supported cameras (OpenCV, Intel RealSense, ZMQ, Reachy 2). Thanks to LeRobot's plugin architecture, the community has built drop-in camera and sensor integrations — from depth cameras to vision-based tactile sensors. This page showcases community-maintained camera and sensor integrations you can use for teleoperation, data collection, and policy deployment.

> [!IMPORTANT]
> These projects are developed and maintained by third parties. Please refer to each repository for installation instructions, hardware requirements, and support.

Drop-in plugins are auto-discovered by package name: LeRobot imports any installed package prefixed with `lerobot_camera_`. Once installed, reference the camera `type` the plugin registers (see its README — it may differ from the package name) directly from any LeRobot command:

```bash
pip install lerobot_camera_<name>

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm \
    --robot.cameras="{ front: {type: <name>, width: 640, height: 480, fps: 30} }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=${HF_USER}/my-dataset \
    --dataset.single_task="Describe your task here" \
    --dataset.num_episodes=5
```

## Tactile Sensors

  
    
    
  
  
    ProjectDescription
  
  
    
      lerobot-camera-xense
      Plugin for Xense vision-based tactile sensors, exposing rectified/difference images, depth, and 2D markers.
    
    
      lerobot_camera_paxini
      Plugin for Paxini PX-6AX GEN3 multidimensional tactile sensors (3-axis force per taxel, 9-239 taxels), rendering the taxel array as force-vector or heatmap images.
    
  

## Depth Cameras

  
    
    
  
  
    ProjectDescription
  
  
    
      lerobot_camera_berxel
      Plugin for the Berxel depth camera, part of the broader HEXFELLOW driver suite.
    
    
      lerobot_camera_zed
      Official plugin for Stereolabs ZED cameras (ZED 2/2i/Mini, the ZED X family, and ZED X One), with metric depth, hardware timestamps, and SVO recording.
    
    
      lerobot_camera_oak
      Plugin for Luxonis OAK and OAK4 cameras over USB or PoE (DepthAI v3), with optional depth maps and point clouds.
    
  

## Networked Cameras

  
    
    
  
  
    ProjectDescription
  
  
    
      lerobot_camera_zmq
      Plugin streaming Stereolabs ZED and USB camera frames from a Raspberry Pi over the network.
    
  

## Virtual Cameras

  
    
    
  
  
    ProjectDescription
  
  
    
      lerobot_camera_dummy
      Plugin simulating a camera for recording without hardware. Useful for debugging !
    
  

## Contributing

Built your own LeRobot camera or sensor integration? Package it as an installable `lerobot_camera_<name>` plugin and it will be auto-discovered by the LeRobot CLI — see the [Bring Your Own Hardware](./integrate_hardware) guide and the [Cameras](./cameras) reference to get started, then share your project with the community!

### Annotation Pipeline
https://huggingface.co/docs/lerobot/v0.7.0/annotation_pipeline.md
