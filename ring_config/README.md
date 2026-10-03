# ring_config

The open_vins configuration of the RK3588_SLAM fisheye ring, generated from a
Kalibr camera-IMU calibration (`camchain-imucam.yaml`): `openvins_config.py`
(cameras, order, masks, noise; `python3 openvins_config.py --help`), the
estimator template it fills in, `openvins_estimator_template.yaml`, and the
Kalibr yaml reader it uses, `kalibr.py`.

One copy for both users, so that the board and the compute server run the
same VIO on the same data:

- the board's slam-web (RK3588_crossplatform_dev `ros/board/web`, which still
  carries its own copy until it imports this one);
- the compute server's map jobs (RK3588_SpatiaIntelli_Computing), which
  replay a board bag through open_vins before building a map.

Not a catkin package: plain Python 3 (PyYAML), imported by path.
