#!/usr/bin/env python3
"""Kalibr camera-IMU result -> an open_vins config directory, with masks.

Used by slam-web on the board (SLAM start, from the calibration imported on
its calibration page) and on the dev PC through
ros/pc/src/slam_pc/scripts/kalibr2openvins.py. Standard library + PyYAML
only: the board has neither numpy nor OpenCV for sure, nor open_vins' source.

    python3 openvins_config.py --camchain camchain-imucam.yaml --out DIR
        [--radius 330] [--topic-suffix image] [--imu-noise imu.yaml] ...

Writes into DIR:
  estimator_config.yaml     openvins_estimator_template.yaml with our values
                            substituted. The template is a verbatim copy of
                            ros/src/open_vins/config/euroc_mav/estimator_config.yaml
                            (fork commit 78987b1): open_vins exits on a missing
                            key, so copy it again when the fork adds parameters
  kalibr_imucam_chain.yaml  per camera: T_imu_cam (the inverse of Kalibr's
                            T_cam_imu, written out so nothing depends on the
                            parser's fallback), intrinsics, equidistant
                            distortion, resolution, raw image topic, time shift
  kalibr_imu_chain.yaml     IMU noise: Kalibr's imu.yaml when given, else the
                            SCH16T-K10 datasheet (see IMU_NOISE)
  mask_cnN.png              white (255) = no features: everything further than
                            --radius px from the image centre -- the same cut
                            the intrinsics were calibrated with, so no feature
                            relies on the extrapolated lens model, and none
                            gets near 90 deg off-axis where open_vins'
                            normalized-plane math (tan theta) blows up

Only cameras with an equidistant model and a T_cam_imu are taken: the
board's camchain-imucam.yaml lists every connector, uncalibrated ones as
all-zero templates. They are ordered around the ring by the azimuth of their
optical axes in the IMU frame (for ours cn2 -> cn4 -> cn5 -> cn3), because
open_vins' ring stereo (stereo='ring', the default: use_stereo with more than
2 cameras) pairs cam i with cam i+1. All cameras of one frame must carry the
same stamp within 1 ms (the board's trigger-grid stamps agree to < 1 us).
Image topics become /cnN/<suffix>: image_raw on the board; on the PC, image,
which slam_pc/republish.launch decompresses the bags' compressed topics into
(open_vins takes sensor_msgs/Image only).
"""
import argparse
import math
import os
import re
import struct
import zlib

import kalibr

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, 'openvins_estimator_template.yaml')

# Murata SCH16T-K10 datasheet, typical values:
#   gyro   noise density 0.006 (deg/s)/sqrt(Hz), ARW 0.26 deg/sqrt(h), bias instability 2 deg/h
#   accel  noise density 0.8 (mm/s^2)/sqrt(Hz),  VRW 30 (mm/s)/sqrt(h), bias instability 0.15 (typ) - 0.3 mm/s^2
# The noise densities convert directly. The bias random walks are not in the
# datasheet: taken from the bias instability as the rate-random-walk line
# through the Allan minimum (0.664 x BI) at tau = 100 s, K = sigma_min sqrt(3/tau)
# -- an assumption (a later minimum would mean a smaller walk), to be replaced
# by an Allan variance of our own unit. VIO usually wants these inflated
# (x5-x10) for vibration and temperature; they are left as the datasheet says.
TAU = 100.0
IMU_NOISE = {
    'gyroscope_noise_density': math.radians(0.006),                         # rad/s/sqrt(Hz)
    'accelerometer_noise_density': 0.8e-3,                                   # m/s^2/sqrt(Hz)
    'gyroscope_random_walk': 0.664 * math.radians(2.0) / 3600 * math.sqrt(3 / TAU),   # rad/s^2/sqrt(Hz)
    'accelerometer_random_walk': 0.664 * 0.15e-3 * math.sqrt(3 / TAU),      # m/s^3/sqrt(Hz)
}
DATASHEET_NOTE = f"""# Murata SCH16T-K10, noise from the DATASHEET (typical values), not from our own
# Allan variance yet: densities converted directly, random walks derived from
# the bias instability (rate random walk through the Allan minimum at tau =
# {TAU:.0f} s). Replace with an Allan variance of a >= 2 h static recording; for VIO
# expect to inflate them x5-x10 (vibration, temperature).
#   gyro  0.006 (deg/s)/sqrt(Hz), bias instability 2 deg/h
#   accel 0.8 (mm/s^2)/sqrt(Hz),  bias instability 0.15 mm/s^2 (typ)"""


def write_mask(path, w, h, radius):
    """8-bit grayscale PNG: 255 further than radius from ((w-1)/2, (h-1)/2), else 0."""
    cx, cy = (w - 1) / 2, (h - 1) / 2
    rows = []
    for y in range(h):
        dy = y - cy
        if abs(dy) > radius:
            rows.append(b'\0' + b'\xff' * w)
            continue
        half = math.sqrt(radius * radius - dy * dy)
        lo, hi = max(0, math.ceil(cx - half)), min(w - 1, math.floor(cx + half))
        rows.append(b'\0' + b'\xff' * lo + b'\0' * (hi - lo + 1) + b'\xff' * (w - hi - 1))

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)) +
                chunk(b'IDAT', zlib.compress(b''.join(rows), 9)) + chunk(b'IEND', b''))


def ring_cameras(chain):
    """Cameras of a camchain-imucam dict usable by open_vins, in ring order,
    and notes on the ones skipped."""
    cams, notes = [], []
    for key in sorted((k for k in chain if re.match(r'^cam\d+$', k)), key=lambda k: int(k[3:])):
        c = chain[key]
        m = re.match(r'^/?(cn\d)', str(c.get('rostopic', '')))
        name = m.group(1) if m else key
        T = kalibr._mat4(c.get('T_cam_imu'))
        if not kalibr.mat4_set(T):
            notes.append(f'{name}: no T_cam_imu (not calibrated), left out')
            continue
        if c.get('distortion_model') != 'equidistant':
            notes.append(f'{name}: {c.get("distortion_model")} distortion, open_vins config wants equidistant, left out')
            continue
        Ti = kalibr._inv(T)
        axis = [Ti[0][2], Ti[1][2], Ti[2][2]]          # optical axis (camera z) in the IMU frame
        cams.append({'name': name, 'c': c, 'T_imu_cam': Ti, 'az': math.degrees(math.atan2(axis[1], axis[0]))})
    cams.sort(key=lambda x: x['az'])
    return cams, notes


def generate(camchain, out, radius=330.0, topic_suffix='image', imu_topic='/imu/data', imu_rate=480.0,
             gravity=9.781, track_hz=20.0, stereo='ring', pts=100, init_pts=50, imu_noise=None,
             template=TEMPLATE, source=None, timing=None):
    """Write the config directory; returns a summary dict. imu_noise: the four
    Kalibr noise values (else the datasheet's). timing: a file for open_vins'
    per-step timing (one CSV line per frame), else none. Raises ValueError on
    input it cannot use."""
    with open(camchain) as f:
        chain = kalibr.load_yaml(f.read())
    cams, notes = ring_cameras(chain)
    if not cams:
        raise ValueError('no camera with an equidistant model and a T_cam_imu: ' + '; '.join(notes))
    os.makedirs(out, exist_ok=True)

    # ---- kalibr_imucam_chain.yaml
    lines = ['%YAML:1.0  # open_vins wants this first line',
             f'# generated by openvins_config.py from {source or os.path.abspath(camchain)}',
             '# cameras in ring order (optical-axis azimuth in the IMU frame): ' +
             ' -> '.join(f'{x["name"]} ({x["az"]:.0f} deg)' for x in cams), '']
    shifts = []
    for i, x in enumerate(cams):
        c = x['c']
        shifts.append(float(c.get('timeshift_cam_imu') or 0.0))
        lines += [f'cam{i}:  # {x["name"]}',
                  '  T_imu_cam:  # camera -> IMU: inverse of Kalibr\'s T_cam_imu (R_CtoI, p_CinI)']
        lines += ['    - [' + ', '.join(f'{v:.9f}' for v in row) + ']' for row in x['T_imu_cam']]
        lines += ['  cam_overlaps: []',
                  '  camera_model: pinhole',
                  '  distortion_model: equidistant',
                  '  distortion_coeffs: [' + ', '.join(f'{float(v):.9f}' for v in c['distortion_coeffs']) + ']',
                  '  intrinsics: [' + ', '.join(f'{float(v):.6f}' for v in c['intrinsics']) + ']',
                  f'  resolution: [{c["resolution"][0]}, {c["resolution"][1]}]',
                  f'  rostopic: /{x["name"]}/{topic_suffix}',
                  f'  timeshift_cam_imu: {shifts[-1]:.6f}  # t_imu = t_cam + shift; open_vins keeps one for all cameras', '']
    with open(os.path.join(out, 'kalibr_imucam_chain.yaml'), 'w') as f:
        f.write('\n'.join(lines))

    # ---- kalibr_imu_chain.yaml
    n = imu_noise or IMU_NOISE
    note = '# IMU noise from Kalibr\'s imu.yaml (imported on the board)' if imu_noise else DATASHEET_NOTE
    imu = f"""%YAML:1.0  # open_vins wants this first line
{note}
imu0:
  T_i_b:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
  accelerometer_noise_density: {n['accelerometer_noise_density']:.4e}  # [ m / s^2 / sqrt(Hz) ]
  accelerometer_random_walk: {n['accelerometer_random_walk']:.4e}    # [ m / s^3 / sqrt(Hz) ]
  gyroscope_noise_density: {n['gyroscope_noise_density']:.4e}      # [ rad / s / sqrt(Hz) ]
  gyroscope_random_walk: {n['gyroscope_random_walk']:.4e}        # [ rad / s^2 / sqrt(Hz) ]
  rostopic: {imu_topic}
  time_offset: 0.0
  update_rate: {imu_rate:.1f}
  model: "kalibr"
  Tw:
    - [ 1.0, 0.0, 0.0 ]
    - [ 0.0, 1.0, 0.0 ]
    - [ 0.0, 0.0, 1.0 ]
  R_IMUtoGYRO:
    - [ 1.0, 0.0, 0.0 ]
    - [ 0.0, 1.0, 0.0 ]
    - [ 0.0, 0.0, 1.0 ]
  Ta:
    - [ 1.0, 0.0, 0.0 ]
    - [ 0.0, 1.0, 0.0 ]
    - [ 0.0, 0.0, 1.0 ]
  R_IMUtoACC:
    - [ 1.0, 0.0, 0.0 ]
    - [ 0.0, 1.0, 0.0 ]
    - [ 0.0, 0.0, 1.0 ]
  Tg:
    - [ 0.0, 0.0, 0.0 ]
    - [ 0.0, 0.0, 0.0 ]
    - [ 0.0, 0.0, 0.0 ]
"""
    with open(os.path.join(out, 'kalibr_imu_chain.yaml'), 'w') as f:
        f.write(imu)

    # ---- masks
    for x in cams:
        w, h = (int(v) for v in x['c']['resolution'])
        write_mask(os.path.join(out, f'mask_{x["name"]}.png'), w, h, radius)

    # ---- estimator_config.yaml: the template with our values
    with open(template) as f:
        text = f.read()
    ncam = len(cams)
    values = {
        'use_stereo': (('true', 'ring stereo, cam i paired with cam i+1 (cameras are in ring order)') if stereo == 'ring'
                       else ('false', 'every camera tracked on its own (monocular)')),
        'max_cameras': (str(ncam), None),
        'calib_cam_extrinsics': ('true', None),
        'calib_cam_intrinsics': ('false', 'trust the Kalibr intrinsics (else 8 more states per camera)'),
        'calib_cam_timeoffset': ('true', 'one offset for all cameras, they share a trigger (no colon in comments, OpenCV YAML misreads it)'),
        'gravity_mag': (f'{gravity}', None),
        # open_vins drops a frame arriving sooner than 1/track_frequency after
        # the last one, and trigger-grid intervals of exactly 1/rate round either way
        'track_frequency': (f'{track_hz}', 'above the camera rate, else frames get dropped'),
        # open_vins divides num_pts and init_max_features by max_cameras (its
        # own comments say per camera, the code does not)
        'num_pts': (str(pts * ncam), f'TOTAL over the {ncam} cameras = {pts} each'),
        'init_max_features': (str(init_pts * ncam), f'TOTAL = {init_pts} per camera before init'),
        # 160 px square cells; the masked-off ones are skipped (TrackKLT grid_budget)
        'grid_x': ('8', None),
        'grid_y': ('5', None),
        # A rig standing still initializes at once: with ZUPT on, open_vins'
        # static init stops waiting for a jerk. ZUPT then holds the state until
        # the first motion and is off after take-off, where a steady hover could
        # pass for standing still. Dynamic init covers a start in motion.
        'try_zupt': ('true', 'a still start initializes at once, no jerk needed'),
        'zupt_only_at_beginning': ('true', 'only until the first motion, never in a hover'),
        'init_dyn_use': ('true', 'for a start in motion; a still start uses the static one'),
        'use_mask': ('true', None),
        'filepath_est': ('"/tmp/ov_estimate.txt"', None),
    }
    if timing:
        values['record_timing_information'] = ('true', 'per-step time of every frame')
        values['record_timing_filepath'] = (f'"{os.path.abspath(timing)}"', None)
    for key, (val, why) in values.items():
        text, k = re.subn(rf'(?m)^{key}:[^\n#]*(#.*)?$', f'{key}: {val}' + (f' # {why}' if why else r' \1'), text)
        if not k:
            raise ValueError(f'template has no "{key}:" line')
    text = re.sub(r'(?m)^mask\d+:.*\n', '', text)
    masks = ''.join(f'mask{i}: "mask_{x["name"]}.png" # {x["name"]}, relative to this file\n' for i, x in enumerate(cams))
    text = re.sub(r'(?m)^(use_mask:.*\n)', lambda mm: mm.group(1) + masks, text)
    header = ('# open_vins config for the RK3588_SLAM 4 x OV9281 fisheye rig, generated by\n'
              '# openvins_config.py (slam-web / kalibr2openvins.py) from openvins_estimator_template.yaml.\n'
              f'# Masks keep {radius:.0f} px around the image centre (~70 deg off-axis), as calibrated.\n')
    text = re.sub(r'^(%YAML:1\.0[^\n]*\n)', lambda mm: mm.group(1) + header, text)
    with open(os.path.join(out, 'estimator_config.yaml'), 'w') as f:
        f.write(text)

    return {'config': os.path.join(out, 'estimator_config.yaml'), 'cameras': [x['name'] for x in cams],
            'azimuth_deg': [round(x['az']) for x in cams], 'timeshift_ms': round(sum(shifts) / len(shifts) * 1000, 2),
            'timeshift_spread_ms': round((max(shifts) - min(shifts)) * 1000, 2), 'stereo': stereo,
            'imu_noise': 'kalibr' if imu_noise else 'datasheet', 'notes': notes}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--camchain', required=True, help='Kalibr camchain-imucam.yaml (calib-web camera-IMU job)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--radius', type=float, default=330.0, help='mask: keep this many px around the image centre')
    ap.add_argument('--template', default=TEMPLATE)
    ap.add_argument('--topic-suffix', default='image',
                    help='/cnN/<suffix>: image = what slam_pc/republish.launch makes on the PC, image_raw on the board')
    ap.add_argument('--imu-topic', default='/imu/data')
    ap.add_argument('--imu-rate', type=float, default=480.0)
    ap.add_argument('--imu-noise', help="Kalibr imu.yaml to take the noise from (else the SCH16T datasheet)")
    ap.add_argument('--gravity', type=float, default=9.781, help='local gravity, m/s^2 (Singapore ~9.781)')
    ap.add_argument('--track-hz', type=float, default=20.0, help='tracking rate cap, above the 15 fps camera rate')
    ap.add_argument('--stereo', choices=['ring', 'mono'], default='ring',
                    help='ring: adjacent cameras as stereo pairs; mono: each camera on its own')
    ap.add_argument('--pts', type=int, default=100, help='features per camera')
    ap.add_argument('--init-pts', type=int, default=50, help='features per camera before initialization')
    a = ap.parse_args()
    noise = None
    if a.imu_noise:
        noise = kalibr.load_imu(a.imu_noise)
        if not kalibr.imu_is_calibrated(noise):
            raise SystemExit(f'{a.imu_noise}: not a calibrated Kalibr imu.yaml (zero noise values)')
    try:
        s = generate(a.camchain, a.out, a.radius, a.topic_suffix, a.imu_topic, a.imu_rate, a.gravity, a.track_hz,
                     a.stereo, a.pts, a.init_pts, noise, a.template)
    except (OSError, ValueError) as e:
        raise SystemExit(str(e))
    for note in s['notes']:
        print('note:', note)
    print(f'{a.out}: {len(s["cameras"])} cameras in ring order ' + ' -> '.join(s['cameras']) +
          f'; time shift {s["timeshift_ms"]:.2f} ms (spread {s["timeshift_spread_ms"]:.2f}); IMU noise {s["imu_noise"]}')


if __name__ == '__main__':
    main()
