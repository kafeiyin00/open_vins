"""Kalibr calibration files <-> what the board keeps in /data/bags/calib.

The camera nodes load cnN.yaml (camera_info_manager's format, camera.launch
calib_dir). Camera calibration is taken only as Kalibr's camera-IMU result,
camchain-imucam.yaml (intrinsics and extrinsics from one run):

    cam0:
      camera_model: pinhole
      intrinsics: [fu, fv, pu, pv]
      distortion_model: radtan          # or equidistant
      distortion_coeffs: [k1, k2, r1, r2]
      resolution: [1280, 800]
      rostopic: /cn2/image_raw          # how a camN maps to a connector
      T_cn_cnm1: [[...] x4]             # cam1..: from the previous camera
      T_cam_imu: [[...] x4]             # required
      timeshift_cam_imu: 0.0

Only pinhole cameras with radtan (-> ROS plumb_bob) or equidistant
distortion have a camera_info equivalent; anything else is reported, not
converted. Extrinsics have no place in camera_info: they are kept in
camchain-imucam.yaml on the board. The IMU noise model is Kalibr's imu.yaml.
"""
import re

import yaml

RESOLUTION = {'cn2': (1280, 800), 'cn3': (1280, 800), 'cn4': (1280, 800),
              'cn5': (1280, 800), 'cn6': (1280, 800), 'cn7': (640, 512)}
TOPIC_RE = re.compile(r'^/?(cn[2-7])(/|$)')


class KalibrError(ValueError):
    pass


IMU_TOPIC = '/imu/data'
IMU_RATE = 480.0
IMU_KEYS = ('accelerometer_noise_density', 'accelerometer_random_walk',
            'gyroscope_noise_density', 'gyroscope_random_walk')


def load_yaml(text):
    """YAML text -> dict. Tolerates OpenCV's "%YAML:1.0" first line, which
    open_vins' kalibr_*.yaml files carry and PyYAML rejects."""
    text = re.sub(r'^\s*%YAML:\s*[\d.]+[^\n]*\n', '', text)
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise KalibrError(f'不是合法的 YAML：{e}')
    if not isinstance(doc, dict):
        raise KalibrError('不是 YAML 字典')
    return doc


def kind(doc):
    """'camchain' (camchain / camchain-imucam), 'imu' (Kalibr imu.yaml, also
    the imu0: form of Kalibr's results and open_vins' kalibr_imu_chain), or None."""
    if any(re.match(r'^cam\d+$', str(k)) for k in doc):
        return 'camchain'
    src = doc.get('imu0') if isinstance(doc.get('imu0'), dict) else doc
    if any(k in src for k in IMU_KEYS):
        return 'imu'
    return None


def _mat4(v):
    """A 4x4 matrix from YAML (list of 4 rows), or None."""
    try:
        m = [[float(x) for x in row] for row in v]
    except (TypeError, ValueError):
        return None
    return m if len(m) == 4 and all(len(r) == 4 for r in m) else None


def mat4_set(m):
    return bool(m) and any(abs(x) > 0 for row in m for x in row)


def _inv(T):
    """Inverse of a rigid transform [R t; 0 1]: [R' -R't; 0 1]."""
    R = [r[:3] for r in T[:3]]
    t = [r[3] for r in T[:3]]
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    ti = [-sum(Rt[i][j] * t[j] for j in range(3)) for i in range(3)]
    return [Rt[0] + [ti[0]], Rt[1] + [ti[1]], Rt[2] + [ti[2]], [0.0, 0.0, 0.0, 1.0]]


def _mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def parse_camchain(text):
    """camchain-imucam.yaml -> ({cam: camera_info dict}, [notes]). Only the
    camera-IMU result is accepted: a camera without a valid T_cam_imu is
    skipped, and a plain camchain.yaml (no T_cam_imu anywhere) is refused,
    so intrinsics and extrinsics on the board always come from one Kalibr
    run. T_cam_imu and timeshift_cam_imu come back in ci['kalibr'].
    Raises KalibrError when nothing in the file can be used."""
    doc = text if isinstance(text, dict) else load_yaml(text)
    cams = [k for k in doc if re.match(r'^cam\d+$', str(k))] if isinstance(doc, dict) else []
    if not cams:
        raise KalibrError('没有 cam0 / cam1 ... 条目，不像 Kalibr 的 camchain-imucam 文件')
    if not any(isinstance(doc[k], dict) and 'T_cam_imu' in doc[k] for k in cams):
        raise KalibrError('这是只有内参的 camchain.yaml：请上传 Kalibr 相机-IMU 标定（kalibr_calibrate_imu_camera）'
                          '输出的 camchain-imucam.yaml，它同时带内参和外参 T_cam_imu')

    out, notes = {}, []
    for key in sorted((k for k in doc if re.match(r'^cam\d+$', str(k))), key=lambda k: int(k[3:])):
        c = doc[key] or {}
        m = TOPIC_RE.match(str(c.get('rostopic', '')))
        if not m:
            notes.append(f'{key}: rostopic "{c.get("rostopic", "")}" 对应不到 cn2..cn7，跳过')
            continue
        cam = m.group(1)
        if c.get('camera_model') != 'pinhole':
            notes.append(f'{key} ({cam}): camera_model {c.get("camera_model")} 没有 camera_info 对应，跳过')
            continue
        try:
            fu, fv, pu, pv = (float(x) for x in c['intrinsics'])
            w, h = (int(x) for x in c['resolution'])
            coeffs = [float(x) for x in c.get('distortion_coeffs', [])]
        except (KeyError, TypeError, ValueError):
            notes.append(f'{key} ({cam}): intrinsics / resolution / distortion_coeffs 缺失或格式不对，跳过')
            continue
        if (w, h) != RESOLUTION[cam]:
            notes.append(f'{key} ({cam}): 分辨率 {w}x{h} 与相机 {RESOLUTION[cam][0]}x{RESOLUTION[cam][1]} 不符，跳过')
            continue
        model = c.get('distortion_model', 'radtan')
        if model == 'radtan' and len(coeffs) == 4:
            dmodel, d = 'plumb_bob', coeffs + [0.0]
        elif model == 'equidistant' and len(coeffs) == 4:
            dmodel, d = 'equidistant', coeffs
        else:
            notes.append(f'{key} ({cam}): 畸变模型 {model}（{len(coeffs)} 个系数）不支持，跳过')
            continue
        if not mat4_set(_mat4(c.get('T_cam_imu'))):
            notes.append(f'{key} ({cam}): 没有有效的 T_cam_imu（4x4 矩阵），跳过')
            continue
        if cam in out:
            notes.append(f'{key}: {cam} 在文件里出现了两次，用后一个')
        out[cam] = {'width': w, 'height': h, 'K': [fu, 0.0, pu, 0.0, fv, pv, 0.0, 0.0, 1.0],
                    'distortion_model': dmodel, 'D': d,
                    'kalibr': {'cam': key, 'has_T_cam_imu': 'T_cam_imu' in c,
                               'T_cam_imu': _mat4(c.get('T_cam_imu')),
                               'timeshift_cam_imu': c.get('timeshift_cam_imu')}}
    if not out:
        raise KalibrError('文件里没有可用的相机：' + '；'.join(notes))
    return out, notes


TEMPLATE_HEAD = (
    '# Not calibrated: all-zero template, the format camera_info_manager reads.\n'
    '# 未标定：全 0 模板，即相机节点读取的格式。\n'
    '# Fill camera_matrix [fx 0 cx; 0 fy cy; 0 0 1] and distortion_coefficients\n'
    '# (plumb_bob: k1 k2 p1 p2 k3), or import a Kalibr camchain in slam-web,\n'
    '# which replaces this file. 填 camera_matrix 和 distortion_coefficients，\n'
    '# 或在网页"标定"页导入 Kalibr camchain（会覆盖本文件）。\n')


def zero_info(cam):
    w, h = RESOLUTION[cam]
    return {'width': w, 'height': h, 'K': [0.0] * 9, 'distortion_model': 'plumb_bob', 'D': [0.0] * 5}


def is_calibrated(ci):
    """camera_info_manager's own test: a focal length is set."""
    return bool(ci) and ci['K'][0] > 0 and ci['K'][4] > 0


def ros_yaml(cam, ci):
    """camera_info_manager's file format (camera_calibration_parsers). An
    uncalibrated ci (zero K) is written as the all-zero template, R included,
    which is exactly what the node publishes without a file."""
    K = ci['K']
    if is_calibrated(ci):
        R, P, head = [1, 0, 0, 0, 1, 0, 0, 0, 1], [K[0], 0.0, K[2], 0.0, 0.0, K[4], K[5], 0.0, 0.0, 0.0, 1.0, 0.0], ''
    else:
        R, P, head = [0.0] * 9, [0.0] * 12, TEMPLATE_HEAD
    fmt = lambda xs: '[' + ', '.join(repr(float(x)) for x in xs) + ']'
    return (head + f'image_width: {ci["width"]}\nimage_height: {ci["height"]}\ncamera_name: {cam}\n'
            f'camera_matrix:\n  rows: 3\n  cols: 3\n  data: {fmt(K)}\n'
            f'distortion_model: {ci["distortion_model"]}\n'
            f'distortion_coefficients:\n  rows: 1\n  cols: {len(ci["D"])}\n  data: {fmt(ci["D"])}\n'
            f'rectification_matrix:\n  rows: 3\n  cols: 3\n  data: {fmt(R)}\n'
            f'projection_matrix:\n  rows: 3\n  cols: 4\n  data: {fmt(P)}\n')


def load_ros(path):
    """cnN.yaml -> camera_info dict, or None."""
    try:
        with open(path) as f:
            d = yaml.safe_load(f)
        return {'width': int(d['image_width']), 'height': int(d['image_height']),
                'K': [float(x) for x in d['camera_matrix']['data']],
                'distortion_model': d.get('distortion_model', 'plumb_bob'),
                'D': [float(x) for x in d['distortion_coefficients']['data']]}
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return None


# ------------------------------------------------------------------ IMU
# Kalibr's IMU config (the imu.yaml kalibr_calibrate_imu_camera takes):
# continuous-time noise densities and bias random walks, from an Allan
# variance of a long static recording.

IMU_UNITS = {'accelerometer_noise_density': 'm/s^2/sqrt(Hz)  accel white noise / 加速度计白噪声',
             'accelerometer_random_walk': 'm/s^3/sqrt(Hz)  accel bias random walk / 加速度计零偏随机游走',
             'gyroscope_noise_density': 'rad/s/sqrt(Hz)  gyro white noise / 陀螺仪白噪声',
             'gyroscope_random_walk': 'rad/s^2/sqrt(Hz)  gyro bias random walk / 陀螺仪零偏随机游走'}


def zero_imu():
    return dict({k: 0.0 for k in IMU_KEYS}, rostopic=IMU_TOPIC, update_rate=IMU_RATE)


def imu_is_calibrated(imu):
    return bool(imu) and all(imu.get(k, 0) > 0 for k in IMU_KEYS)


def parse_imu(doc):
    """Kalibr imu.yaml (top-level keys) or the imu0: form -> (imu dict, notes)."""
    src = doc.get('imu0') if isinstance(doc.get('imu0'), dict) else doc
    notes = []
    try:
        imu = {k: float(src[k]) for k in IMU_KEYS}
    except KeyError as e:
        raise KalibrError(f'IMU 参数缺 {e.args[0]}')
    except (TypeError, ValueError):
        raise KalibrError('IMU 噪声参数必须是数字')
    imu['rostopic'] = str(src.get('rostopic', IMU_TOPIC))
    try:
        imu['update_rate'] = float(src.get('update_rate', IMU_RATE))
    except (TypeError, ValueError):
        imu['update_rate'] = IMU_RATE
    if imu['rostopic'] != IMU_TOPIC:
        notes.append(f'rostopic 是 {imu["rostopic"]}，本机 IMU 发布在 {IMU_TOPIC}，导出时改成 {IMU_TOPIC}')
        imu['rostopic'] = IMU_TOPIC
    if abs(imu['update_rate'] - IMU_RATE) > 1:
        notes.append(f'update_rate {imu["update_rate"]:g} Hz 与本机 IMU 的 {IMU_RATE:g} Hz 不同，已按原值保存')
    if not imu_is_calibrated(imu):
        notes.append('有噪声参数为 0：按"未标定"处理')
    return imu, notes


def imu_yaml(imu):
    head = '' if imu_is_calibrated(imu) else (
        '# Not calibrated: all-zero template in Kalibr imu.yaml format (the IMU config\n'
        '# kalibr_calibrate_imu_camera takes). Get the values from an Allan variance of\n'
        '# a static recording of 2 h or more (capture page: tick only the IMU), e.g.\n'
        '# with allan_variance_ros; Kalibr suggests inflating them for real use.\n'
        '# 未标定：Kalibr imu.yaml 格式的全 0 模板（相机-IMU 标定时 Kalibr 要的 IMU 配置）。\n'
        '# 数值来自静止录制 2 小时以上（采集页只勾 IMU）做 Allan 方差分析，\n'
        '# 例如用 allan_variance_ros；实际使用时 Kalibr 建议适当放大。\n')
    body = ''.join(f'{k}: {imu[k]:.6e}  # {IMU_UNITS[k]}\n' for k in IMU_KEYS)
    return head + body + f'rostopic: {imu["rostopic"]}\nupdate_rate: {imu["update_rate"]:.1f}  # Hz\n'


def load_imu(path):
    try:
        with open(path) as f:
            doc = load_yaml(f.read())
        return parse_imu(doc)[0]
    except (OSError, KalibrError):
        return None


# ------------------------------------------------------------------ camera-IMU
# camchain-imucam.yaml as Kalibr writes it. On the board it is the store for
# the extrinsics (T_cam_imu, timeshift_cam_imu); the intrinsics in it are a
# copy of cnN.yaml, which stays the source the camera nodes read.

def load_extrinsics(path):
    """camchain-imucam.yaml -> {cam: {'T_cam_imu': 4x4 or None, 'timeshift_cam_imu': float}}."""
    try:
        with open(path) as f:
            doc = load_yaml(f.read())
    except (OSError, KalibrError):
        return {}
    out = {}
    for key, c in doc.items():
        if not re.match(r'^cam\d+$', str(key)) or not isinstance(c, dict):
            continue
        m = TOPIC_RE.match(str(c.get('rostopic', '')))
        if not m:
            continue
        try:
            ts = float(c.get('timeshift_cam_imu') or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        out[m.group(1)] = {'T_cam_imu': _mat4(c.get('T_cam_imu')), 'timeshift_cam_imu': ts}
    return out


def imucam_yaml(cams, extr):
    """[(cam, camera_info dict)] + {cam: extrinsics} -> camchain-imucam text.
    T_cn_cnm1 is derived from consecutive T_cam_imu when both are set."""
    fmt = lambda xs: '[' + ', '.join(f'{float(x):.10g}' for x in xs) + ']'
    mat = lambda m: ''.join(f'  - {fmt(r)}\n' for r in m)
    zero4 = [[0.0] * 4 for _ in range(4)]
    any_set = any(is_calibrated(ci) for _, ci in cams)
    out = [] if any_set else [
        '# Not calibrated: all-zero template in the format kalibr_calibrate_imu_camera\n'
        '# writes (camchain-imucam.yaml). Import Kalibr\'s result in slam-web to fill it.\n'
        '# T_cam_imu: IMU frame -> camera frame; T_cn_cnm1: camera n-1 -> camera n;\n'
        '# timeshift_cam_imu: t_imu = t_cam + shift [s]. Intrinsics follow cnN.yaml.\n'
        '# 未标定：Kalibr 相机-IMU 标定输出格式的全 0 模板，在网页导入 Kalibr 结果即可填好。\n'
        '# T_cam_imu：IMU 系到相机系；T_cn_cnm1：前一个相机到本相机；timeshift_cam_imu：\n'
        '# t_imu = t_cam + shift（秒）。内参以 cnN.yaml 为准。\n']
    prev = None
    for i, (cam, ci) in enumerate(cams):
        K, e = ci['K'], extr.get(cam, {})
        T = e.get('T_cam_imu') if mat4_set(e.get('T_cam_imu')) else None
        model = 'equidistant' if ci['distortion_model'] == 'equidistant' else 'radtan'
        out.append(f'cam{i}:\n  camera_model: pinhole\n  intrinsics: {fmt([K[0], K[4], K[2], K[5]])}\n'
                   f'  distortion_model: {model}\n  distortion_coeffs: {fmt(ci["D"][:4])}\n'
                   f'  resolution: [{ci["width"]}, {ci["height"]}]\n  rostopic: /{cam}/image_raw\n')
        out.append('  T_cam_imu:\n' + mat(T or zero4))
        if i:
            out.append('  T_cn_cnm1:\n' + mat(_mul(T, _inv(prev)) if T and prev else zero4))
        out.append(f'  timeshift_cam_imu: {float(e.get("timeshift_cam_imu") or 0.0):.10g}\n')
        prev = T
    return ''.join(out)
