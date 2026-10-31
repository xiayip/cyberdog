#!/usr/bin/env bash
# Set up Cyberdog on the existing Zephyr Ubuntu 24.04 / ROS Jazzy GPU image.
# Does not start ROS nodes, move the arm, or play audio.
set -Eeuo pipefail
trap 'printf "[ERROR] 第 %s 行失败，安装未完成。修复报错后可重新运行。\n" "$LINENO" >&2' ERR

usage() {
    cat <<'EOF'
用法：bash scripts/setup_docker_env.sh [选项]
默认在现有 Zephyr Docker 容器内安装缺失依赖、检查环境、构建 cyberdog。

  --workspace DIR   工作区（默认 /workspaces/zephyr-dev/zephyr_ws）
  --check           仅检查，不安装、不构建
  --skip-build      安装并检查依赖，跳过 colcon 构建
  --skip-audio-check 暂不连接音箱时使用：跳过桌面连接检查，仍安装音频工具
  --host-audio      在宿主机桌面用户下开启回环音频连接（与容器模式分开运行）
  -h, --help        显示帮助

前提：Ubuntu 24.04、ROS Jazzy、匹配 NVIDIA/CUDA 的 torch/torchvision，
以及已构建的 Zephyr 工作区。容器用 host 网络连接宿主机音箱。
脚本不替换 GPU 依赖、不下载模型、不启动机械臂。
EOF
}
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
info() { printf '[cyberdog-env] %s\n' "$*"; }
workspace=/workspaces/zephyr-dev/zephyr_ws
check=false
build=true
audio_check=true
host_audio=false
original_args=("$@")
while (($#)); do
    case "$1" in
        --workspace)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die '--workspace 需要目录参数'
            workspace=$2; shift 2 ;;
        --check) check=true; build=false; shift ;;
        --skip-build) build=false; shift ;;
        --skip-audio-check) audio_check=false; shift ;;
        --host-audio) host_audio=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "未知参数：$1（使用 --help 查看用法）" ;;
    esac
done
script_dir=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)
in_container=false
if [[ -f /.dockerenv || -f /run/.containerenv ]]; then in_container=true; fi

if $host_audio; then
    ((${#original_args[@]} == 1)) || die '--host-audio 请单独运行'
    $in_container && die '--host-audio 必须在宿主机运行，不能在 Docker 内运行'
    ((EUID != 0)) || die '--host-audio 请用已登录桌面的普通用户运行，不要 sudo'
    if ! command -v pactl >/dev/null; then
        info '为宿主机安装 PulseAudio 客户端工具'
        sudo apt-get update
        sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends pulseaudio-utils
    fi
    [[ -f "$script_dir/enable_desktop_audio.py" ]] || die '缺少同目录 enable_desktop_audio.py，请保留完整 cyberdog 包'
    python3 "$script_dir/enable_desktop_audio.py"
    PULSE_SERVER=tcp:127.0.0.1:4713 timeout 5 pactl info
    info '宿主机音频连接就绪；容器须使用 --network host。音频服务重启后重新运行此命令。'
    exit 0
fi

$in_container || die '安装模式请在 Docker 内运行；宿主机音频设置使用 --host-audio'
# shellcheck source=/dev/null
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] || die '仅适用于现有 Ubuntu 24.04 Zephyr 镜像'
[[ -f /opt/ros/jazzy/setup.bash ]] || die '缺少 ROS Jazzy；请先使用工作区 docker/ 中的 Zephyr GPU 镜像'
workspace=$(realpath -e -- "$workspace")
package="$workspace/src/cyberdog"
[[ -f "$package/package.xml" ]] || die "未找到 $package/package.xml；请用 --workspace 指定工作区"
[[ -f "$workspace/install/setup.bash" ]] || die '工作区尚未构建：先按 Zephyr 工作区说明构建底层驱动和控制器'

# docker exec defaults to root; keep the bind-mounted build and venv owned by
# the source owner. Root-owned Docker build contexts may intentionally use root.
owner_uid=$(stat -c '%u' "$package")
if ((EUID == 0 && owner_uid != 0)); then
    owner_name=$(getent passwd "$owner_uid" | cut -d: -f1)
    [[ -n "$owner_name" ]] || die "工作区属 UID $owner_uid，但容器没有该用户，请用正确用户运行"
    info "切换到工作区用户 $owner_name，避免生成 root 所有的文件"
    exec sudo -H -u "$owner_name" bash "$script_dir/setup_docker_env.sh" "${original_args[@]}"
fi

# The custom CUDA wheels belong to the base image, never to pip's resolver here.
info '检查镜像 GPU 环境（不安装或替换 torch/torchvision）'
python3 -B - <<'PY'
try:
    import torch, torchvision
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用；请检查 NVIDIA 驱动及容器 GPU 透传')
    print(f'  torch={torch.__version__}, torchvision={torchvision.__version__}, CUDA={torch.version.cuda}')
except Exception as exc:
    raise SystemExit(f'GPU 环境不完整：{exc}\n请使用匹配当前硬件的 Zephyr GPU 镜像；此脚本不会用普通 PyPI torch 覆盖 NVIDIA 版本。')
PY

# Install only absent packages: no apt upgrade and no global pip installation.
apt_packages=(
    python3-venv python3-pip python3-colcon-common-extensions python3-setuptools
    python3-pytest
    pulseaudio-utils iproute2 procps
)
# The image also provides some modules through /usr/local. Do not install a
# second apt copy merely because that module has no dpkg record.
python_missing=$(python3 -B - <<'PY'
from importlib.util import find_spec
for module, package in (
    ('tkinter', 'python3-tk'), ('cv2', 'python3-opencv'), ('numpy', 'python3-numpy'),
    ('yaml', 'python3-yaml'), ('matplotlib', 'python3-matplotlib'),
    ('PIL', 'python3-pil'), ('requests', 'python3-requests'), ('psutil', 'python3-psutil'),
):
    if find_spec(module) is None:
        print(package)
PY
)
while IFS= read -r name; do
    if [[ -n "$name" ]]; then apt_packages+=("$name"); fi
done <<< "$python_missing"
ros_packages=(
    rclpy std-msgs rcl-interfaces std-srvs sensor-msgs control-msgs trajectory-msgs
    action-msgs controller-manager-msgs launch-ros ament-index-python tf2-ros
    geometry-msgs moveit-msgs shape-msgs image-transport rqt-image-view
)
for name in "${ros_packages[@]}"; do apt_packages+=("ros-jazzy-$name"); done
missing=()
for name in "${apt_packages[@]}"; do
    if [[ $(dpkg-query -W -f='${Status}' "$name" 2>/dev/null || true) != 'install ok installed' ]]; then
        missing+=("$name")
    fi
done
if ((${#missing[@]})); then
    $check && die "缺少系统包：${missing[*]}；不带 --check 重新运行可安装"
    elevate=()
    if ((EUID != 0)); then elevate=(sudo); fi
    info "安装缺少的系统包：${missing[*]}"
    "${elevate[@]}" apt-get update
    "${elevate[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
else
    info '系统依赖已齐全，跳过 apt'
fi

# ROS setup scripts are not nounset-safe.
set +u
# shellcheck source=/dev/null
source /opt/ros/jazzy/setup.bash
# shellcheck source=/dev/null
source "$workspace/install/setup.bash"
set -u
info '检查工作区底层依赖与 Python 模块'
python3 -B - <<'PY'
from ament_index_python.packages import get_package_prefix, PackageNotFoundError
import cv2, numpy, yaml, tkinter, rclpy, launch_ros
import matplotlib, PIL, requests, psutil
from control_msgs.action import FollowJointTrajectory, ParallelGripperCommand
from moveit_msgs.srv import GetCartesianPath
from controller_manager_msgs.srv import SwitchController
required = ('ghost_game_orchestrator', 'rebot_arm_hmi_bridge', 'zephyr_arm_bringup',
            'zephyr_arm_moveit_config', 'mit_position_adapter_controller', 'orbbec_camera')
missing = []
for name in required:
    try:
        print(f'  {name}: {get_package_prefix(name)}')
    except PackageNotFoundError:
        missing.append(name)
if missing:
    raise SystemExit('请先构建工作区底层包：' + ', '.join(missing))
print(f'  numpy={numpy.__version__}, OpenCV={cv2.__version__}')
PY

venv="$workspace/.cyberdog-yolo"
if [[ ! -x "$venv/bin/python" ]]; then
    $check && die "缺少 $venv；不带 --check 重新运行可创建"
    [[ ! -e "$venv" ]] || die "$venv 已存在但不完整，请检查该目录后再运行"
    info '创建独立 YOLOE 环境，复用镜像 GPU 库'
    python3 -m venv --system-site-packages "$venv"
fi
grep -Eq '^include-system-site-packages = true$' "$venv/pyvenv.cfg" || die "$venv 必须以 --system-site-packages 创建"
if ! "$venv/bin/python" -B - <<'PY'
import importlib.metadata as m
try:
    good = m.version('ultralytics') == '8.4.176'
except m.PackageNotFoundError:
    good = False
raise SystemExit(0 if good else 1)
PY
then
    $check && die 'YOLOE 环境缺少 ultralytics==8.4.176；不带 --check 重新运行可安装'
    # Cyberdog only uses inference; training/export/cloud dependencies are not needed.
    "$venv/bin/python" -m pip install --disable-pip-version-check --no-deps 'ultralytics==8.4.176'
else
    info 'ultralytics==8.4.176 已就绪，跳过 pip'
fi

info '检查 YOLOE 导入、GPU 算子、模型及音频文件'
PYTHONPATH="$package${PYTHONPATH:+:$PYTHONPATH}" YOLO_AUTOINSTALL=false \
    "$venv/bin/python" -B - "$package" <<'PY'
import hashlib, json, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from ultralytics import YOLOE
from cyberdog.audio_player import load_sounds
root = Path(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit('YOLOE 虚拟环境 CUDA 不可用')
# Exercise the actual CUDA NMS operator; importing torchvision alone is insufficient.
boxes = torch.tensor([[0., 0., 10., 10.]], device='cuda:0')
assert torchvision.ops.nms(boxes, torch.ones(1, device='cuda:0'), .5).numel() == 1
assert cv2.cvtColor(np.zeros((2, 2, 3), dtype=np.uint8), cv2.COLOR_BGR2GRAY).shape == (2, 2)
manifest = json.loads((root/'models/manifest.json').read_text())
weights = root/'models/cyberdog-rope-yoloe.pt'
if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest['sha256']:
    raise SystemExit('毛球模型校验失败，请恢复包内原始权重（脚本不会下载或覆盖权重）')
sounds = load_sounds(root/'audio')
print(f'  YOLOE 导入/CUDA NMS/OpenCV 正常，模型 SHA256 正确，{len(sounds)} 个音频校验通过')
PY

if $audio_check; then
    # Read the same server as the player; an empty value uses the desktop default.
    audio_server=$(python3 -B - "$package/config/cyberdog.yaml" <<'PY'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
print(config.get('cyberdog_audio', {}).get('ros__parameters', {}).get('audio_server', ''))
PY
)
    info "检查桌面音频连接：${audio_server:-默认服务器}（不播放声音）"
    audio_env=()
    if [[ -n "$audio_server" ]]; then audio_env=("PULSE_SERVER=$audio_server"); fi
    if ! env "${audio_env[@]}" timeout 5 pactl info; then
        die '音频连接失败：在宿主机桌面运行本脚本 --host-audio，并确认容器使用 host 网络；暂不接音箱可用 --skip-audio-check'
    fi
else
    info '已跳过桌面音频连接检查；运行时仍需连接音频服务器'
fi

if $build; then
    info '仅构建 cyberdog 包'
    cd "$workspace"
    colcon build --packages-select cyberdog --symlink-install
fi
info '环境检查通过。没有启动或重启机器人进程。'
printf '\n启动时在容器终端执行：\ncd %q\nsource /opt/ros/jazzy/setup.bash\nsource install/setup.bash\nros2 launch cyberdog bringup.launch.py\n' "$workspace"
