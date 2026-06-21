$ErrorActionPreference = "Stop"

$python = "python"

Write-Host "Python:"
& $python --version
& $python -c "import sys; print(sys.executable)"

Write-Host "Installing PaddlePaddle GPU for CUDA 13.0..."
$wheel = "https://paddle-whl.bj.bcebos.com/stable/cu130/paddlepaddle-gpu/paddlepaddle_gpu-3.3.1-cp313-cp313-win_amd64.whl"
& $python -m pip install --user $wheel

Write-Host "Installing PaddleOCR and helpers..."
& $python -m pip install --user paddleocr opencc-python-reimplemented imageio-ffmpeg tqdm requests pgsrip

Write-Host "Restoring modern setuptools after pgsrip install..."
& $python -m pip install --user "setuptools>=80.9"

Write-Host "Verifying Paddle GPU..."
& $python -c "import os, site, pathlib; dirs=[]; [dirs.extend([p]+[c for c in p.iterdir() if c.is_dir()]) for root in site.getsitepackages()+[site.getusersitepackages()] for p in (pathlib.Path(root)/'nvidia').glob('**/bin') if (pathlib.Path(root)/'nvidia').exists()]; [os.add_dll_directory(str(d)) for d in dirs if d.exists()]; os.environ['PATH']=os.pathsep.join(str(d) for d in dirs if d.exists())+os.pathsep+os.environ.get('PATH',''); import paddle; print('paddle', paddle.__version__); paddle.set_device('gpu:0'); x=paddle.rand([2,2]); print('device', paddle.get_device(), x.numpy().shape)"
