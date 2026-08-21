from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension
)
import shutil
import sys
import importlib.util

base_dir = os.environ.get('BASE_DIR', '')
extra_compile_option = os.environ.get('COMPILE_CXX_FLAGS', '')
extra_link_option = os.environ.get('LINK_FLAGS', '')
extra_link_libs = os.environ.get('LINK_LIBS', '')
extra_build_type = os.environ.get('CMAKE_BUILD_TYPE', '')

extra_compile_flags = extra_compile_option.split()
extra_compile_flags = [f for f in extra_compile_flags if f]
extra_link_flags = extra_link_option.split()
extra_link_flags = [l for l in extra_link_flags if l]
extra_link_libs = extra_link_libs.split()
extra_link_libs = [l for l in extra_link_libs if l]

_abs_file_ = os.path.abspath(__file__)
source_root_dir = os.path.dirname(_abs_file_)

# Get version from '.version' file & __version__ & torch
def get_version():
    import sys
    sys.path.append(os.path.dirname(_abs_file_))
    from __version__ import __version__ as _ver
    from torch import __version__ as torch_full_version
    torch_version = torch_full_version.split('+')[0]

    version = os.environ.get('PACKAGE_VERSION', None)
    if not version:
        version_file = os.path.join(source_root_dir, '.version')
        with open(version_file) as f:
            version = f.read().strip()
    if extra_build_type.lower() == 'debug':
        _build_type = '.debug'
    else:
        _build_type = ''

    return "{}+torch.{}.gcu.{}{}".format(_ver, torch_version, version, _build_type)

# check ccache | sccache tool, but torch.utils cannot support for now.
def get_cache_tool() -> str:
    """ Check if which cache tool(ccache/sccache) is available

    Use sccache is 'ENABLE_SCCACHE' is set.

    """
    cache_tool = ''
    if os.environ.get('ENABLE_SCCACHE', '0').lower()\
                            in ('true', '1', 'on', 't'):
        cache_tool = shutil.which("sccache")
    if not cache_tool:
        cache_tool = shutil.which("ccache")
    return cache_tool if cache_tool else ''

def get_pkg_install_dir(pkg_name: str) -> str:
    try:
        spec = importlib.util.find_spec(pkg_name)
        if spec and spec.origin:
            return os.path.dirname(os.path.abspath(spec.origin))
        else:
            # raise("{} is not installed".format(pkg_name))
            print("Error: {} is not installed".format(pkg_name))
            return None
    except Exception as e:
        # raise("{} is not installed".format(pkg_name))
        print("Error: {} is not installed".format(pkg_name))
        return None


torch_install_dir = get_pkg_install_dir("torch")
torch_gcu_install_dir = get_pkg_install_dir("torch_gcu")

aten_include_dir = os.path.join(torch_install_dir, "include/ATen")
torch_include_dir = os.path.join(torch_install_dir, "include")
torch_api_include_dir = os.path.join(torch_install_dir, "include/torch/csrc/api/include")
torch_gcu_include_dir = os.path.join(torch_gcu_install_dir, "include")
top_runtime_include_dir = base_dir + "/opt/tops/include"
top_aten_include_dir = base_dir + "/usr/include/gcu"
common_include = [ base_dir+"/usr/include" ]


sys_include_dirs = [aten_include_dir, torch_include_dir, torch_api_include_dir, torch_gcu_include_dir, top_runtime_include_dir, top_aten_include_dir]

include_dirs = ["/usr/include"]
include_dirs.extend(common_include)

torch_lib_dir = os.path.join(torch_install_dir, "lib")
torch_gcu_lib_dir = os.path.join(torch_gcu_install_dir, "lib")
common_lib_dir = [
    base_dir + '/usr/lib',
    base_dir + '/opt/top/lib']
library_dirs = []
library_dirs.extend([torch_lib_dir, torch_gcu_lib_dir])
library_dirs.extend(common_lib_dir)
libraries = extra_link_libs
libraries.extend(["c10", "torch", "torch_cpu", "torch_python", "torch_gcu", "topsaten"])
cxx_flags = [
    "-O3",
    "-std=c++17",
    "-DTORCH_EXTENSION_NAME=flash_attn_gcu",
    "-fvisibility=hidden",
    "-Wall", "-Wextra", "-Werror"
]
cxx_flags.extend(extra_compile_flags)
cxx_flags.extend(sum([['-isystem', str(path)] for path in sys_include_dirs], []))

module = CppExtension(
    name='flash_attn_gcu',
    sources=[os.path.join(source_root_dir,'csrc_gcu/flash_attn/flash_api.cpp')],
    include_dirs=[str(inc) for inc in include_dirs],
    libraries=[str(lib) for lib in libraries],
    library_dirs=[str(lib_dir) for lib_dir in library_dirs],
    extra_compile_args=cxx_flags,
    extra_link_args=extra_link_flags + [f'-Wl,-rpath,{torch_lib_dir}', f'-Wl,-rpath,{torch_gcu_lib_dir}']
)

relative_path = os.path.relpath(source_root_dir, os.getcwd())
package_name = 'flash_attn'

packages = find_packages(
    where=source_root_dir,
    include=[
        "flash_attn",
        "flash_attn.layers",
        "flash_attn.losses",
        "flash_attn.modules",
        "flash_attn.ops.triton",
        "flash_attn.utils",
        "flash_attn.vllm_flash_attn"
    ],
)
package_dir = {}
for p in packages:
    package_dir[p] = os.path.join(relative_path, p.replace('.', '/'))

# Debug custom package
# print(packages)
# print(package_dir)
# exit(1)
setup(
    name=package_name,
    version=get_version(),
    description='FlashAttention PyTorch package for GCU environments.',
    ext_modules=[module],
    cmdclass={'build_ext': BuildExtension},
    packages=packages,
    package_dir=package_dir,
)
