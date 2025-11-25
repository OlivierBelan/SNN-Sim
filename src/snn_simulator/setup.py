from setuptools import setup
from setuptools._distutils.extension import Extension
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import pathlib
import os


file_absolute_location:str = str(pathlib.Path(__file__).parent.resolve()) +"/"
file_running_location:str = ""

snn_cython = Extension(
    name='SNN_cython_cuda.SNN_cython.snn_cython',
    sources=[str(file_absolute_location + "SNN_cython_cuda/SNN_cython/snn_cython.pyx")],
    extra_compile_args=['-O3', '-march=native'],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
)

record_cython = Extension(name='SNN_cython_cuda.SNN_cython.record_cython',
        sources=[
            file_absolute_location + "SNN_cython_cuda/SNN_cython/record_cython.pyx"
            ],
        extra_compile_args=['-O3', '-march=native'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
        )

runner_SL_cython = Extension(name='SNN_cython_cuda.SNN_cython.runner_SL_cython',
        sources=[
            file_absolute_location + "SNN_cython_cuda/SNN_cython/runner_SL_cython.pyx"
            ],
        extra_compile_args=['-O3', '-march=native'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
        )

runner_RL_cython = Extension(name='SNN_cython_cuda.SNN_cython.runner_RL_cython',
        sources=[
            file_absolute_location + "SNN_cython_cuda/SNN_cython/runner_RL_cython.pyx"
            ],
        extra_compile_args=['-O3', '-march=native'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        )

tools_cython = Extension(name='SNN_cython_cuda.SNN_cython.tools_cython',
        sources=[
            file_absolute_location + "SNN_cython_cuda/SNN_cython/tools_cython.pyx"
            ],
        extra_compile_args=['-O3', '-march=native'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        )


encoder = Extension(name='SNN_cython_cuda.SNN_cython.encoder',
        sources=[
            file_absolute_location + "SNN_cython_cuda/SNN_cython/encoder.pyx"
            ],
        extra_compile_args=['-O3', '-march=native'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        )


augmented = Extension(name='SNN_cython_cuda.SNN_cython.augmented',
        sources=[
            file_absolute_location + "SNN_cython_cuda/SNN_cython/augmented.pyx"
            ],
        extra_compile_args=['-O3', '-march=native'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        )


energy = Extension(name='SNN_cython_cuda.SNN_cython.energy',
        sources=[
            file_absolute_location + "SNN_cython_cuda/SNN_cython/energy.pyx"
            ],
        extra_compile_args=['-O3', '-march=native'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        )


r_stdp = Extension(name='SNN_cython_cuda.SNN_cython.r_stdp',
        sources=[
            file_absolute_location + "SNN_cython_cuda/SNN_cython/r_stdp.pyx"
            ],
        extra_compile_args=['-O3', '-march=native'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        )


#  BUILD CUDA
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess
import shutil

nvcc_path = shutil.which("nvcc")
HAS_CUDA = nvcc_path is not None
path_to_cuda_folder = file_absolute_location + 'SNN_cython_cuda/SNN_cuda/'


if nvcc_path:
    # Définir les chemins CUDA
    CUDA_HOME = os.environ.get('cuda', '/usr/local/')
    CUDA_HOME_2 = os.environ.get('cuda', '/usr/lib/')
    if not os.path.exists(CUDA_HOME) or not os.path.exists(CUDA_HOME_2): raise FileNotFoundError("CUDA_HOME not found, in " + CUDA_HOME + " or " + CUDA_HOME_2)
    if not os.path.exists(CUDA_HOME): CUDA_HOME = CUDA_HOME_2

    # Arguments de compilation et de liaison
    cuda_lib_dir = os.path.join(CUDA_HOME, 'lib64') if sys.platform != 'win32' else os.path.join(CUDA_HOME, 'lib')
    extra_compile_args = ['-std=c++11']
    extra_link_args = ['-L' + cuda_lib_dir, '-lcudart']


    class CustomBuildExt(build_ext):
        def build_extensions(self):
            # Parcourir les extensions (ici une seule)
            for ext in self.extensions:
                # Extraire les sources CUDA
                cuda_sources = [s for s in ext.sources if s.endswith('.cu')]
                # Supprimer les sources CUDA de la liste principale
                ext.sources = [s for s in ext.sources if not s.endswith('.cu')]
                # Compiler chaque fichier CUDA
                for src in cuda_sources:
                    obj = self.compile_cuda(src)
                    ext.extra_objects.append(obj)
                # Ajouter les répertoires d'inclusion CUDA
                ext.include_dirs.append('/usr/local/cuda/include')
                # Ajouter les répertoires de bibliothèque CUDA
                ext.library_dirs.append('/usr/local/cuda/lib64')
                # Ajouter les bibliothèques CUDA nécessaires
                ext.libraries.append('cudart')
            # Appeler le build standard
            build_ext.build_extensions(self)

        def compile_cuda(self, src):
            # Définir le nom de l'objet
            obj = os.path.splitext(src)[0] + '.o'
            # Vérifier si l'objet doit être recompilé
            if not os.path.exists(obj) or os.path.getmtime(obj) < os.path.getmtime(src):
                nvcc = 'nvcc'  # Assurez-vous que nvcc est dans le PATH
                arch = '-arch=sm_86'  # Ajustez selon votre GPU (https://developer.nvidia.com/cuda-gpus#compute)
                compile_flags = '-c -O3 -Xcompiler -fPIC'
                include_dirs = ['-I/usr/local/cuda/include', f"-I{numpy.get_include()}"]
                cmd = f"{nvcc} {arch} {compile_flags} {' '.join(include_dirs)} {src} -o {obj}"
                print(f"Compiling CUDA source: {cmd}")
                subprocess.check_call(cmd, shell=True)
            return obj

    snn_cuda_wrapper = Extension(
            name="SNN_cython_cuda.SNN_cuda.snn_cuda_wrapper",
            sources=[path_to_cuda_folder + "snn_cuda_wrapper.pyx", path_to_cuda_folder + "snn_cuda_f32.cu"],
            language="c++",
            include_dirs=[numpy.get_include(), CUDA_HOME + '/include'],
            library_dirs=[CUDA_HOME + '/lib64'],
            libraries=['cudart'],
            runtime_library_dirs=[CUDA_HOME + '/lib64'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            depends=[path_to_cuda_folder + "snn_cuda.cu"] 
        )
else:
    print("nvcc not found. CUDA code will not be compiled. If you want to compile CUDA code, make sure nvcc is in your PATH.")
    snn_cuda_wrapper = None
    CustomBuildExt = build_ext


extension = [snn_cython, runner_SL_cython, runner_RL_cython, record_cython, tools_cython, encoder, augmented, energy, r_stdp]
if snn_cuda_wrapper:
    extension.append(snn_cuda_wrapper)

setup(
    name='snn_simulator',
    packages=["SNN_cython_cuda"],
    ext_modules=cythonize(extension,
    language_level=3, 
    annotate=False,
    # annotate=True, # generate html file with the cython code
    compile_time_env={'HAS_CUDA': HAS_CUDA}
    ),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
    cmdclass={'build_ext': CustomBuildExt},
)