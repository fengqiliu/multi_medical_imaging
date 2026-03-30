from setuptools import setup, find_packages

setup(
    name='multimodal-medical-imaging',
    version='1.0.0',
    description='Multi-modal Medical Image Segmentation and Prognosis Prediction',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'monai>=0.8.0',
        'SimpleITK>=2.1.0',
        'nibabel>=4.0.0',
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'tqdm>=4.62.0',
        'pyyaml>=5.4.0',
        'tensorboard>=2.6.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
