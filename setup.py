

from setuptools import setup, find_packages

setup(
    name="breakroom",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mmdet>=3.0.0",
        "mmcv>=2.0.0",
        "mmengine>=0.7.0",
        "opencv-python",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            # Example CLI entry point: adjust target function path as needed
            "bbox-detector=scripts.bbox_detection:main",
        ],
    },
    author="Daveed Vodonenko",
    author_email="Daveed@example.com",
    description="Breakroom project with MMDetection bounding box utilities",
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)