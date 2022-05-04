import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    

setuptools.setup(
     name='vision_aided_loss',  
     version='0.1.0',
     author="Nupur Kumari",
     author_email="nkumari@andrew.cmu.edu",
     description="Vision-aided adversarial training",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/nupurkmr9/vision_aided_loss",
     packages=['vision_aided_loss'],
     install_requires=[
                      "torch>=1.8.0",
                      "torchvision>=0.9.0",
                       "numpy>=1.14.3", 
                       "requests",
                       "timm",
                       "antialiased_cnns",
                       "gdown==4.4.0",
                       "ftfy",
                       "regex", 
                       "tqdm",
                       "clip@git+https://github.com/openai/CLIP.git"
                      ],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
     ],
 )