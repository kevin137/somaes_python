from setuptools import setup

setup(
   name='somaes_python',
   version='0.0.1',
   author='Kevin Cook',
   author_email='kcook001@ikasle.ehu.eus',
   packages=['somaes_python'],
   url='https://github.com/kevin137/somaes_python',
   license='LICENSE.txt',
   description='This package includes functions created for KISA SoMaEs, Fall 2024',
   long_description=open('README.md').read(),
   install_requires=[
      "numpy >=2.1.2",
      "pandas >= 2.2.3",
      "matplotlib >= 3.9.2",
      "seaborn >= 0.13.2",
   ],
)
