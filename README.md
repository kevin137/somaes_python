# somaes_python
Python libraries developed for Software matemático y estadístico, Fall 2024

## Installation

```bash
git clone https://github.com/kevin137/somaes_python.git
cd somaes_python/
pip install .
```

## Usage

```bash
# For interactive use in the python interpreter:
python3

Python 3.13.0 | packaged by conda-forge | (main, Oct  8 2024, 20:04:32) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import somaes_python as sp
>>> sp.discretizeEW(sp.p03_disc_values,4)
(['I3:(10.5,15.5)', 'I2:(5.5,10.5)', 'I1:(-Inf,5.5)', 'I1:(-Inf,5.5)', 'I1:(-Inf,5.5)', 'I4:(15.5,+Inf)', 'I2:(5.5,10.5)'], array([ 5.5, 10.5, 15.5]))
>>> 
```

For interactive use as a Jupyter notebook, open *somaes_python_demo.ipynb* in your favorite frontend.