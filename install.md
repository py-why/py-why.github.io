---
layout: page
---

# DoWhy Installation

DoWhy support Python 3.6+. To install, you can use pip or conda.

### Latest Release

Install the latest [release](https://pypi.org/project/dowhy/) using pip.

```shell
pip install dowhy
```

Install the latest [release](https://pypi.org/project/dowhy/) using conda.

```shell
conda install -c conda-forge dowhy
```

If you face "Solving environment" problems with conda, then try conda update --all and then install dowhy. If that does not work, then use conda config --set channel_priority false and try to install again. If the problem persists, please add your issue [here](https://github.com/microsoft/dowhy/issues/197).

### Development Version

If you prefer the latest dev version, clone this repository and run the following command from the top-most folder of the repository.

```shell
pip install -e .
```

### Requirements

DoWhy requires the following packages:

- numpy
- scipy
- scikit-learn
- pandas
- networkx (for analyzing causal graphs)
- matplotlib (for general plotting)
- sympy (for rendering symbolic expressions)
- If you face any problems, try installing dependencies manually.

```shell
pip install -r requirements.txt
```

Optionally, if you wish to input graphs in the dot format, then install pydot (or pygraphviz).

For better-looking graphs, you can optionally install pygraphviz. To proceed, first install graphviz and then pygraphviz (on Ubuntu and Ubuntu WSL).

```shell
sudo apt install graphviz libgraphviz-dev graphviz-dev pkg-config
## from https://github.com/pygraphviz/pygraphviz/issues/71
pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" \
--install-option="--library-path=/usr/lib/graphviz/"
```