# Latent Struct
Latent Struct is a software package, written in C++, that provides routines for training classifiers on data that contain structured latent variables. Currently, log-linear and max-margin (i.e., support vector machine) models can be trained. A trained model can then be employed to make predictions on unseen data. In addition to choosing between log-linear and max-margin models, you can also train either a binary or a multiclass variant of each. Training of fully supervised models (i.e., without latent variables) is also supported.

## Getting Started
After installing the dependencies (see below), run the following commands to download and compile the code:
```
git clone https://github.com/kdd133/latent-struct.git
cd latent-struct
make all
```

### Dependencies
* [Boost C++ libraries](http://www.boost.org/) (version 1.48 or higher)
* [libLBFGS](https://github.com/chokkan/liblbfgs) (version 1.10 or higher)
* [QuadProg++](https://sourceforge.net/projects/quadprog/) (version 1.2.1)

### Installing
The following instructions have been tested on Ubuntu 16.04.

##### Install Boost
```
sudo apt install libboost-program-options* libboost-timer* libboost-system* libboost-thread* libboost-regex* libboost-filesystem*
```

##### Install QuadProg++
Download the file `quadprog-1.2.1.tar.gz` from https://sourceforge.net/projects/quadprog/files/. More recent versions of QuadProg++ do not include the required Boost uBLAS bindings.
```
tar xf quadprog-1.2.1.tar.gz
cd quadprog
./configure
sudo make install
```

##### Install libLBFGS
Note: If you have an Intel processor that supports SSE2, you can pass the `--enable-sse2` flag to `configure` below.
```
git clone https://github.com/chokkan/liblbfgs.git
cd liblbfgs
./autogen.sh
./configure
sudo make install
```
By default, the QuadProg++ and libLBFGS libraries will be installed to `/usr/local`. You may need to add `/usr/local/lib` to your `PATH` environment variable in order to run the `latent_struct` binary. If you're using Bash, you can append the following line to your `~/.bashrc` file:
```
export LD_LIBRARY_PATH=/usr/local/lib
```

## License
This project is licensed under the GNU General Public License (GPLv3). See the [LICENSE](LICENSE.md) file for details.

## Acknowledgments
* Thanks to [Dale Schuurmans](https://webdocs.cs.ualberta.ca/~dale/), [Colin Cherry](https://sites.google.com/site/colinacherry/), and [Robert Holte](https://webdocs.cs.ualberta.ca/~holte/) for many inspiring discussions and help with understanding various models and algorithms.
* To [Shane Bergsma](https://sites.google.com/site/shaneabergsma/) for helping to reproduce the feature extraction algorithm and experiments described in his ACL 2007 [paper](https://aclweb.org/anthology/P/P07/P07-1083.pdf).
* To Zhifei Li and Jason Eisner for this excellent [paper](http://www.aclweb.org/anthology/D09-1005) on semirings and dynamic programming.
* To Choon Hui Teo and colleagues for this very instructive [paper](http://www.jmlr.org/papers/v11/teo10a.html) on bundle methods.
* To Andrew McCallum and colleagues for this [paper](http://www.cs.umass.edu/~mccallum/papers/crfstredit-uai05.pdf) that had a large influence on my work.
