# Third-party software

This app is Apache-2.0 (see `LICENSE`). The image contains these third-party components, installed
from PyPI at the pinned versions in `requirements.txt`; their licences apply to the image. Licences
were read from each installed package's metadata in the built image (`importlib.metadata`).
**No GPL code is imported.** `imageio-ffmpeg` (which ships a GPL ffmpeg executable) is deliberately
left out of the image; it is only needed by movement's video helpers, which this app does not use.

| Component | Authors / copyright holder | What | Licence | Source |
|---|---|---|---|---|
| movement 0.17.0 | Nikoloz Sirmpilatze, Chang Huan Lo, Sofía Miñano, Brandon D. Peri, Dhruv Sharma, Laura Porta, Iván Varela, Adam L. Tyson and contributors (SWC/GCNU Neuroinformatics Unit) | Loading DeepLabCut / SLEAP / Lightning Pose files, confidence filter, interpolation, smoothing, speed | BSD-3-Clause | https://github.com/neuroinformatics-unit/movement |
| pynwb 4.2.0, hdmf 6.2.0 | Neurodata Without Borders contributors (Lawrence Berkeley National Laboratory et al.) | Reading and writing NWB | BSD-3-Clause | https://github.com/NeurodataWithoutBorders/pynwb |
| ndx-pose 0.4.0 | Ryan Ly, Ben Dichter, Alexander Mathis, Liezl Maree, Chris Brozdowski, Heberto Mayorquin, Talmo Pereira, Elizabeth Berrigan, Paul Adkisson, Alessandra Trapani | NWB pose extension | BSD-3-Clause | https://github.com/rly/ndx-pose |
| ndx-multisubjects 0.1.1 | NWB community | Dependency of movement | BSD-3-Clause | PyPI |
| sleap-io 0.9.5 | Talmo Lab (Salk Institute) | SLEAP file reading (through movement) | BSD-3-Clause | https://github.com/talmolab/sleap-io |
| NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.1 | NumPy, SciPy, pandas developers | Arrays, statistics (Welch / Mann-Whitney), tables | BSD-3-Clause (NumPy: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0) | numpy.org, scipy.org, pandas.pydata.org |
| xarray 2026.9.0 | xarray developers | movement's data model | Apache-2.0 | https://github.com/pydata/xarray |
| pyarrow 26.0.1 | Apache Software Foundation | Parquet output | Apache-2.0 | https://arrow.apache.org |
| h5py 3.16.0, tables 3.11.1 (PyTables), blosc2 | h5py, PyTables, Blosc developers | HDF5 (NWB, DeepLabCut .h5) | BSD-3-Clause | h5py.org, pytables.org |
| Matplotlib 3.11.2 | Matplotlib Development Team | PNG figures | Matplotlib licence (PSF-based) | https://matplotlib.org |
| opencv-python-headless 5.0.0.93 | OpenCV team | Dependency of movement's video helpers (not used here); bundles FFmpeg libraries under LGPL-2.1 and OpenBLAS / libgfortran (GPL-3.0 with GCC runtime exception) as shared libraries | Apache-2.0 (bundled libraries under their own licences) | https://github.com/opencv/opencv-python |
| skia-python 144.1 | Kota Yamaguchi et al. | Dependency of sleap-io (rendering; not used here) | BSD-3-Clause | https://github.com/kyamagu/skia-python |
| numba 0.68.1, llvmlite 0.50.0, numbagg, flox, Bottleneck, numexpr | respective developers | Dependencies of xarray / movement | BSD-2/3-Clause, MIT, Apache-2.0 WITH LLVM-exception (llvmlite) | PyPI |
| netCDF4, cftime, h5netcdf, Cartopy, pyproj, pyshp, shapely, pydap, nc-time-axis, zarr, numcodecs | respective developers | Optional I/O and plotting dependencies of xarray / movement (not used here) | MIT, BSD-3-Clause, Apache-2.0 | PyPI |
| napari-video, pyvideoreader, qt-niu, QtPy, superqt, ImageIO | respective developers | Dependencies of movement's GUI / video helpers (no Qt binaries installed; not used here) | MIT, BSD-2/3-Clause | PyPI |
| certifi, tqdm, orjson | respective developers | Dependencies | MPL-2.0 (tqdm: MPL-2.0 AND MIT; orjson: MPL-2.0 AND (Apache-2.0 OR MIT)); shipped unmodified | PyPI |
| colorcet 3.2.1 | Holoviz | Dependency (colour maps) | CC-BY-4.0 | https://github.com/holoviz/colorcet |
| Other transitive dependencies (aiohttp, attrs, click, httpx, jsonschema, Jinja2, loguru, pydantic, PyYAML, requests, rich, typer, ...) | respective developers | Versions in `requirements.txt` | MIT / BSD / Apache-2.0 / PSF-2.0 / ISC / MIT-CMU (Pillow) | PyPI |
| python:3.12.16-slim-bookworm | Python Software Foundation, Debian | Base image | PSF-2.0 (Python); Debian packages under their own free licences | https://hub.docker.com/_/python |

## Not bundled

- **DeepLabCut, SLEAP, Lightning Pose**: only their output files are read (through movement); the
  tools themselves are not in the image. SLEAP `.slp` and analysis `.h5` files are read by movement
  through **sleap-io** (BSD-3-Clause). **sleap-nn** (SLEAP's new PyTorch backend) is deliberately not
  installed: its LICENSE file says GPL-3.0 while its package metadata says BSD-3-Clause (checked
  2026-11-09), and this app does not need it to read SLEAP files.
- **Demo data** (not in the image or the repository): DANDI:002047 (Plusnin, Pospelov et al.,
  doi:10.48324/dandi.002047/0.261106.0903), CC-BY-4.0.
