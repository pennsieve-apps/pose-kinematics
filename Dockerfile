FROM python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /processor

# The complete pinned dependency set (requirements.in resolved in this base image; wheels only),
# including the AWS Lambda runtime interface client: the image runs as a container or as a Lambda.
COPY requirements.txt /processor/requirements.txt
RUN pip install --no-deps --only-binary=:all: -r /processor/requirements.txt \
    && find /usr/local/lib/python3.12/site-packages -name '*.pyc' -delete \
    && find /usr/local/lib/python3.12/site-packages -type d -name tests -prune -exec rm -rf {} +

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    XDG_CACHE_HOME=/tmp/cache

# Build matplotlib's font cache and import the stack once, so a run needs no writes outside /tmp and
# no network (movement, pynwb and ndx-pose load everything they need from the installed packages).
RUN python -c "import matplotlib.pyplot, movement.io.load_poses, movement.filtering, movement.kinematics, pynwb, ndx_pose, pyarrow.parquet" \
    && mkdir -p /tmp/matplotlib /tmp/numba /tmp/cache /tmp/.movement && chmod -R a+rwX /tmp/matplotlib /tmp/numba /tmp/cache /tmp/.movement

# Runs as uid 1000 on the platform; nothing here needs root.
RUN useradd --uid 1000 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin processor
USER 1000

# Runs main() once as a container, or serves processor.handler as a Lambda function (processor/entry.py).
ENTRYPOINT ["python3", "-m", "processor.entry"]
