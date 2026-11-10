FROM python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /processor

# The complete pinned dependency set (requirements.in resolved in this base image; wheels only).
COPY requirements.txt /processor/requirements.txt
RUN pip install --no-deps --only-binary=:all: -r /processor/requirements.txt \
    && find /usr/local/lib/python3.12/site-packages -name '*.pyc' -delete \
    && find /usr/local/lib/python3.12/site-packages -type d -name tests -prune -exec rm -rf {} +

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    USER=app \
    LOGNAME=app \
    MPLBACKEND=Agg

# Check that the stack imports (no network at run time: movement, pynwb and ndx-pose need nothing
# else). Nothing is baked into /tmp or HOME: on Lambda both are empty and only INPUT_DIR, OUTPUT_DIR,
# HOME and TMPDIR are writable; processor/common.py points matplotlib / numba / XDG caches at $TMPDIR.
RUN MPLCONFIGDIR=/dev/shm/mpl HOME=/dev/shm python -c "import matplotlib.pyplot, movement.io.load_poses, movement.filtering, movement.kinematics, pynwb, ndx_pose, pyarrow.parquet"

# Runs as uid 1000 on ECS and as an arbitrary unprivileged user on Lambda; nothing here needs root or a
# particular uid (USER=app above answers getpass).
RUN useradd --uid 1000 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin processor
USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
